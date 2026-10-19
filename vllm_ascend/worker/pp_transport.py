# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pipeline transport adapter shared by Model Runner V1 and V2."""

import copy
from types import NoneType
from typing import TYPE_CHECKING, Any

from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm_ascend.distributed.pp_transport import ReceiveTicket


class WanPPIntermediateTensors(IntermediateTensors):
    """Wait for a receive on first tensor access, on the execution thread."""

    def __init__(self, ticket: "ReceiveTicket") -> None:
        self._ticket = ticket

    def __getattribute__(self, name: str):
        if name == "tensors":
            return object.__getattribute__(self, "_ticket").wait()
        return object.__getattribute__(self, name)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return IntermediateTensors({name: tensor[key] for name, tensor in self.tensors.items()})
        return self.tensors[key]


def execute_with_wan_pp(worker: Any, scheduler_output: Any, pp_group: Any):
    """Reserve, execute and publish one step on the worker's computation stream."""
    transport = worker._wan_pp_transport
    assert transport is not None
    transport.check_error()
    forward_pass = scheduler_output.total_num_scheduled_tokens > 0
    ticket = None
    reservation = None
    publish_attempted = False
    try:
        intermediate_tensors = None
        if forward_pass and not pp_group.is_first_rank:
            ticket = transport.receive(expected_batch_id=worker._wan_pp_batch_id)
            intermediate_tensors = WanPPIntermediateTensors(ticket)
        if forward_pass and not pp_group.is_last_rank:
            # Reserve before launching another forward. The output schema
            # is model-defined, so reserve a whole slot until publication.
            reservation = transport.reserve()

        if worker.profiler is not None:
            worker.profiler.step()
        output = worker.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if worker.use_v2_model_runner and worker.model_runner.is_pooling_model and output is None:
            output = worker.model_runner.pool()

        if isinstance(output, IntermediateTensors):
            if reservation is None:
                raise RuntimeError("A PP output requires an outbound reservation before execution.")
            publish_attempted = True
            reservation.publish(output.tensors, batch_id=worker._wan_pp_batch_id)
            reservation = None
        elif reservation is not None:
            reservation.cancel()
            reservation = None

        # The consumer event follows both input reads and any outbound
        # snapshot. Remote credit is returned only after that event.
        if ticket is not None:
            transport.release(ticket)
        if forward_pass:
            worker._wan_pp_batch_id += 1
    except Exception as exc:
        transport.fail(exc)
        if reservation is not None and not publish_attempted:
            reservation.cancel()
        raise

    if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
        return output
    assert isinstance(output, IntermediateTensors)
    if worker.use_v2_model_runner:
        return None
    kv_connector_output = output.kv_connector_output
    if not kv_connector_output:
        return None
    if not kv_connector_output.finished_sending and not kv_connector_output.finished_recving:
        return EMPTY_MODEL_RUNNER_OUTPUT
    result = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
    result.kv_connector_output = kv_connector_output
    return result
