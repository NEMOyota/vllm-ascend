# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests of the worker's model-independent PP adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.worker.pp_transport import execute_with_wan_pp


@pytest.mark.parametrize("use_v2", [False, True])
def test_reserves_before_compute_and_releases_after_publish(use_v2):
    transport = MagicMock()
    worker = SimpleNamespace(
        use_v2_model_runner=use_v2,
        profiler=None,
        _wan_pp_batch_id=0,
        _wan_pp_transport=transport,
    )
    payload = {"hidden_states": torch.arange(4)}
    transport.receive.return_value.wait.return_value = payload
    connector_output = KVConnectorOutput(finished_sending={"request-1"})
    output = IntermediateTensors(payload)
    output.kv_connector_output = connector_output

    def execute(scheduled, inputs):
        transport.reserve.assert_called_once_with()
        transport.release.assert_not_called()
        assert torch.equal(inputs["hidden_states"], payload["hidden_states"])
        assert torch.equal(inputs[:2]["hidden_states"], payload["hidden_states"][:2])
        return output

    def release(ticket):
        transport.reserve.return_value.publish.assert_called_once_with(payload, batch_id=0)
        assert ticket is transport.receive.return_value

    worker.model_runner = SimpleNamespace(execute_model=execute, is_pooling_model=False)
    transport.release.side_effect = release
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    result = execute_with_wan_pp(worker, SimpleNamespace(total_num_scheduled_tokens=4), pp_group)

    assert worker._wan_pp_batch_id == 1
    transport.release.assert_called_once()
    transport.reserve.return_value.cancel.assert_not_called()
    transport.fail.assert_not_called()
    if use_v2:
        assert result is None
    else:
        assert result.kv_connector_output is connector_output


@pytest.mark.parametrize("failure_stage", ["compute", "publish"])
def test_failure_does_not_release_receive_or_cancel_a_published_frame(failure_stage):
    transport = MagicMock()
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        profiler=None,
        _wan_pp_batch_id=0,
        _wan_pp_transport=transport,
        model_runner=MagicMock(is_pooling_model=False),
    )
    worker.model_runner.execute_model.return_value = IntermediateTensors({"hidden_states": torch.zeros(4)})
    error = RuntimeError("injected failure")
    if failure_stage == "compute":
        worker.model_runner.execute_model.side_effect = error
    else:
        transport.reserve.return_value.publish.side_effect = error

    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    with pytest.raises(RuntimeError, match="injected failure"):
        execute_with_wan_pp(worker, SimpleNamespace(total_num_scheduled_tokens=4), pp_group)

    transport.fail.assert_called_once_with(error)
    transport.release.assert_not_called()
    assert worker._wan_pp_batch_id == 0
    assert transport.reserve.return_value.cancel.call_count == (failure_stage == "compute")


def test_idle_step_progresses_model_runner_without_allocating_a_pp_frame():
    transport = MagicMock()
    runner = MagicMock(is_pooling_model=False)
    runner.execute_model.return_value = None
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        profiler=None,
        _wan_pp_batch_id=0,
        _wan_pp_transport=transport,
        model_runner=runner,
    )
    scheduled = SimpleNamespace(total_num_scheduled_tokens=0)
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)

    assert execute_with_wan_pp(worker, scheduled, pp_group) is None
    runner.execute_model.assert_called_once_with(scheduled, None)
    transport.receive.assert_not_called()
    transport.reserve.assert_not_called()
    transport.release.assert_not_called()
    assert worker._wan_pp_batch_id == 0
