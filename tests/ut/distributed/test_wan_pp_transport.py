# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small CPU/Gloo experiments for the model-independent WAN PP transport.

Run with ``pytest --noconftest``: these tests exercise actual process-group
communication and own their dependencies, without the NPU model-runner mocks.
They do not establish HCCL/NPU event correctness or WAN performance.
"""

import multiprocessing as mp
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_ascend.distributed.pp_transport import PipelineTransportError, WanPPConfig, WanPPTransport
from vllm_ascend.distributed.pp_transport.buffers import DeviceRuntime

_PROCESS_TIMEOUT_S = 30
_SIGNAL_TIMEOUT_S = 10
_SLOT_BYTES = 4096


def _config(**overrides):
    values = {
        "enabled": True,
        "num_slots": 1,
        "slot_bytes": _SLOT_BYTES,
        "max_inflight_bytes": _SLOT_BYTES,
        "timeout_s": 10,
        "max_metadata_bytes": 1024,
    }
    values.update(overrides)
    return WanPPConfig(**values)


def _metadata_fifo(rank, transport, signals):
    receive_returned, source_reused = signals[:2]
    if rank == 0:
        # This deliberately requires receive() to return before metadata exists.
        assert receive_returned.wait(_SIGNAL_TIMEOUT_S), "receive blocked before publication"
        source = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        transport.send(
            {
                "hidden_states": source[:, ::2],
                "residual": None,
                "empty": torch.empty((0, 3), dtype=torch.int64),
                "step": 0,
            },
            batch_id="first",
        )
        source.fill_(-1)
        source_reused.set()
        for step in range(1, 3):
            transport.send({"hidden_states": torch.full((4, 4), float(step)), "step": step}, batch_id=f"batch-{step}")
        transport.drain()
    else:
        ticket = transport.receive()
        receive_returned.set()
        assert source_reused.wait(_SIGNAL_TIMEOUT_S)
        payload = ticket.wait()
        torch.testing.assert_close(
            payload["hidden_states"], torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
        )
        assert payload["residual"] is None
        assert payload["empty"].shape == (0, 3)
        assert payload["empty"].dtype == torch.int64
        assert payload["step"] == 0
        transport.release(ticket)
        for step in range(1, 3):
            ticket = transport.receive()
            payload = ticket.wait()
            torch.testing.assert_close(payload["hidden_states"], torch.full((4, 4), float(step)))
            assert payload["step"] == step
            transport.release(ticket)


def _consumer_credit(rank, transport, signals):
    first_visible, second_attempted, second_published = signals[:3]
    if rank == 0:
        transport.send({"hidden_states": torch.tensor([1.0])}, batch_id="first")
        assert first_visible.wait(_SIGNAL_TIMEOUT_S)
        deadline = time.monotonic() + _SIGNAL_TIMEOUT_S
        while transport.stats()["sent_frames"] != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        stats = transport.stats()
        assert stats["sent_frames"] == 1, stats
        assert stats["remote_inflight_frames"] == 1, stats
        second_attempted.set()
        transport.send({"hidden_states": torch.tensor([2.0])}, batch_id="second")
        second_published.set()
        transport.drain()
    else:
        first = transport.receive()
        torch.testing.assert_close(first.wait()["hidden_states"], torch.tensor([1.0]))
        first_visible.set()
        assert second_attempted.wait(_SIGNAL_TIMEOUT_S)
        # Remote readability is insufficient: the one-slot pool is still owned
        # by this consumer, so the next publication must remain blocked.
        assert not second_published.wait(0.2), "credit was returned before consumer release"
        transport.release(first)
        second = transport.receive()
        torch.testing.assert_close(second.wait()["hidden_states"], torch.tensor([2.0]))
        transport.release(second)
        assert second_published.wait(_SIGNAL_TIMEOUT_S), "consumer release did not restore credit"


def _cancel_reservation(rank, transport, signals):
    if rank == 0:
        reservation = transport.reserve(expected_bytes=16)
        reservation.cancel()
        stats = transport.stats()
        for name in ("reserved_send_frames", "reserved_send_bytes", "remote_inflight_frames", "remote_inflight_bytes"):
            assert stats[name] == 0, stats
        # Cancellation before publication must restore the only slot and must
        # not advance the peer's expected sequence or emit a partial frame.
        transport.send({"hidden_states": torch.arange(4, dtype=torch.float32)}, batch_id="after-cancel")
        transport.drain()
    else:
        ticket = transport.receive()
        torch.testing.assert_close(ticket.wait()["hidden_states"], torch.arange(4, dtype=torch.float32))
        transport.release(ticket)


def _run_rank(rank, init_method, case, signals):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    transport = None
    try:
        # Distinct CPU/control and data groups reflect the production topology.
        cpu_group = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=15))
        group = SimpleNamespace(
            device=torch.device("cpu"),
            cpu_group=cpu_group,
            device_group=dist.group.WORLD,
            ranks=[0, 1],
            rank_in_group=rank,
            world_size=2,
            is_first_rank=rank == 0,
            is_last_rank=rank == 1,
        )
        transport = WanPPTransport(group, _config())
        {
            "metadata_fifo": _metadata_fifo,
            "consumer_credit": _consumer_credit,
            "cancel_reservation": _cancel_reservation,
        }[case](rank, transport, signals)
        dist.barrier()
        transport.close()
        transport = None
        dist.destroy_process_group(cpu_group)
    finally:
        # A failed assertion is reported by the child exit status. Avoid hiding
        # it behind another potentially blocking drain/close during cleanup.
        if transport is None:
            dist.destroy_process_group()


@pytest.mark.parametrize("case", ["metadata_fifo", "consumer_credit", "cancel_reservation"])
def test_wan_pp_two_process_lifecycle(tmp_path, case):
    context = mp.get_context("spawn")
    signals = [context.Event() for _ in range(3)]
    _launch_ranks(tmp_path, context, _run_rank, 2, case, signals)


def _launch_ranks(tmp_path, context, target, world_size, *args):
    init_method = (tmp_path / "gloo-init").as_uri()
    processes = [context.Process(target=target, args=(rank, init_method, *args)) for rank in range(world_size)]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + _PROCESS_TIMEOUT_S
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                break
            if time.monotonic() >= deadline:
                pytest.fail(f"WAN PP {target.__name__} exceeded {_PROCESS_TIMEOUT_S}s")
            time.sleep(0.02)
        assert [process.exitcode for process in processes] == [0] * world_size
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=5)


def _tp_payload(step):
    values = torch.arange(8, dtype=torch.float32) + step
    return {
        "bf16": values.to(torch.bfloat16).reshape(2, 4),
        "float8": values.to(torch.float8_e4m3fn),
        "indices": torch.arange(8, dtype=torch.int64) + step,
        "noncontiguous": (torch.arange(16, dtype=torch.float32) + step).reshape(2, 8)[:, ::2],
        "odd": torch.arange(3, dtype=torch.int32) + step,
        "empty": torch.empty((0, 2), dtype=torch.float16),
    }


def _run_pp_tp_rank(rank, init_method):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=15)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=4, timeout=timeout)
    groups = []
    transport = None
    try:
        pp_ranks = [[0, 2], [1, 3]]
        tp_ranks = [[0, 1], [2, 3]]
        # All ranks create every group in the same order.
        for ranks in pp_ranks + tp_ranks:
            groups.append(dist.new_group(ranks, backend="gloo", timeout=timeout))
        pp_index = rank % 2
        stage_index = rank // 2
        group = SimpleNamespace(
            device=torch.device("cpu"),
            cpu_group=groups[pp_index],
            device_group=groups[pp_index],
            ranks=pp_ranks[pp_index],
            rank_in_group=stage_index,
            world_size=2,
            is_first_rank=stage_index == 0,
            is_last_rank=stage_index == 1,
        )
        tp_group = SimpleNamespace(
            world_size=2,
            rank_in_group=pp_index,
            cpu_group=groups[2 + stage_index],
            device_group=groups[2 + stage_index],
        )
        transport = WanPPTransport(group, _config(max_metadata_bytes=4096), all_gather_group=tp_group)
        assert transport.reserved_memory_bytes == _SLOT_BYTES * (1 if stage_index == 0 else 2)
        pointers = None
        for step in range(2):
            if stage_index == 0:
                transport.send(_tp_payload(step), batch_id=step)
            else:
                ticket = transport.receive()
                result = ticket.wait()
                expected = _tp_payload(step)
                for key, tensor in expected.items():
                    actual = result[key]
                    assert actual.dtype == tensor.dtype
                    assert actual.shape == tensor.shape
                    # Compare the actual bit pattern, including float8/BF16.
                    assert torch.equal(actual.contiguous().view(torch.uint8), tensor.contiguous().view(torch.uint8))
                current_pointers = {key: result[key].data_ptr() for key in ("bf16", "float8", "indices")}
                if pointers is not None:
                    assert current_pointers == pointers, "TP gather allocated a new output instead of reusing its slot"
                pointers = current_pointers
                transport.release(ticket)
        if stage_index == 0:
            transport.drain()
        dist.barrier()
        transport.close()
        transport = None
        for group in groups:
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)
    finally:
        if transport is None:
            dist.destroy_process_group()


def test_wan_pp_tp_gather_preserves_dtype_bits_and_reuses_storage(tmp_path):
    _launch_ranks(tmp_path, mp.get_context("spawn"), _run_pp_tp_rank, 4)


def _run_partial_publication_failure(_rank, _init_method):
    # No process group is needed: this backend represents a peer that received
    # part of a header before the sender learned that its channel had failed.
    # Run in a child so the deliberately failed daemon owners are isolated.
    failed = threading.Event()
    stop_backend = threading.Event()
    config = _config()

    class PartialPublishBackend:
        def __init__(self):
            self.headers = []
            self.greeted = False

        def recv_ack(self, src):
            if not self.greeted:
                self.greeted = True
                return {
                    "kind": "hello",
                    "version": 2,
                    "epoch": "test-receiver",
                    "gather_size": 1,
                    "device_type": "cpu",
                    "num_slots": config.num_slots,
                    "slot_bytes": config.slot_bytes,
                    "max_inflight_bytes": config.max_inflight_bytes,
                    "max_metadata_bytes": config.max_metadata_bytes,
                    "max_fragment_bytes": config.slot_bytes,
                }
            assert stop_backend.wait(_SIGNAL_TIMEOUT_S)
            raise OSError("test backend stopped")

        def publish(self, header, dst):
            self.headers.append(header)
            raise OSError("peer accepted header bytes before publication failed")

        def submit_send(self, buffer, dst):
            pytest.fail("data submission must not continue after partial publication failure")

    backend = PartialPublishBackend()
    group = SimpleNamespace(device=torch.device("cpu"), ranks=[0, 1], rank_in_group=0, world_size=2)
    transport = WanPPTransport(group, config, backend=backend, on_fatal=lambda error: failed.set())
    expected = torch.arange(4, dtype=torch.float32)
    source = expected.clone()
    try:
        # send() owns a snapshot before returning; backend failure is delivered
        # asynchronously by the publication owner through on_fatal/check_error.
        transport.send({"hidden_states": source}, batch_id="partial")
        source.fill_(-1)
        assert failed.wait(_SIGNAL_TIMEOUT_S)
        stats = transport.stats()
        assert stats["failed"]
        assert stats["reserved_send_frames"] == stats["remote_inflight_frames"] == 1
        assert stats["reserved_send_bytes"] == stats["remote_inflight_bytes"] == expected.nbytes
        assert stats["published_frames"] == stats["sent_frames"] == 0
        assert len(backend.headers) == 1
        assert backend.headers[0]["sequence"] == 0
        assert backend.headers[0]["batch_id"] == "partial"
        with pytest.raises(PipelineTransportError):
            transport.reserve(expected.nbytes)
        with pytest.raises(PipelineTransportError):
            transport.close()
        # Neither the failed close nor the failed admission refunds uncertain
        # ownership; the retained pool slot still contains the original data.
        assert transport.reserved_memory_bytes == _SLOT_BYTES
        pool = transport._send_pool
        with pytest.raises(RuntimeError, match="no available slot"):
            pool.acquire()
        torch.testing.assert_close(pool.buffer(0, expected.nbytes).view(torch.float32), expected)
        assert transport.stats()["remote_inflight_frames"] == 1
    finally:
        stop_backend.set()


def test_partial_publication_failure_retains_buffers_and_credit(tmp_path):
    _launch_ranks(tmp_path, mp.get_context("spawn"), _run_partial_publication_failure, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_slots": 0},
        {"slot_bytes": 0},
        {"max_inflight_bytes": _SLOT_BYTES - 1},
        {"timeout_s": 0},
    ],
)
def test_wan_pp_config_rejects_invalid_capacity(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_wan_pp_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown_setting"):
        _config(unknown_setting=True)


@pytest.mark.parametrize("pp_size,backend", [(1, "mp"), (2, "external_launcher")])
def test_wan_pp_config_requires_supported_parallel_path(pp_size, backend):
    parallel_config = SimpleNamespace(pipeline_parallel_size=pp_size, distributed_executor_backend=backend)
    with pytest.raises(ValueError):
        _config().validate_parallel_config(parallel_config)
    _config(enabled=False).validate_parallel_config(parallel_config)


def test_device_completion_waits_for_event_after_work_wait(monkeypatch):
    runtime = DeviceRuntime("cpu")
    work_waited = threading.Event()
    event_waited = threading.Event()
    device_completed = threading.Event()
    finished = threading.Event()
    errors = []

    class Work:
        def wait(self):
            work_waited.set()

    class Event:
        def synchronize(self):
            event_waited.set()
            assert device_completed.wait(_SIGNAL_TIMEOUT_S)

    monkeypatch.setattr(runtime, "record_event", Event)

    def finish():
        try:
            runtime.finish([Work()], None)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=finish, daemon=True)
    thread.start()
    try:
        assert work_waited.wait(_SIGNAL_TIMEOUT_S)
        assert event_waited.wait(_SIGNAL_TIMEOUT_S)
        assert not finished.wait(0.05), "Work.wait alone was treated as device completion"
    finally:
        device_completed.set()
        thread.join(timeout=_SIGNAL_TIMEOUT_S)
    assert not thread.is_alive()
    assert not errors
