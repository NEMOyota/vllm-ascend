# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Focused CPU/Gloo experiments for fragment windows and shared egress.

Run with ``pytest --noconftest``. These tests exercise real process groups and
the TCP admission service; they do not measure NPU or long-distance performance.
"""

import math
import multiprocessing as mp
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_ascend.distributed.pp_transport import WanPPConfig, WanPPTransport
from vllm_ascend.distributed.pp_transport.buffers import FrameLayout
from vllm_ascend.distributed.pp_transport.egress import EgressLimits
from vllm_ascend.distributed.pp_transport.egress_service import EgressBudgetServer

_PROCESS_TIMEOUT_S = 40
_SIGNAL_TIMEOUT_S = 10
_SLOT_BYTES = 4096
_FRAGMENT_BYTES = 128
_WINDOW_FRAGMENTS = 2
_WINDOW_BYTES = _FRAGMENT_BYTES * _WINDOW_FRAGMENTS


def _config(endpoint=None):
    joint = {
        "enabled": True,
        "compute_chunk_tokens": 32,
        "fragment_bytes": _FRAGMENT_BYTES,
        "window_fragments": _WINDOW_FRAGMENTS,
        "window_bytes": _WINDOW_BYTES,
    }
    if endpoint is not None:
        joint["egress_rank_endpoints"] = {0: endpoint, 2: endpoint}
    return WanPPConfig(
        enabled=True,
        num_slots=1,
        slot_bytes=_SLOT_BYTES,
        max_inflight_bytes=_SLOT_BYTES,
        timeout_s=_SIGNAL_TIMEOUT_S,
        max_metadata_bytes=4096,
        joint_config=joint,
    )


def _payload(step):
    return {
        "bf16": (torch.arange(173, dtype=torch.float32) + step).to(torch.bfloat16),
        "indices": torch.arange(31, dtype=torch.int64) + (1 << 54) + step,
        "float8": (torch.arange(49, dtype=torch.float32) % 8 + step).to(torch.float8_e4m3fn),
        "empty": torch.empty((0, 3), dtype=torch.float16),
        "step": step,
    }


def _assert_payload(actual, expected):
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert actual[name].dtype == value.dtype
            assert actual[name].shape == value.shape
            assert torch.equal(actual[name].view(torch.uint8), value.view(torch.uint8))
        else:
            assert actual[name] == value


def _assert_drained(transport, expected_fragments):
    stats = transport.stats()
    assert stats["network_inflight_bytes"] == stats["network_inflight_fragments"] == 0, stats
    assert 0 < stats["peak_network_bytes"] <= _WINDOW_BYTES, stats
    assert 0 < stats["peak_network_fragments"] <= _WINDOW_FRAGMENTS, stats
    assert stats["sent_fragments"] == stats["completed_fragments"] == expected_fragments, stats


def _group(ranks, rank, process_group):
    stage = ranks.index(rank)
    return SimpleNamespace(
        device=torch.device("cpu"),
        cpu_group=process_group,
        device_group=process_group,
        ranks=ranks,
        rank_in_group=stage,
        world_size=len(ranks),
        is_first_rank=stage == 0,
        is_last_rank=stage == len(ranks) - 1,
    )


def _fragment_window_rank(rank, init_method, source_reused):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=15)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timeout)
    control = dist.new_group([0, 1], backend="gloo", timeout=timeout)
    group = _group([0, 1], rank, control)
    group.device_group = dist.group.WORLD
    transport = WanPPTransport(group, _config())
    expected = _payload(0)
    wire_bytes = FrameLayout.from_tensors(expected).wire_bytes
    assert wire_bytes > _WINDOW_BYTES and wire_bytes % _FRAGMENT_BYTES != 0
    if rank == 0:
        source = _payload(0)
        transport.send(source, batch_id="fragmented")
        for tensor in source.values():
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
        source_reused.set()
        transport.send({"empty": torch.empty((0, 3)), "step": "empty"}, batch_id="empty")
        transport.drain()
        _assert_drained(transport, math.ceil(wire_bytes / _FRAGMENT_BYTES))
    else:
        ticket = transport.receive(expected_batch_id="fragmented")
        assert source_reused.wait(_SIGNAL_TIMEOUT_S), "producer did not return after snapshot"
        _assert_payload(ticket.wait(), expected)
        transport.release(ticket)
        empty = transport.receive(expected_batch_id="empty")
        result = empty.wait()
        assert result["empty"].shape == (0, 3)
        assert result["step"] == "empty"
        transport.release(empty)
    dist.barrier()
    transport.close()
    dist.destroy_process_group(control)
    dist.destroy_process_group()


def _shared_egress_rank(rank, init_method, endpoint):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=15)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=4, timeout=timeout)
    lanes = ([0, 1], [2, 3])
    groups = [dist.new_group(ranks, backend="gloo", timeout=timeout) for ranks in lanes]
    lane = rank // 2
    transport = WanPPTransport(_group(lanes[lane], rank, groups[lane]), _config(endpoint))
    # Both sending processes have connected to the same service before either
    # starts publishing frames larger than the shared physical-exit window.
    dist.barrier()
    total_fragments = 0
    for step in range(2):
        payload = _payload(lane * 2 + step)
        total_fragments += math.ceil(FrameLayout.from_tensors(payload).wire_bytes / _FRAGMENT_BYTES)
        if rank % 2 == 0:
            transport.send(payload, batch_id=step)
        else:
            ticket = transport.receive(expected_batch_id=step)
            _assert_payload(ticket.wait(), payload)
            transport.release(ticket)
    if rank % 2 == 0:
        transport.drain()
        _assert_drained(transport, total_fragments)
    dist.barrier()
    transport.close()
    for group in groups:
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)
    dist.destroy_process_group()


def _launch(tmp_path, context, target, world_size, *args):
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
                pytest.fail(f"{target.__name__} exceeded {_PROCESS_TIMEOUT_S}s")
            time.sleep(0.02)
        assert [process.exitcode for process in processes] == [0] * world_size
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=5)


def test_frame_larger_than_fragment_window_preserves_snapshot_and_tail(tmp_path):
    context = mp.get_context("spawn")
    _launch(tmp_path, context, _fragment_window_rank, 2, context.Event())


def test_two_pp_lanes_share_one_egress_budget(tmp_path):
    limits = EgressLimits(max_bytes=_WINDOW_BYTES, max_fragments=_WINDOW_FRAGMENTS)
    server = EgressBudgetServer(("127.0.0.1", 0), limits, socket_timeout_s=15)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        endpoint = f"127.0.0.1:{server.server_address[1]}"
        _launch(tmp_path, mp.get_context("spawn"), _shared_egress_rank, 4, endpoint)
        stats = server.arbiter.stats()
        expected_bytes = sum(FrameLayout.from_tensors(_payload(step)).wire_bytes for step in range(4))
        assert stats["granted_bytes"] == expected_bytes, stats
        assert stats["inflight_bytes"] == stats["inflight_fragments"] == 0, stats
        assert 0 < stats["peak_inflight_bytes"] <= limits.max_bytes, stats
        assert 0 < stats["peak_inflight_fragments"] <= limits.max_fragments, stats
        assert not stats["failed"], stats
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
