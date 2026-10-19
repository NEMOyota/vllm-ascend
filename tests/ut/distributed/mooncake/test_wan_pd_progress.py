# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU byte-buffer experiments; no NPU or RDMA performance claims."""

import ctypes
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.config import WanPDConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import (
    ContinuationManifest,
    StateComponent,
    fingerprint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.progress import (
    BoundedReadProgress,
    TransferRange,
    transfer_chunks,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.staging import StagingPool


def _config(**kwargs):
    return WanPDConfig(
        **{
            "enabled": True,
            "max_pending_requests": 2,
            "max_active_requests": 2,
            "staging_slots": 2,
            "chunk_bytes": 8,
            "max_inflight_bytes": 16,
            "max_descriptors_per_chunk": 2,
            "timeout_s": 10,
            **kwargs,
        }
    )


class _Ticket:
    def __init__(self, local, remote, sizes):
        self.local, self.remote, self.sizes = local, remote, sizes
        self.state = "WAITING"
        self.error = "injected transport failure"

    def poll(self):
        return self.state

    def deliver(self):
        for local, remote, size in zip(self.local, self.remote, self.sizes, strict=True):
            ctypes.memmove(local, remote, size)
        self.state = "COMPLETED"


class _Engine:
    def __init__(self):
        self.tickets = []
        self.raise_after_submit = False

    def set_managed_batch_limit(self, limit):
        self.limit = limit

    def register_memory(self, address, size):
        return 0

    def batch_transfer_async_read_ticket(self, peer, local, remote, sizes):
        ticket = _Ticket(local, remote, sizes)
        self.tickets.append(ticket)
        if self.raise_after_submit:
            raise RuntimeError("Python wrapper failed after submission")
        return ticket


def _runtime(config=None, **kwargs):
    config = config or _config()
    source = torch.arange(32, dtype=torch.uint8)
    destination = torch.full_like(source, 255)
    engine = _Engine()
    pool = StagingPool(config, {"layer": destination})
    pool.register(engine)
    completed = []
    progress = BoundedReadProgress(
        config, engine, pool, lambda request, error: completed.append((request, error)), **kwargs
    )
    return source, destination, engine, pool, completed, progress


def test_chunks_bound_bytes_descriptors_and_keep_peer_boundaries():
    chunks = list(
        transfer_chunks(
            [
                TransferRange("p0", 100, 200, 17),
                TransferRange("p0", 300, 400, 1),
                TransferRange("p0", 500, 600, 1),
                TransferRange("p1", 700, 800, 3),
            ],
            _config(),
        )
    )
    assert [item.nbytes for item in chunks] == [8, 8, 2, 1, 3]
    assert all(len(item.pieces) <= 2 and item.nbytes <= 8 for item in chunks)
    assert chunks[2].pieces[0].source == 116
    assert chunks[-1].peer == "p1"


def test_real_byte_copy_tail_window_and_pending_admission():
    source, target, engine, pool, completed, progress = _runtime()
    assert pool.reserved_memory_bytes == 16
    plan = lambda: [TransferRange("p", source.data_ptr(), target.data_ptr(), 19)]
    assert progress.enqueue("a", plan)
    assert progress.enqueue("b", lambda: [])
    assert not progress.enqueue("overflow", plan)
    progress.advance()
    assert completed == [("b", None)]
    progress.advance()
    assert len(engine.tickets) == 2
    assert progress.stats()["network_inflight_bytes"] == 16
    assert torch.all(target == 255)
    engine.tickets[0].deliver()
    progress.advance()
    assert torch.equal(target[:8], source[:8])
    assert len(engine.tickets) == 2  # source slot waits for copy event query
    engine.tickets[1].deliver()
    progress.advance()
    assert engine.tickets[-1].sizes == [3]
    engine.tickets[-1].deliver()
    progress.advance()
    progress.advance()
    assert completed == [("b", None), ("a", None)]
    assert torch.equal(target[:19], source[:19])
    assert torch.all(target[19:] == 255)
    assert not progress.enqueue("overflow", plan)  # completed but not delivered
    progress.retire("a")
    assert progress.enqueue("next", lambda: [])


def test_copy_event_gates_commit_and_source_egress_retires_at_network_completion():
    releases = []
    lease = SimpleNamespace(release=lambda: releases.append("released"))
    egress = SimpleNamespace(try_acquire=lambda *args: lease)
    source, target, engine, pool, completed, progress = _runtime(egress=egress)
    event = SimpleNamespace(ready=False)
    event.query = lambda: event.ready
    real_copy = pool.copy

    def deferred_copy(slot, pieces):
        real_copy(slot, pieces)
        return event

    pool.copy = deferred_copy
    progress.enqueue("a", lambda: [TransferRange("p", source.data_ptr(), target.data_ptr(), 8)])
    progress.advance()
    engine.tickets[0].deliver()
    progress.advance()
    assert releases == ["released"]
    assert not completed
    progress.advance()
    assert not completed
    event.ready = True
    progress.advance()
    assert completed == [("a", None)]


@pytest.mark.parametrize("failure", ["ticket", "submit_exception", "timeout"])
def test_late_dma_is_isolated_from_reused_decode_blocks(failure):
    now = [0]
    source, target, engine, pool, completed, progress = _runtime(clock=lambda: now[0])
    engine.raise_after_submit = failure == "submit_exception"
    progress.enqueue("a", lambda: [TransferRange("p", source.data_ptr(), target.data_ptr(), 8)])
    progress.advance()
    if failure == "ticket":
        engine.tickets[0].state = "FAILED_UNDRAINED"
    if failure == "timeout":
        now[0] = 11
    progress.advance()
    assert completed[0][0] == "a" and completed[0][1]
    assert progress.stats()["quarantined_slots"] == 1
    assert progress.stats()["network_inflight_bytes"] == 8
    target.fill_(77)  # scheduler safely reuses failed D allocation
    engine.tickets[0].deliver()  # old DMA eventually arrives in staging
    assert torch.all(target == 77)
    assert pool.acquire() == 1
    assert pool.acquire() is None


def test_copy_failure_never_reports_decode_allocation_reusable():
    source, target, engine, pool, completed, progress = _runtime()
    progress.enqueue("a", lambda: [TransferRange("p", source.data_ptr(), target.data_ptr(), 8)])
    progress.advance()
    engine.tickets[0].deliver()

    def broken_copy(*args):
        raise RuntimeError("device copy outcome unknown")

    pool.copy = broken_copy
    with pytest.raises(RuntimeError, match="device copy outcome unknown"):
        progress.advance()
    assert not completed
    assert progress.fatal_error is not None
    assert pool.acquire() == 1
    assert pool.acquire() is None


@pytest.fixture
def pull_worker_class(monkeypatch):
    # Execute the real worker module and address mapper. Replace only NPU
    # bootstrap imports: no device/engine initialization runs in this test.
    existing = set(sys.modules)
    stubs = {
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker": {"MooncakeBaseConnectorWorker": object},
        "vllm_ascend.utils": {"get_kv_cache_tensor_layers": lambda value: []},
        "vllm_ascend.distributed.parallel_state": {"get_p_tp_group": lambda: None},
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[4] / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py"
    spec = importlib.util.spec_from_file_location("wan_pd_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module.MooncakePullRecvingThread
    for name in set(sys.modules) - existing:
        if name.startswith("vllm_ascend.") and name not in stubs:
            sys.modules.pop(name, None)


@pytest.mark.parametrize("tamper", ["none", "source_table", "missing_destination"])
def test_real_pull_planner_validates_manifest_before_address_submission(pull_worker_class, tamper):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    config = _config()
    sources = [torch.arange(16, dtype=torch.uint8).reshape(4, 1, 4, 1) for _ in range(2)]
    destinations = [torch.zeros_like(tensor) for tensor in sources]
    local = SimpleNamespace(
        layer_names=["layer"],
        layer_block_sizes=[4],
        group_indices=[0],
        block_strides=[[4, 4]],
        block_lens=[[4, 4]],
        block_shapes=[[(1, 4, 1), (1, 4, 1)]],
        block_size_scales=[[1, 1]],
        state_kinds=["full_attention"],
        state_dtypes=[["torch.uint8", "torch.uint8"]],
        num_blocks=4,
        kv_caches_base_addr=[[tensor.data_ptr() for tensor in destinations]],
    )
    pp = SimpleNamespace(**{key: value for key, value in vars(local).items() if key != "kv_caches_base_addr"})
    endpoint = SimpleNamespace(
        local_ip="p", te_rpc_port=1, kv_caches_base_addr=[[tensor.data_ptr() for tensor in sources]]
    )
    pp.metadata_by_pcp_rank = {0: SimpleNamespace(metadata_by_tp_rank={0: endpoint})}
    remote = SimpleNamespace(
        engine_epoch="epoch",
        namespace="namespace",
        layout_fingerprint="layout",
        tp_size=1,
        pcp_size=1,
        dcp_size=1,
        use_kv_pp=False,
        metadata_by_pp_rank={0: pp},
    )
    request = SimpleNamespace(
        remote_block_ids=([1, 2],),
        local_full_block_ids=([1, 2],),
        local_block_ids=([1, 2],),
        remote_engine_id="producer",
        remote_request_id="request-p",
        remote_num_prompt_tokens=7,
        remote_host="p",
        remote_port=2,
        num_computed_tokens=0,
        local_num_prompt_tokens=7,
    )
    plan_hash = fingerprint({key: value for key, value in vars(request).items() if key.startswith("remote_")})
    components = tuple(
        StateComponent(f"layer:{index}", "layer", 0, "full_attention", "torch.uint8", (1, 4, 1), 0, 7)
        for index in range(2)
    )
    request.continuation_manifest = ContinuationManifest(
        "request-p",
        "attempt",
        "epoch",
        "namespace",
        "layout",
        "input",
        7,
        7,
        0,
        components,
        transfer_plan_fingerprint=plan_hash,
    ).to_wire(config)
    worker = pull_worker_class.__new__(pull_worker_class)
    worker.wan_pd_config, worker._wan_namespace, worker.local_metadata = config, "namespace", local
    for key, value in vars(local).items():
        setattr(worker, key, value)
    worker.kv_cache_specs = [FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.uint8)]
    worker.spec_indices = [0]
    worker.pcp_size = worker.dcp_size = worker.tp_size = worker.num_key_value_heads = 1
    worker.pcp_rank = worker.dcp_rank = worker.tp_rank = worker.num_speculative_tokens = 0
    cache_key = ("producer", "epoch", "p", 2)
    worker.remote_layer_index_pairs = {cache_key: {0: [(0, 0)]}}
    worker.remote_tp_rank_groups = {cache_key: {0: {(0, 0): [[0]]}}}
    worker._get_remote_metadata = lambda *args, **kwargs: remote
    worker.staging_pool = StagingPool(config, {"layer": destinations})
    if tamper == "source_table":
        request.remote_block_ids = ([2, 3],)
        with pytest.raises(ValueError, match="block table or source endpoint changed"):
            worker._plan_wan_request("request-d", request)
    elif tamper == "missing_destination":
        request.local_block_ids = ([],)
        request.local_full_block_ids = ([],)
        with pytest.raises(ValueError, match="block tables do not cover"):
            worker._plan_wan_request("request-d", request)
    else:
        ranges = worker._plan_wan_request("request-d", request)
        assert len(ranges) == 2
        assert [item.nbytes for item in ranges] == [8, 8]
        assert [item.destination for item in ranges] == [tensor.data_ptr() + 4 for tensor in destinations]
        assert [item.source for item in ranges] == [tensor.data_ptr() + 4 for tensor in sources]
