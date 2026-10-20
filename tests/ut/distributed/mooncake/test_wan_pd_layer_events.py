# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU experiments of production source events, READ gates and sealing."""

import ctypes
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.commit import SourceLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.config import WanPDConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.layer_events import (
    LayerEventPublisher,
    ReadinessGate,
)
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
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.publication import (
    GET_READY_MSG,
    UPDATE_READY_MSG,
    SourcePublicationManager,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.staging import StagingPool


class _Ticket:
    def __init__(self, local, remote, sizes):
        self.local, self.remote, self.sizes = local, remote, sizes
        self.state = "WAITING"

    def poll(self):
        return self.state

    def deliver(self):
        for local, remote, size in zip(self.local, self.remote, self.sizes, strict=True):
            ctypes.memmove(local, remote, size)
        self.state = "COMPLETED"


class _Engine:
    def __init__(self):
        self.tickets = []

    def set_managed_batch_limit(self, limit):
        self.limit = limit

    def register_memory(self, address, size):
        return 0

    def batch_transfer_async_read_ticket(self, peer, local, remote, sizes):
        ticket = _Ticket(local, remote, sizes)
        self.tickets.append(ticket)
        return ticket


def _setup():
    """Join the real publication, event, readiness and staging components.

    Controlled events and READ tickets model completion ordering while CPU
    tensors keep payload copy assertions observable without an NPU.
    """
    config = WanPDConfig(
        enabled=True,
        receiver_first=True,
        layerwise_publication=True,
        max_retained_source_requests=1,
        max_pending_requests=2,
        max_active_requests=2,
        chunk_bytes=8,
        staging_slots=2,
        max_inflight_bytes=16,
    )
    params = dict(
        remote_block_ids=[[1]],
        remote_engine_id="producer",
        remote_request_id="request",
        remote_num_prompt_tokens=4,
        remote_host="127.0.0.1",
        remote_port=5000,
    )
    manifest = ContinuationManifest(
        request_id="request",
        attempt_id="attempt",
        engine_epoch="epoch",
        namespace="weights",
        layout_fingerprint="layout",
        input_fingerprint="input",
        prompt_tokens=4,
        state_position=4,
        prefix_tokens=0,
        components=tuple(
            StateComponent(f"layer.{index}:0", f"layer.{index}", 0, "full_attention", "uint8", (8,), 0, 4)
            for index in range(2)
        ),
        transfer_plan_fingerprint=fingerprint(params),
        reservation_fingerprint="reservation",
        publication_mode="layerwise",
    )
    block = SimpleNamespace(block_id=1, ref_cnt=1)
    pool = SimpleNamespace(
        touch=lambda blocks: setattr(blocks[0], "ref_cnt", blocks[0].ref_cnt + 1),
        free_blocks=lambda blocks: setattr(blocks[0], "ref_cnt", blocks[0].ref_cnt - 1),
    )
    manager = SimpleNamespace(block_pool=pool, get_blocks=lambda _: SimpleNamespace(blocks=([block],)))
    ledger = SourceLedger(config, "epoch")
    catalog = SourcePublicationManager(config, ledger, manager)
    catalog.prepare("request", params, manifest, {"0:0:0": ["layer.0", "layer.1"]}, ticket_fingerprint="reservation")
    params["continuation_manifest"] = manifest.to_wire(config)

    class Exchange:
        closed = False

        def advance(self, payload):
            message = msgspec.msgpack.decode(payload)
            if message[0] == UPDATE_READY_MSG:
                try:
                    catalog.update_ready(*message[1:])
                except ValueError:
                    return b"REJECTED"
                return b"ACK"
            assert message[0] == GET_READY_MSG
            return msgspec.msgpack.encode(catalog.get_ready(*message[1:]))

        def close(self):
            self.closed = True

    events = []

    def record_event():
        event = SimpleNamespace(ready=False)
        event.query = lambda: event.ready
        events.append(event)
        return event

    publisher = LayerEventPublisher(
        config,
        "0:0:0",
        ["layer.0", "layer.1"],
        event_factory=record_event,
        exchange_factory=lambda *args: Exchange(),
    )
    publisher.begin({"request": params}, set())
    now = [0.0]
    gate = ReadinessGate(config, manifest, "127.0.0.1", 5000, exchange=Exchange(), clock=lambda: now[0])
    source = torch.arange(16, dtype=torch.uint8)
    target = torch.full_like(source, 255)
    engine = _Engine()
    staging = StagingPool(config, {"kv": target})
    staging.register(engine)
    completed = []
    progress = BoundedReadProgress(
        config, engine, staging, lambda request_id, error: completed.append((request_id, error))
    )
    ranges = [
        TransferRange("peer", source.data_ptr() + 8 * i, target.data_ptr() + 8 * i, 8, f"0:0:0/layer.{i}")
        for i in range(2)
    ]
    assert progress.enqueue("request", lambda: ranges, gate=gate)

    def advance():
        now[0] += 0.01
        publisher.advance()
        progress.advance()

    return SimpleNamespace(**locals())


def test_layer_event_starts_read_before_final_layer_but_decode_waits_for_seal():
    run = _setup()
    run.publisher.save_layer("layer.0")
    run.advance()
    assert not run.engine.tickets
    run.events[0].ready = True
    run.advance()
    assert len(run.engine.tickets) == 1  # layer 1 has not even run
    assert run.block.ref_cnt == 2
    run.engine.tickets[0].deliver()
    run.advance()
    run.advance()
    assert torch.equal(run.target[:8], run.source[:8])
    assert torch.all(run.target[8:] == 255)
    assert not run.completed

    run.publisher.save_layer("layer.1")
    run.publisher.finalize()
    run.events[1].ready = True
    run.advance()
    assert len(run.engine.tickets) == 2
    run.engine.tickets[1].deliver()
    run.advance()
    run.advance()
    assert torch.equal(run.target, run.source)
    assert not run.completed  # all bytes copied, final stream event still pending
    run.events[2].ready = True
    run.advance()
    assert not run.completed  # workers finalized, scheduler has not sealed
    run.catalog.seal("request")
    run.advance()
    assert run.completed == [("request", None)]
    assert run.gate.exchange.closed
    assert run.catalog.validate_commit("epoch", "attempt", run.manifest.digest, True)
    assert run.ledger.finish("epoch", "attempt", run.manifest.digest, True) == "request"
    assert run.catalog.retire("request")
    run.publisher.begin({}, {"request"})
    assert run.block.ref_cnt == 1


def test_source_abort_retains_pin_and_late_read_cannot_write_reused_decode_kv():
    run = _setup()
    run.publisher.save_layer("layer.0")
    run.events[0].ready = True
    run.advance()
    assert len(run.engine.tickets) == 1
    run.catalog.abort("request")
    run.publisher.save_layer("layer.1")
    run.events[1].ready = True
    run.advance()
    assert run.completed[0][0] == "request" and "failed" in run.completed[0][1]
    assert run.progress.stats()["quarantined_slots"] == 1
    assert run.block.ref_cnt == 2
    assert run.publisher.failure is None and run.publisher._entries["request"].rejected
    run.target.fill_(77)
    run.engine.tickets[0].deliver()
    assert torch.all(run.target == 77)


def test_layer_boundaries_and_empty_endpoint_do_not_publish_false_readiness():
    run = _setup()
    chunks = list(transfer_chunks(run.ranges, run.config))
    assert [chunk.layer_key for chunk in chunks] == ["0:0:0/layer.0", "0:0:0/layer.1"]
    run.publisher.save_layer("layer.0")
    with pytest.raises(ValueError, match="Repeated layer"):
        run.publisher.save_layer("layer.0")
    empty = LayerEventPublisher(run.config, "1:0:0", [], event_factory=lambda: pytest.fail("No KV event expected"))
    empty.begin({"request": run.params}, set())
    empty.save_layer("")
    empty.finalize()
    empty.advance()
    assert not empty._entries
