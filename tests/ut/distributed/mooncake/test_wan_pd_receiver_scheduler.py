# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Production scheduler handoff tests with CPU cache and control-plane doubles."""

import importlib.util
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.reservation import ReservationResult

_spec = importlib.util.spec_from_file_location(
    "wan_pd_scheduler_helpers", Path(__file__).with_name("test_wan_pd_scheduler.py")
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
modules = _shared.modules


def _scheduler(modules, monkeypatch, *, producer, dense=False, **options):
    """Reuse the production fixture, changing configuration before construction."""
    constructor = modules.scheduler.MooncakePullConnectorScheduler

    def configured(config, engine_id, cache):
        config.kv_transfer_config.kv_connector_extra_config["wan_pd"].update(
            receiver_first=True, reservation_callback_url="http://router.invalid/reserve", **options
        )
        config.model_config.enforce_eager = True
        config.scheduler_config = SimpleNamespace(async_scheduling=False)
        if dense:
            cache = replace(cache, kv_cache_groups=cache.kv_cache_groups[:1])
        return constructor(config, engine_id, cache)

    monkeypatch.setattr(modules.scheduler.BoundedReservationDispatcher, "start", lambda self: None)
    with monkeypatch.context() as context:
        context.setattr(modules.scheduler, "MooncakePullConnectorScheduler", configured)
        scheduler = _shared._scheduler(modules, monkeypatch, producer=producer)
    if producer:
        if dense:
            metadata = modules.scheduler.MooncakeTransferMetadata(
                engine_id="producer",
                te_rpc_port=5001,
                block_size=2,
                num_blocks=16,
                layer_names=["layer.0"],
                layer_block_sizes=[2],
                group_indices=[0],
                kv_caches_base_addr=[[1000]],
                block_strides=[[64]],
                block_lens=[[64]],
                block_shapes=[[(2, 8)]],
                block_size_scales=[[1]],
                local_ip="127.0.0.1",
                state_kinds=["full_attention"],
                state_dtypes=[["bfloat16"]],
            )
            scheduler._sending_thread = modules.scheduler.MooncakeSchedulerSendingThread(
                "127.0.0.1",
                5002,
                "producer",
                {0: metadata},
                1,
                1,
                1,
                1,
                False,
                threading.Event(),
                scheduler._source_ledger,
                scheduler._namespace,
                scheduler._publications,
            )
        else:
            scheduler._sending_thread.publications = scheduler._publications
    return scheduler


def _decode_request(request_id="decode"):
    return _shared._request(
        request_id,
        kv_transfer_params={
            "do_remote_prefill": True,
            "wan_pd_receiver_first": True,
            "wan_pd_rendezvous_id": "route-" + request_id,
        },
    )


def _blocks():
    # Include extra hybrid allocation slots. The scheduler must forward exactly
    # what D allocated, rather than re-trimming by the original prompt length.
    local, full = ([11, 12], [21, 22]), ([10, 11, 12], [20, 21, 22])
    return SimpleNamespace(get_unhashed_block_ids_all_groups=lambda: local, get_block_ids=lambda: full)


def test_receiver_allocates_hybrid_blocks_before_p_and_commits_exact_manifest(modules, monkeypatch):
    producer = _scheduler(modules, monkeypatch, producer=True)
    decoder = _scheduler(modules, monkeypatch, producer=False)
    request, blocks = _decode_request(), _blocks()
    calls = []

    def callback(payload, timeout, max_bytes):
        ticket = modules.scheduler.ReservationTicket.from_wire(payload["ticket"], decoder.wan_pd)
        assert request.request_id in decoder._awaiting_reservations
        assert ticket.allocation_fingerprint == modules.contracts.fingerprint(
            {"local": blocks.get_unhashed_block_ids_all_groups(), "full": blocks.get_block_ids()}
        )
        calls.append(ticket)
        source = _shared._request(kv_transfer_params={"do_remote_decode": True, "wan_pd_reservation": ticket.to_wire()})
        return {"transfer_params": _shared._publish(producer, modules, source)}

    decoder._reservations._callback = callback
    assert decoder.get_num_new_matched_tokens(request, 2) == (2, True)
    assert not calls and decoder._reservations.stats()["queued"] == 0
    decoder.update_state_after_alloc(request, blocks, 2)
    assert not calls and decoder._reservations.stats()["queued"] == 1
    assert decoder.build_connector_meta(None).requests == {}  # Callback is not KV readiness.
    ticket = decoder._awaiting_reservations[request.request_id][-1]
    response = decoder._reservations._callback({"ticket": ticket.to_wire()}, 1, decoder.wan_pd.max_manifest_bytes)
    result = ReservationResult(request.request_id, ticket, response["transfer_params"])
    completions = [result]
    monkeypatch.setattr(decoder._reservations, "poll_completed", lambda: [completions.pop()] if completions else [])
    metadata = decoder.build_connector_meta(None)
    received = metadata.requests[request.request_id]
    assert received.local_block_ids == blocks.get_unhashed_block_ids_all_groups()
    assert received.local_full_block_ids == blocks.get_block_ids()
    assert received.num_external_tokens == 2 and received.num_computed_tokens == 2
    manifest = modules.contracts.ContinuationManifest.from_wire(received.continuation_manifest)
    assert manifest.reservation_fingerprint == ticket.fingerprint
    assert len(manifest.components) == 2 and manifest.state_position == 4
    records = []
    decoder._commit_sender.send = records.append
    assert not decoder._commit_sender.progress()
    decoder.update_connector_output(modules.output(finished_recving={request.request_id}))
    assert decoder._commit_sender.progress()
    assert [(record.epoch, record.attempt_id, record.digest, record.committed) for record in records] == [
        (manifest.engine_epoch, manifest.attempt_id, manifest.digest, True)
    ]
    _shared._deliver_completion(modules, producer, records[0])
    output = modules.output()
    producer.update_connector_output(output)
    assert output.finished_sending == {manifest.request_id}
    assert producer._source_ledger.stats()["reserved"] == 0
    assert decoder._reservations.stats()["reserved"] == 0


def test_pending_callback_cancel_timeout_and_late_result_never_start_decode(modules, monkeypatch):
    start = modules.scheduler.BoundedReservationDispatcher.start
    decoder = _scheduler(modules, monkeypatch, producer=False)
    now, release = [0.0], threading.Event()
    started = {request_id: threading.Event() for request_id in ("cancel", "timeout")}

    def callback(payload, timeout, max_bytes):
        started[payload["ticket"]["request_id"]].set()
        assert release.wait(3)
        return {"transfer_params": {"late": True}}

    dispatcher = modules.scheduler.BoundedReservationDispatcher(
        decoder.wan_pd, "receiver-epoch", callback, clock=lambda: now[0]
    )
    decoder._reservations = dispatcher
    start(dispatcher)  # The sole test that executes real callback workers.
    try:
        requests = [_decode_request(request_id) for request_id in started]
        blocks = _blocks()
        for request in requests:
            assert decoder.get_num_new_matched_tokens(request, 0) == (4, True)
            decoder.update_state_after_alloc(request, blocks, 4)
        assert all(event.wait(2) for event in started.values())
        assert decoder.build_connector_meta(None).requests == {}
        requests[0].status = modules.scheduler.RequestStatus.FINISHED_ABORTED
        assert decoder.request_finished(requests[0], blocks.get_block_ids()) == (False, None)
        now[0] = decoder.wan_pd.timeout_s + 1
        metadata = decoder.build_connector_meta(None)
        assert metadata.requests == {}
        assert metadata.failed_recv == {
            request.request_id: blocks.get_unhashed_block_ids_all_groups() for request in requests
        }
        assert dispatcher.stats()["cancelled_active"] == 2
        release.set()
        with dispatcher._condition:
            assert dispatcher._condition.wait_for(lambda: not dispatcher._entries, timeout=2)
        assert decoder.build_connector_meta(None).requests == {}
        decoder.update_connector_output(modules.output(finished_recving=set(started), failed_recving=set(started)))
        assert not decoder._recv_manifests and not decoder._reqs_need_recv
        assert decoder._commit_sender.stats()["reserved"] == 0
    finally:
        release.set()
        dispatcher.close(timeout=2)


def test_full_cache_hit_skips_p_and_disabled_receiver_first_preserves_fallback(modules, monkeypatch):
    decoder = _scheduler(modules, monkeypatch, producer=False)
    request = _decode_request("cached")
    assert decoder.get_num_new_matched_tokens(request, 4) == (0, False)
    assert decoder._reservations.stats()["reserved"] == 0
    assert not request.kv_transfer_params["do_remote_prefill"]
    # Native allocation may fail, a later schedule may find a complete prefix,
    # or alignment may remove the external hit. None may leak admission credits.
    for outcome in ("abort-before-allocation", "prefix-grew", "zero-external"):
        waiting = _decode_request(outcome)
        assert decoder.get_num_new_matched_tokens(waiting, 0) == (4, True)
        if outcome == "abort-before-allocation":
            waiting.status = modules.scheduler.RequestStatus.FINISHED_ABORTED
            assert decoder.request_finished(waiting, ([], [])) == (False, None)
        elif outcome == "prefix-grew":
            assert decoder.get_num_new_matched_tokens(waiting, 4) == (0, False)
        else:
            decoder.update_state_after_alloc(waiting, _blocks(), 0)
        assert decoder._reservations.stats()["reserved"] == 0
        assert decoder._commit_sender.stats()["reserved"] == 0
    fallback = _shared._scheduler(modules, monkeypatch, producer=False)
    unconfigured = _decode_request("disabled")
    assert fallback.get_num_new_matched_tokens(unconfigured, 0) == (0, False)
    assert fallback._reservations is None and not unconfigured.kv_transfer_params["do_remote_prefill"]
    producer = _shared._scheduler(modules, monkeypatch, producer=True)
    params = _shared._publish(producer, modules, _shared._request(kv_transfer_params={"do_remote_decode": True}))
    original = _shared._request("ordinary-pd", kv_transfer_params=params)
    assert fallback.get_num_new_matched_tokens(original, 0) == (4, True)
    fallback.update_state_after_alloc(original, _blocks(), 4)
    assert set(fallback.build_connector_meta(None).requests) == {original.request_id}


def test_early_attention_publication_pins_before_finish_seals_and_retains_on_abort(modules, monkeypatch):
    from vllm.v1.core.sched.output import KVConnectorBlockState

    producer = _scheduler(modules, monkeypatch, producer=True, dense=True, layerwise_publication=True)
    allocations, touches, frees = {}, [], []

    def touch(blocks):
        for block in blocks:
            touches.append(block.block_id)
            block.ref_cnt += 1

    def free(blocks):
        for block in blocks:
            frees.append(block.block_id)
            block.ref_cnt -= 1

    pool = SimpleNamespace(touch=touch, free_blocks=free)
    producer.bind_kv_cache_manager(
        SimpleNamespace(
            block_pool=pool, get_blocks=lambda request_id: SimpleNamespace(blocks=(allocations[request_id],))
        )
    )

    def prepare(request_id, ids):
        request = _shared._request(
            request_id, output_token_ids=[], num_computed_tokens=2, status=modules.scheduler.RequestStatus.RUNNING
        )
        ticket = modules.scheduler.ReservationTicket(
            "receiver-epoch",
            "reservation-" + request_id,
            "decode-" + request_id,
            "route-" + request_id,
            producer._identity(request),
            5,
            0,
            modules.contracts.fingerprint({"d_blocks": ids}),
        )
        request.kv_transfer_params = {"do_remote_decode": True, "wan_pd_reservation": ticket.to_wire()}
        allocations[request_id] = [SimpleNamespace(block_id=block_id, ref_cnt=1, is_null=False) for block_id in ids]
        producer.on_new_request(request)
        assert producer.get_num_new_matched_tokens(request, 0) == (0, False)
        state = KVConnectorBlockState({request_id}, lambda req_id: (ids,), {})
        output = SimpleNamespace(num_scheduled_tokens={request_id: 3}, kv_connector_block_state=state)
        meta = producer.build_connector_meta(output)
        transfer = meta.producer_publications[request_id]
        manifest = modules.contracts.ContinuationManifest.from_wire(transfer["continuation_manifest"])
        assert producer._publications.get_publication(ticket.fingerprint)["status"] == "PREPARED"
        assert all(block.ref_cnt == 2 for block in allocations[request_id])
        assert not producer._reqs_need_send  # Request is still computing its final step.
        return request, ticket, manifest

    request, ticket, manifest = prepare("complete", [1, 2, 3])
    assert touches == [1, 2, 3] and not frees
    identity = manifest.engine_epoch, manifest.attempt_id, manifest.digest
    with pytest.raises(ValueError, match="not sealed"):
        producer._publications.validate_commit(*identity, True)
    producer._publications.update_ready(*identity, "0:0:0", ["layer.0"], True)
    assert producer._publications.get_ready(*identity)["status"] == "PREPARED"
    request.status = modules.scheduler.RequestStatus.FINISHED_LENGTH_CAPPED
    request.output_token_ids = [9]
    held, final = producer.request_finished(request, ([1, 2, 3],))
    assert held and final["continuation_manifest"] == manifest.to_wire(producer.wan_pd)
    assert producer._publications.get_ready(*identity)["status"] == "SEALED"
    record = modules.control.CommitRecord("127.0.0.1", 5002, *identity, True)
    _shared._deliver_completion(modules, producer, record)
    output = modules.output()
    producer.update_connector_output(output)
    assert output.finished_sending == {request.request_id}
    assert frees == [3, 2, 1] and all(block.ref_cnt == 1 for block in allocations[request.request_id])

    aborted, ticket, manifest = prepare("aborted", [4, 5, 6])
    producer.build_connector_meta(
        SimpleNamespace(num_scheduled_tokens={}, preempted_req_ids={aborted.request_id}, kv_connector_block_state=None)
    )
    # A PP batch may be preempted while earlier device work is still in flight.
    # Its old pages stay pinned, but re-admission must not wait on its own
    # published ledger entry or issue a second writer for the same attempt.
    assert producer.get_num_new_matched_tokens(aborted, 0) == (0, False)
    assert not aborted.kv_transfer_params["do_remote_decode"]
    aborted.status = modules.scheduler.RequestStatus.FINISHED_ABORTED
    held, error = producer.request_finished(aborted, ([4, 5, 6],))
    assert held and "aborted" in error["wan_pd_error"]
    assert producer._publications.get_publication(ticket.fingerprint)["status"] == "FAILED"
    assert all(block.ref_cnt == 2 for block in allocations[aborted.request_id])
    assert producer._source_ledger.stats()["quarantined"] == 1
    assert frees == [3, 2, 1]
