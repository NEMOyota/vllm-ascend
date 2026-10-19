# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks of the production scheduler and PD completion ownership."""

import importlib
import sys
import threading
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    # Import upstream first, while transformers can still discover that this
    # machine has no NPU. Only installation metadata and the unused NPU module
    # need substitutes; scheduler, cache specs and wire types stay real.
    from vllm.v1.outputs import KVConnectorOutput

    import vllm_ascend

    if not hasattr(vllm_ascend, "_build_info"):
        build_info = ModuleType("vllm_ascend._build_info")
        build_info.__soc_version__ = "ASCEND910B1"
        monkeypatch.setitem(sys.modules, build_info.__name__, build_info)
        monkeypatch.setattr(vllm_ascend, "_build_info", build_info, raising=False)
    if "torch_npu" not in sys.modules and importlib.util.find_spec("torch_npu") is None:
        monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    package = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake"
    return SimpleNamespace(
        scheduler=importlib.import_module(package + ".pull_scheduler"),
        contracts=importlib.import_module(package + ".continuation"),
        commit=importlib.import_module(package + ".continuation.commit"),
        control=importlib.import_module(package + ".continuation.control"),
        output=KVConnectorOutput,
    )


def _request(request_id="request-a", **overrides):
    values = dict(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3, 4, 5],
        prompt_embeds=None,
        num_prompt_tokens=5,
        block_hashes=[b"block-0", b"block-1"],
        kv_transfer_params={},
        _all_token_ids=[1, 2, 3, 4, 5],
        max_tokens=1,
        output_token_ids=[6],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _manifest(contracts, request=None):
    identity = contracts.request_identity(request or _request(), namespace="weights-v1", hash_block_size=2)
    return contracts.ContinuationManifest(
        request_id="request-a",
        attempt_id="attempt-a",
        engine_epoch="producer-epoch",
        namespace=identity.namespace,
        layout_fingerprint="layout-v1",
        input_fingerprint=identity.input_fingerprint,
        prompt_tokens=identity.prompt_tokens,
        state_position=4,
        prefix_tokens=0,
        prefix_hashes=identity.prefix_hashes,
        prefix_reusable=identity.prefix_reusable,
        prefix_hash_block_size=2,
        components=(
            contracts.StateComponent("k", "layer.0", 0, "attention_k", "bfloat16", (2, 8), 0, 4),
            contracts.StateComponent("s", "layer.1", 1, "ssm", "float32", (8, 8), 0, 4, 4),
        ),
    )


def test_source_commit_replay_never_unpins_another_attempt(modules):
    config = modules.contracts.WanPDConfig(max_retained_source_requests=1, max_terminal_requests=1)
    ledger = modules.commit.SourceLedger(config, "producer-epoch")
    manifest = _manifest(modules.contracts)
    assert ledger.reserve(manifest.request_id)
    ledger.publish(manifest)
    with pytest.raises(ValueError, match="epoch"):
        ledger.finish("previous-process", manifest.attempt_id, manifest.digest, True)
    assert ledger.finish(manifest.engine_epoch, manifest.attempt_id, manifest.digest, False) is None
    assert ledger.stats()["quarantined"] == 1
    assert not ledger.reserve("other-request")
    assert ledger.finish(manifest.engine_epoch, manifest.attempt_id, manifest.digest, True) == manifest.request_id

    # Reusing a public request ID must not let a delayed old ACK free its blocks.
    next_attempt = replace(manifest, attempt_id="attempt-b")
    assert ledger.reserve(next_attempt.request_id)
    ledger.publish(next_attempt)
    assert ledger.finish(manifest.engine_epoch, manifest.attempt_id, manifest.digest, True) is None
    assert ledger.stats()["published"] == 1
    assert not ledger.reserve("other-request")

    assert (
        ledger.finish(next_attempt.engine_epoch, next_attempt.attempt_id, next_attempt.digest, True)
        == manifest.request_id
    )
    active_attempt = replace(manifest, attempt_id="attempt-c")
    assert ledger.reserve(active_attempt.request_id)
    ledger.publish(active_attempt)
    # attempt-a's tombstone was evicted, but its retry remains a harmless ACK.
    assert ledger.finish(manifest.engine_epoch, manifest.attempt_id, manifest.digest, True) is None
    assert ledger.stats()["published"] == 1 and ledger.stats()["terminal"] == 1
    with pytest.raises(ValueError, match="Unknown or mismatched"):
        ledger.finish(manifest.engine_epoch, manifest.attempt_id, manifest.digest, False)
    with pytest.raises(ValueError, match="Unknown or mismatched"):
        ledger.finish(active_attempt.engine_epoch, active_attempt.attempt_id, "wrong-digest", True)
    assert not ledger.reserve("other-request")


def test_commit_outbox_bounds_retries_and_services_healthy_peer(modules, monkeypatch):
    monkeypatch.setattr(modules.control, "monotonic", lambda: 0)
    config = modules.contracts.WanPDConfig(
        max_pending_requests=2,
        max_active_requests=2,
        commit_retry_limit=2,
        commit_retry_interval_s=1,
    )
    sent = []

    def send(record):
        sent.append(record.host)
        if record.host == "unreachable":
            raise TimeoutError("peer unavailable")

    sender = modules.control.BoundedCommitSender(config, send)
    bad = modules.control.CommitRecord("unreachable", 1, "epoch", "attempt-a", "digest-a", True)
    good = replace(bad, host="reachable", attempt_id="attempt-b")
    assert sender.reserve("bad") and sender.reserve("good")
    assert not sender.reserve("overflow")
    sender.enqueue("bad", bad)
    sender.enqueue("good", good)
    assert sender.progress(now=0)
    assert sender.progress(now=0)
    assert sent == ["unreachable", "reachable"]
    assert not sender.progress(now=0)
    assert sender.progress(now=1)
    assert not sender.progress(now=100)
    assert sender.stats() == {"reserved": 1, "failed": 1}
    assert sender.reserve("replacement")
    assert not sender.reserve("overflow")


def _scheduler(modules, monkeypatch, *, producer):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

    base = importlib.import_module(modules.scheduler.__package__ + ".base_scheduler")
    monkeypatch.setattr(base, "init_ascend_config", lambda config: None)
    monkeypatch.setattr(base, "get_ascend_config", lambda: SimpleNamespace(kvpp_config=SimpleNamespace(size=1)))
    monkeypatch.setattr(base, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(modules.control.BoundedCommitSender, "start", lambda self: None)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=producer,
            is_kv_consumer=not producer,
            kv_role="kv_producer" if producer else "kv_consumer",
            kv_port=5000,
            kv_connector_extra_config={
                "wan_pd": {"enabled": True, "max_pending_requests": 2, "max_active_requests": 2}
            },
        ),
        model_config=SimpleNamespace(
            model="organization/model", hf_config={"model_type": "hybrid"}, revision="revision-a", dtype=torch.bfloat16
        ),
        cache_config=SimpleNamespace(block_size=2, cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
        speculative_config=None,
    )
    groups = [
        KVCacheGroupSpec(
            ["layer.0"], FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
        ),
        KVCacheGroupSpec(["layer.1"], MambaSpec(block_size=2, shapes=((8, 8),), dtypes=(torch.float32,))),
    ]
    cache = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=groups, hash_block_size=2)
    scheduler = modules.scheduler.MooncakePullConnectorScheduler(config, "producer" if producer else "decoder", cache)
    if producer:
        metadata = modules.scheduler.MooncakeTransferMetadata(
            engine_id="producer",
            te_rpc_port=5001,
            block_size=2,
            num_blocks=16,
            layer_names=["layer.0", "layer.1"],
            layer_block_sizes=[2, 2],
            group_indices=[0, 1],
            kv_caches_base_addr=[[1000], [2000]],
            block_strides=[[64], [256]],
            block_lens=[[64], [256]],
            block_shapes=[[(2, 8)], [(8, 8)]],
            block_size_scales=[[1], [1]],
            local_ip="127.0.0.1",
            state_kinds=["attention", "mamba"],
            state_dtypes=[["bfloat16"], ["float32"]],
        )
        # Run the real merge/publication constructor; network dispatch is driven
        # explicitly below rather than leaving daemon sockets behind this test.
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
        )
    return scheduler


def _publish(scheduler, modules, request):
    scheduler.on_new_request(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    request.status = modules.scheduler.RequestStatus.FINISHED_LENGTH_CAPPED
    held, params = scheduler.request_finished(request, ([1, 2], [3]))
    assert held
    return params


def _deliver_completion(modules, producer, record):
    import msgspec

    messages = iter(
        [
            [
                b"consumer",
                b"",
                msgspec.msgpack.encode(
                    (modules.control.COMMIT_MSG, record.epoch, record.attempt_id, record.digest, record.committed)
                ),
            ]
        ]
    )
    replies = []
    socket = SimpleNamespace(recv_multipart=lambda: next(messages), send_multipart=replies.append)
    # Exhausting this transport double stops the real production dispatch loop.
    with pytest.raises(StopIteration):
        producer._sending_thread._run_busy_loop(socket)
    assert replies == [(b"consumer", b"", modules.scheduler.ACK_MSG)]


def test_embedding_handoff_preserves_original_identity_and_state_boundary(modules, monkeypatch):
    import torch

    producer = _scheduler(modules, monkeypatch, producer=True)
    decoder = _scheduler(modules, monkeypatch, producer=False)
    unidentified = _request(
        "missing-embedding-identity",
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(5, 8),
        kv_transfer_params={"do_remote_decode": True},
        status=modules.scheduler.RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    producer.on_new_request(unidentified)
    assert not unidentified.kv_transfer_params["do_remote_decode"]
    assert unidentified.num_prompt_tokens == 5 and unidentified.prompt_embeds.shape[0] == 5
    assert producer.get_num_new_matched_tokens(unidentified, 0) == (0, False)
    held, error = producer.request_finished(unidentified, ([1, 2, 3], [4]))
    assert not held and "wan_pd_input_fingerprint" in error["wan_pd_error"]
    assert producer._source_ledger.stats()["reserved"] == 0

    request = _request(
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(5, 8),
        kv_transfer_params={"do_remote_decode": True, "wan_pd_input_fingerprint": "encoder-r1-content-hash"},
    )
    params = _publish(producer, modules, request)
    manifest = modules.contracts.ContinuationManifest.from_wire(params["continuation_manifest"])
    assert request.num_prompt_tokens == 4
    assert params["remote_num_prompt_tokens"] == 4
    assert manifest.prompt_tokens == 5 and manifest.state_position == 4
    assert [component.dtype for component in manifest.components] == ["bfloat16", "float32"]
    assert manifest.components[1].checkpoint_token == 4
    decode_request = _request(
        "decoder-request", prompt_token_ids=None, prompt_embeds=torch.zeros(5, 8), kv_transfer_params=params
    )
    # E sends embeddings with no token-ID list. D must load four prefill tokens,
    # then consume the final prompt token once for the hybrid state transition.
    assert decoder.get_num_new_matched_tokens(decode_request, 0) == (4, True)
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: ([10, 11], [12]), get_block_ids=lambda: ([10, 11], [12])
    )
    decoder.update_state_after_alloc(decode_request, blocks, 4)
    records = []
    decoder._commit_sender.send = records.append
    decoder.update_connector_output(modules.output(finished_recving={decode_request.request_id}))
    assert decoder._commit_sender.progress()
    assert len(records) == 1 and records[0].committed
    _deliver_completion(modules, producer, records[0])
    output = modules.output()
    producer.update_connector_output(output)
    assert output.finished_sending == {request.request_id}
    assert producer._source_ledger.stats()["reserved"] == 0


def test_decode_admission_prefix_and_failure_isolation(modules, monkeypatch):
    producer = _scheduler(modules, monkeypatch, producer=True)
    decoder = _scheduler(modules, monkeypatch, producer=False)
    params = _publish(producer, modules, _request(kv_transfer_params={"do_remote_decode": True}))
    bad = _request("wrong-prefix", kv_transfer_params=dict(params), block_hashes=[b"different", b"block-1"])
    assert decoder.get_num_new_matched_tokens(bad, 2) == (0, False)
    assert not bad.kv_transfer_params["do_remote_prefill"]
    assert decoder._commit_sender.stats()["reserved"] == 0

    altered_table = _request("wrong-table", kv_transfer_params=dict(params, remote_block_ids=([9, 10], [11])))
    assert decoder.get_num_new_matched_tokens(altered_table, 0) == (0, False)
    assert decoder._commit_sender.stats()["reserved"] == 0

    request = _request("decode", kv_transfer_params=dict(params))
    assert decoder.get_num_new_matched_tokens(request, 2) == (2, True)
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: ([11], [12]), get_block_ids=lambda: ([10, 11], [12])
    )
    decoder.update_state_after_alloc(request, blocks, 2)
    metadata = decoder.build_connector_meta(None)
    assert metadata.requests["decode"].continuation_manifest == params["continuation_manifest"]
    assert metadata.requests["decode"].num_computed_tokens == 2

    records = []
    decoder._commit_sender.send = records.append
    decoder.update_connector_output(modules.output(finished_recving={"decode"}, failed_recving={"decode"}))
    assert decoder._commit_sender.progress()
    assert len(records) == 1 and not records[0].committed
    _deliver_completion(modules, producer, records[0])
    assert producer._source_ledger.stats()["quarantined"] == 1
    assert producer._sending_thread.get_and_clear_finished_requests() == set()
