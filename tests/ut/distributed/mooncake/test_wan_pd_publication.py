# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU ownership experiments for the real early-publication catalog."""

from dataclasses import replace
from threading import Thread
from types import SimpleNamespace

import pytest
from vllm.v1.core.block_pool import BlockPool

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.commit import SourceLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.config import WanPDConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import (
    ContinuationManifest,
    StateComponent,
    fingerprint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.publication import SourcePublicationManager


def _setup(**config_kwargs):
    config = WanPDConfig(enabled=True, **config_kwargs)
    ledger = SourceLedger(config, "epoch")
    pool = BlockPool(num_gpu_blocks=5, enable_caching=False, hash_block_size=2)
    blocks = pool.get_new_blocks(2)
    manager = SimpleNamespace(block_pool=pool, get_blocks=lambda _: SimpleNamespace(blocks=(blocks,)))
    catalog = SourcePublicationManager(config, ledger, manager)
    params = {
        "remote_block_ids": [[1, 2]],
        "remote_engine_id": "producer",
        "remote_request_id": "request",
        "remote_num_prompt_tokens": 4,
        "remote_host": "127.0.0.1",
        "remote_port": 12345,
    }
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
            StateComponent(f"{layer}:0", layer, 0, "full_attention", "bfloat16", (2, 4), 0, 4)
            for layer in ("layer.0", "layer.1")
        ),
        transfer_plan_fingerprint=fingerprint(params),
    )
    return SimpleNamespace(catalog=catalog, manifest=manifest, params=params, blocks=blocks, pool=pool, ledger=ledger)


def _prepare(runtime, workers=None):
    return runtime.catalog.prepare(
        "request",
        runtime.params,
        runtime.manifest,
        workers or {"0:0:0": ("layer.0", "layer.1")},
        ticket_fingerprint="ticket",
    )


def _identity(runtime):
    manifest = runtime.manifest
    return manifest.engine_epoch, manifest.attempt_id, manifest.digest


def test_real_references_survive_request_free_until_all_events_and_commit():
    runtime = _setup()
    wire = _prepare(runtime)
    assert [block.ref_cnt for block in runtime.blocks] == [2, 2]
    assert wire["status"] == "PREPARED"
    # A caller cannot mutate the pinned publication by changing its wire copy.
    wire["transfer_params"]["remote_block_ids"][0][0] = 99
    assert runtime.catalog.get_publication("ticket")["transfer_params"]["remote_block_ids"] == [[1, 2]]
    runtime.pool.free_blocks(runtime.blocks)
    assert [block.ref_cnt for block in runtime.blocks] == [1, 1]
    assert runtime.pool.get_num_free_blocks() == 2

    identity = _identity(runtime)
    runtime.catalog.update_ready(*identity, "0:0:0", ["layer.0"], False)
    assert runtime.catalog.get_ready(*identity)["ready_layers"] == ["0:0:0/layer.0"]
    runtime.catalog.seal("request")
    with pytest.raises(ValueError, match="not sealed"):
        runtime.catalog.validate_commit(*identity, True)
    runtime.catalog.update_ready(*identity, "0:0:0", ["layer.1"], True)
    assert runtime.catalog.get_ready(*identity)["status"] == "SEALED"
    assert runtime.catalog.validate_commit(*identity, True)
    assert runtime.ledger.finish(*identity, True) == "request"
    assert runtime.catalog.retire("request")
    assert [block.ref_cnt for block in runtime.blocks] == [0, 0]
    assert runtime.pool.get_num_free_blocks() == 4
    assert runtime.catalog.get_publication("ticket") is None
    assert runtime.catalog.validate_commit(*identity, True)
    assert not runtime.catalog.retire("request")
    with pytest.raises(ValueError, match="stale"):
        runtime.catalog.update_ready(*identity, "0:0:0", ["layer.0"], False)


def test_mixed_recurrent_state_is_rejected_before_any_reference_changes():
    runtime = _setup()
    recurrent = replace(runtime.manifest.components[1], kind="mamba", checkpoint_token=4)
    runtime.manifest = replace(runtime.manifest, components=(runtime.manifest.components[0], recurrent))
    with pytest.raises(ValueError, match="full_attention"):
        _prepare(runtime)
    assert [block.ref_cnt for block in runtime.blocks] == [1, 1]
    assert runtime.catalog.stats()["source_block_pins"] == 0


def test_abort_quarantines_source_and_never_allows_successful_retirement():
    runtime = _setup()
    _prepare(runtime)
    identity = _identity(runtime)
    # D can fail before P's final model output reaches the scheduler. A late
    # successful P completion must preserve isolation, without crashing P.
    assert runtime.catalog.validate_commit(*identity, False)
    assert runtime.ledger.finish(*identity, False) is None
    runtime.catalog.seal("request")
    assert runtime.catalog.get_ready(*identity)["status"] == "FAILED"
    with pytest.raises(ValueError, match="not sealed"):
        runtime.catalog.validate_commit(*identity, True)
    assert runtime.catalog.abort("request", "producer aborted after first layer")
    assert runtime.catalog.get_ready(*identity)["status"] == "FAILED"
    assert runtime.ledger.stats()["quarantined"] == 1
    runtime.pool.free_blocks(runtime.blocks)
    assert [block.ref_cnt for block in runtime.blocks] == [1, 1]
    with pytest.raises(ValueError, match="not sealed"):
        runtime.catalog.validate_commit(*identity, True)
    with pytest.raises(ValueError, match="successful commit"):
        runtime.catalog.retire("request")
    with pytest.raises(ValueError, match="failed"):
        runtime.catalog.update_ready(*identity, "0:0:0", ["layer.0"], False)


def test_disjoint_pp_owners_cannot_mark_another_workers_layers_ready():
    runtime = _setup()
    _prepare(runtime, {"0:0:0": ["layer.0"], "1:0:0": ["layer.1"]})
    identity = _identity(runtime)
    with pytest.raises(ValueError, match="does not own"):
        runtime.catalog.update_ready(*identity, "0:0:0", ["layer.1"], False)
    with pytest.raises(ValueError, match="before all"):
        runtime.catalog.update_ready(*identity, "0:0:0", [], True)
    runtime.catalog.update_ready(*identity, "0:0:0", ["layer.0"], True)
    runtime.catalog.update_ready(*identity, "0:0:0", ["layer.0"], True)
    runtime.catalog.update_ready(*identity, "1:0:0", ["layer.1"], True)
    assert runtime.catalog.get_ready(*identity)["status"] == "PREPARED"
    runtime.catalog.seal("request")
    assert runtime.catalog.get_ready(*identity)["status"] == "SEALED"


def test_duplicate_preparation_does_not_double_pin_and_conflicts_are_rejected():
    runtime = _setup(max_retained_source_requests=1)
    _prepare(runtime)
    _prepare(runtime)
    assert [block.ref_cnt for block in runtime.blocks] == [2, 2]
    with pytest.raises(ValueError, match="immutable"):
        runtime.catalog.prepare(
            "request", runtime.params, runtime.manifest, ["0:0:0"], ticket_fingerprint="different-ticket"
        )
    other_params = {**runtime.params, "remote_request_id": "other-request"}
    other_manifest = replace(
        runtime.manifest,
        request_id="other-request",
        attempt_id="other-attempt",
        transfer_plan_fingerprint=fingerprint(other_params),
    )
    with pytest.raises(ValueError, match="capacity"):
        runtime.catalog.prepare("other-request", other_params, other_manifest, ["0:0:0"], ticket_fingerprint="other")
    assert [block.ref_cnt for block in runtime.blocks] == [2, 2]
    with pytest.raises(ValueError, match="stale"):
        runtime.catalog.get_ready("old-epoch", runtime.manifest.attempt_id, runtime.manifest.digest)


def test_failed_publication_rolls_back_refs_before_plan_is_visible():
    runtime = _setup()
    runtime.manifest = replace(runtime.manifest, engine_epoch="stale-epoch")
    with pytest.raises(ValueError, match="reservation"):
        _prepare(runtime)
    assert [block.ref_cnt for block in runtime.blocks] == [1, 1]
    assert runtime.catalog.get_publication("ticket") is None


def test_readiness_metadata_limit_is_checked_before_pinning_source_blocks():
    runtime = _setup(max_manifest_bytes=2048)
    with pytest.raises(ValueError, match="max_manifest_bytes"):
        _prepare(runtime, {"worker" * 300: ["layer.0", "layer.1"]})
    assert [block.ref_cnt for block in runtime.blocks] == [1, 1]
    assert runtime.catalog.get_publication("ticket") is None


def test_control_thread_can_update_events_but_cannot_release_block_pool_refs():
    runtime = _setup()
    _prepare(runtime)
    identity = _identity(runtime)
    failures = []

    def control():
        runtime.catalog.update_ready(*identity, "0:0:0", ["layer.0", "layer.1"], True)
        try:
            runtime.catalog.retire("request")
        except RuntimeError as exc:
            failures.append(str(exc))

    thread = Thread(target=control)
    thread.start()
    thread.join()
    assert failures == ["Publication block ownership must be changed on the scheduler thread"]
    assert runtime.catalog.get_ready(*identity)["finalized_workers"] == ["0:0:0"]
    assert [block.ref_cnt for block in runtime.blocks] == [2, 2]
