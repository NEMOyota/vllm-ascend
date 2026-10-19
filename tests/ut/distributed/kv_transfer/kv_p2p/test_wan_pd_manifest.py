# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts, importing the public package without NPU connector setup."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import continuation


@pytest.fixture(scope="module")
def contracts():
    return continuation


def _request(**overrides):
    values = dict(
        request_id="request-a",
        prompt_token_ids=[1, 2, 3, 4, 5],
        num_prompt_tokens=5,
        block_hashes=[b"block-0", b"block-1"],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _manifest(contracts, identity):
    return contracts.ContinuationManifest(
        request_id="request-a",
        attempt_id="attempt-a",
        engine_epoch="producer-epoch",
        namespace=identity.namespace,
        layout_fingerprint=contracts.fingerprint({"dtype": ["bfloat16", "float32"], "tp": 2}),
        input_fingerprint=identity.input_fingerprint,
        prompt_tokens=identity.prompt_tokens,
        state_position=4,
        prefix_tokens=0,
        prefix_hashes=identity.prefix_hashes,
        prefix_reusable=identity.prefix_reusable,
        prefix_hash_block_size=identity.prefix_hash_block_size,
        components=(
            contracts.StateComponent("k", "layer.0", 0, "attention_k", "bfloat16", (2, 8), 0, 4),
            contracts.StateComponent("s", "layer.1", 1, "ssm", "float32", (8, 8), 0, 4, 4),
        ),
    )


def test_config_is_opt_in_strict_and_bounded(contracts):
    config = contracts.WanPDConfig.from_extra_config({})
    assert not config.enabled
    with pytest.raises(FrozenInstanceError):
        config.enabled = True
    for options in (
        {"enabled": 1},
        {"max_pending_requests": 0},
        {"max_active_requests": 129},
        {"chunk_bytes": True},
        {"max_inflight_bytes": 32 * 1024 * 1024},
        {"timeout_s": float("inf")},
        {"max_retained_source_requests": 0},
        {"commit_retry_limit": 0},
        {"commit_retry_interval_s": False},
        {"unknown_option": 1},
    ):
        with pytest.raises(ValueError):
            contracts.WanPDConfig.from_extra_config({"wan_pd": options})


def test_manifest_roundtrip_and_mixed_state_boundary(contracts):
    identity = contracts.request_identity(_request(), namespace="weights-v1")
    manifest = _manifest(contracts, identity)
    wire = manifest.to_wire()
    assert contracts.ContinuationManifest.from_wire(wire) == manifest
    assert manifest.components[0].dtype == "bfloat16"
    assert manifest.components[1].dtype == "float32"
    manifest.validate_identity(identity)
    wire["manifest"]["attempt_id"] = "late-other-attempt"
    with pytest.raises(ValueError, match="digest"):
        contracts.ContinuationManifest.from_wire(wire)
    with pytest.raises(ValueError, match="continuation boundary"):
        replace(manifest, components=(replace(manifest.components[0], token_end=3), manifest.components[1]))
    with pytest.raises(ValueError, match="checkpoint"):
        replace(manifest.components[1], checkpoint_token=3)


def test_full_prompt_identity_and_prefix_are_independent(contracts):
    identity = contracts.request_identity(_request(), namespace="weights-v1", hash_block_size=2)
    manifest = _manifest(contracts, identity)
    manifest.validate_prefix(identity, prefix_tokens=2, block_size=2)
    # Matching complete block hashes cannot hide a different trailing token.
    changed = contracts.request_identity(_request(prompt_token_ids=[1, 2, 3, 4, 6]), namespace="weights-v1")
    with pytest.raises(ValueError, match="input"):
        manifest.validate_identity(changed)
    wrong_hash = replace(identity, prefix_hashes=("other", identity.prefix_hashes[1]))
    with pytest.raises(ValueError, match="block identity"):
        manifest.validate_prefix(wrong_hash, prefix_tokens=2, block_size=2)
    with pytest.raises(ValueError, match="complete cache blocks"):
        manifest.validate_prefix(identity, prefix_tokens=1, block_size=2)
    with pytest.raises(ValueError, match="namespace"):
        manifest.validate_identity(replace(identity, namespace="weights-v2"))
    # Hashing can be finer than a scheduler/cache block. The second hash must
    # be checked too when a four-token block contains two two-token hashes.
    with pytest.raises(ValueError, match="block identity"):
        manifest.validate_prefix(
            replace(identity, prefix_hashes=(identity.prefix_hashes[0], "wrong-second-half")),
            prefix_tokens=4,
            block_size=4,
        )
    with pytest.raises(ValueError, match="granularity"):
        manifest.validate_prefix(replace(identity, prefix_hash_block_size=1), prefix_tokens=4, block_size=4)


def test_special_input_requires_explicit_identity_and_disables_prefix(contracts):
    request = _request(prompt_token_ids=None, prompt_embeds=SimpleNamespace(shape=(5, 128)))
    with pytest.raises(ValueError, match="wan_pd_input_fingerprint"):
        contracts.request_identity(request, namespace="weights-v1")
    request.kv_transfer_params = {"wan_pd_input_fingerprint": "edge-encoder-revision-and-content"}
    identity = contracts.request_identity(request, namespace="weights-v1")
    assert identity.prompt_tokens == 5
    assert not identity.prefix_reusable
    request.cache_salt = "different-tenant"
    assert contracts.request_identity(request, namespace="weights-v1").input_fingerprint != identity.input_fingerprint
    _manifest(contracts, identity).validate_prefix(identity, prefix_tokens=0, block_size=2)
    with pytest.raises(ValueError, match="reusable"):
        _manifest(contracts, identity).validate_prefix(identity, prefix_tokens=2, block_size=2)
    for field in ("lora_request", "mm_features"):
        with pytest.raises(ValueError, match="wan_pd_input_fingerprint"):
            contracts.request_identity(_request(**{field: [object()]}), namespace="weights-v1")


def test_metadata_limits_and_no_tensor_coercion(contracts):
    identity = contracts.request_identity(_request(), namespace="weights-v1")
    manifest = _manifest(contracts, identity)
    with pytest.raises(ValueError, match="limits"):
        manifest.to_wire(contracts.WanPDConfig(max_manifest_components=1))
    with pytest.raises(ValueError, match="max_manifest_bytes"):
        manifest.to_wire(contracts.WanPDConfig(max_manifest_bytes=16))
    with pytest.raises(ValueError, match="max_prefix_hashes"):
        contracts.request_identity(
            _request(), namespace="weights-v1", config=contracts.WanPDConfig(max_prefix_hashes=1)
        )
    with pytest.raises(ValueError, match="JSON primitives"):
        contracts.canonical_bytes({"not_metadata": object()})
