# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-only checks of input identity metadata; no device content hashing."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.config import WanPDConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.input_identity import (
    derive_structured_input_fingerprint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import request_identity


def _request():
    return SimpleNamespace(
        num_prompt_tokens=4,
        prompt_token_ids=[1, 2, 2, 3],
        cache_salt="tenant-a",
        mm_features=[
            SimpleNamespace(
                identifier="untrusted-cache-key",
                modality="image",
                mm_position=SimpleNamespace(offset=1, length=2, is_embed=None),
            )
        ],
        kv_transfer_params={
            "wan_pd_input_identity": {
                "version": 1,
                "kind": "multimodal",
                "processor_fingerprint": "processor-and-encoder-revision-1",
                "media": [{"sha256": "a" * 64, "modality": "image", "offset": 1, "length": 2}],
            }
        },
    )


def _derive(request, **kwargs):
    return derive_structured_input_fingerprint(request, namespace="weights-v1", **kwargs)


def test_media_content_text_tenant_and_processor_are_bound_not_cache_identifier():
    request = _request()
    original = _derive(request)
    assert original == _derive(deepcopy(request))
    request.mm_features[0].identifier = "same-media-new-cache-key"
    assert original == _derive(request)
    for mutation in (
        lambda req: req.prompt_token_ids.__setitem__(0, 7),
        lambda req: setattr(req, "cache_salt", "tenant-b"),
        lambda req: req.kv_transfer_params["wan_pd_input_identity"]["media"][0].update(sha256="b" * 64),
        lambda req: req.kv_transfer_params["wan_pd_input_identity"].update(processor_fingerprint="processor-v2"),
    ):
        changed = deepcopy(request)
        mutation(changed)
        assert original != _derive(changed)
    assert original != derive_structured_input_fingerprint(request, namespace="weights-v2")


@pytest.mark.parametrize("change", [{"offset": 0}, {"length": 1}, {"modality": "video"}])
def test_declared_media_must_match_actual_request(change):
    request = _request()
    request.kv_transfer_params["wan_pd_input_identity"]["media"][0].update(change)
    with pytest.raises(ValueError, match="positions or modalities"):
        _derive(request)


def test_processor_abi_and_explicit_fingerprint_agreement():
    request = _request()
    expected = _derive(request, processor_fingerprint="processor-and-encoder-revision-1")
    request.kv_transfer_params["wan_pd_input_fingerprint"] = expected
    assert _derive(request) == expected
    with pytest.raises(ValueError, match="configured ABI"):
        _derive(request, processor_fingerprint="another-processor")
    request.kv_transfer_params["wan_pd_input_fingerprint"] = "opaque-old-value"
    with pytest.raises(ValueError, match="disagree"):
        _derive(request)
    del request.kv_transfer_params["wan_pd_input_identity"]
    assert _derive(request) is None  # The original opaque identity path stays available.


def test_manifest_opt_in_preserves_identity_without_enabling_special_prefix_reuse():
    request = _request()
    request.block_hashes = [b"prefix-block"]
    with pytest.raises(ValueError, match="disabled"):
        request_identity(request, namespace="weights-v1")
    config = WanPDConfig(enabled=True, structured_input_identity=True)
    identity = request_identity(request, namespace="weights-v1", config=config, hash_block_size=2)
    request.mm_features = []  # The D side can carry identity without encoder input tensors.
    assert identity == request_identity(request, namespace="weights-v1", config=config, hash_block_size=2)
    assert not identity.prefix_reusable


class _DeviceEmbeddings:
    shape = (4, 8)
    dtype = "torch.bfloat16"

    def __getattr__(self, name):
        raise AssertionError(f"Tensor data operation attempted: {name}")


def test_embedding_identity_checks_metadata_without_reading_device_data():
    request = _request()
    request.mm_features = []
    request.prompt_token_ids = None
    request.prompt_embeds = _DeviceEmbeddings()
    value = {
        "version": 1,
        "kind": "prompt_embeds",
        "sha256": "c" * 64,
        "shape": [4, 8],
        "dtype": "bfloat16",
        "encoder_fingerprint": "encoder-weights-and-preprocessing-v1",
    }
    request.kv_transfer_params["wan_pd_input_identity"] = value
    result = _derive(request)
    value["dtype"] = "torch.bfloat16"
    assert _derive(request) == result
    value["shape"] = [4, 16]
    with pytest.raises(ValueError, match="shape or dtype"):
        _derive(request)


def test_reject_incomplete_or_unsupported_identity_semantics():
    request = _request()
    value = request.kv_transfer_params["wan_pd_input_identity"]
    value["media"][0]["sha256"] = "https://media.invalid/image.png"
    with pytest.raises(ValueError, match="sha256"):
        _derive(request)
    value["media"][0]["sha256"] = "a" * 64
    request.prompt_token_ids = None
    with pytest.raises(ValueError, match="complete prompt token IDs"):
        _derive(request)
    request.prompt_token_ids = [1, 2, 2, 3]
    request.mm_features[0].mm_position.is_embed = object()
    with pytest.raises(ValueError, match="Masked multimodal"):
        _derive(request)
    request.mm_features[0].mm_position.is_embed = None
    request.lora_request = object()
    with pytest.raises(ValueError, match="LoRA"):
        _derive(request)
