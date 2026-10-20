# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in identities for special inputs, inspired by legacy MM prefix sharing.

Content SHA256 digests are supplied by the trusted E/router ingestion path;
this module checks metadata, never reads tensor contents or fetches media URLs.
It does not authenticate callers, verify their content digests, or enable prefix
reuse. There is no HMAC without a separately managed server-side secret.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import regex as re

MAX_MEDIA_ITEMS = 1024
MAX_TEXT_LENGTH = 4096
MAX_INTEGER = 2**31 - 1


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= MAX_TEXT_LENGTH:
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_INTEGER:
        raise ValueError(f"Invalid {name}")
    return value


def _object(value: Any, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise ValueError("Structured input identity has missing or unknown fields")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Content sha256 must be a lowercase 64-character hex digest")
    return value


def _media_position(item: dict[str, Any], prompt_tokens: int) -> tuple[int, int, str]:
    offset = _integer(item["offset"], "media offset")
    length = _integer(item["length"], "media length", 1)
    if offset + length > prompt_tokens:
        raise ValueError("Media position exceeds the logical prompt")
    return offset, length, _text(item["modality"], "modality")


def _multimodal_identity(request: Any, value: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    _object(value, {"version", "kind", "processor_fingerprint", "media"})
    processor = _text(value["processor_fingerprint"], "processor_fingerprint")
    if getattr(request, "prompt_embeds", None) is not None:
        raise ValueError("Multimodal identity cannot describe prompt_embeds")
    media = value["media"]
    if not isinstance(media, list) or not 0 < len(media) <= MAX_MEDIA_ITEMS:
        raise ValueError("Structured input identity requires bounded media items")
    normalized = []
    for item in media:
        _object(item, {"sha256", "modality", "offset", "length"})
        offset, length, modality = _media_position(item, prompt_tokens)
        normalized.append(dict(offset=offset, length=length, modality=modality, sha256=_digest(item["sha256"])))
    normalized.sort(key=lambda item: (item["offset"], item["length"], item["modality"]))
    for previous, current in zip(normalized, normalized[1:]):
        if previous["offset"] + previous["length"] > current["offset"]:
            raise ValueError("Overlapping media positions are unsupported")

    features = getattr(request, "mm_features", None) or []
    if features:
        actual = []
        for feature in features:
            position = feature.mm_position
            if getattr(position, "is_embed", None) is not None:
                raise ValueError("Masked multimodal positions require an explicit input fingerprint")
            actual.append((position.offset, position.length, feature.modality))
        expected = [(item["offset"], item["length"], item["modality"]) for item in normalized]
        if sorted(actual) != expected:
            raise ValueError("Structured media positions or modalities differ from the request")
        # feature.identifier can be a caller-provided cache key, not a content hash.

    tokens = getattr(request, "prompt_token_ids", None)
    if tokens is None or len(tokens) != prompt_tokens:
        raise ValueError("Multimodal identity requires complete prompt token IDs")
    token_hash = hashlib.sha256()
    for token in tokens:
        token_hash.update(_integer(token, "token ID").to_bytes(4, "big"))
    return dict(
        kind="multimodal", processor_fingerprint=processor, media=normalized, tokens_sha256=token_hash.hexdigest()
    )


def _embedding_identity(request: Any, value: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    _object(
        value,
        {"version", "kind", "sha256", "shape", "dtype", "encoder_fingerprint"},
        {"processor_fingerprint"},
    )
    if getattr(request, "mm_features", None):
        raise ValueError("Embedding identity cannot omit multimodal feature semantics")
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("Embedding shape must be [prompt_tokens, hidden_size]")
    shape = [_integer(dim, "embedding dimension", 1) for dim in shape]
    if shape[0] != prompt_tokens:
        raise ValueError("Embedding shape differs from the logical prompt length")
    dtype = _text(value["dtype"], "embedding dtype").removeprefix("torch.")
    if dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValueError("Unsupported embedding dtype")
    embeds = getattr(request, "prompt_embeds", None)
    if embeds is not None:
        if tuple(embeds.shape) != tuple(shape) or str(embeds.dtype).removeprefix("torch.") != dtype:
            raise ValueError("Embedding identity shape or dtype differs from the request")
    result = dict(
        kind="prompt_embeds",
        sha256=_digest(value["sha256"]),
        shape=shape,
        dtype=dtype,
        encoder_fingerprint=_text(value["encoder_fingerprint"], "encoder_fingerprint"),
    )
    if "processor_fingerprint" in value:
        result["processor_fingerprint"] = _text(value["processor_fingerprint"], "processor_fingerprint")
    return result


def derive_structured_input_fingerprint(
    request: Any,
    *,
    namespace: str,
    processor_fingerprint: str | None = None,
) -> str | None:
    """Return a whole-input fingerprint, or None when no structured input exists.

    ``kv_transfer_params.wan_pd_input_identity`` version 1 accepts ``multimodal``
    (processor_fingerprint and media sha256/modality/offset/length entries), or
    ``prompt_embeds`` (sha256, shape, dtype and encoder_fingerprint). Embedding
    metadata may additionally carry processor_fingerprint. A configured processor
    fingerprint requires an exact match. Namespace and cache salt scope identity;
    deployments remain responsible for binding the correct tenant and revisions.

    A simultaneous ``wan_pd_input_fingerprint`` must equal the derived digest.
    LoRA, mixed token/embedding inputs and masked media positions retain the
    existing opaque-fingerprint path until their full semantics are represented.
    The caller controls the opt-in flag and must keep special-input prefix reuse
    disabled; this helper does not establish intermediate checkpoint compatibility.
    """
    params = getattr(request, "kv_transfer_params", None) or {}
    if "wan_pd_input_identity" not in params:
        return None
    value = params["wan_pd_input_identity"]
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("Unsupported structured input identity version")
    if getattr(request, "lora_request", None) is not None or getattr(request, "prompt_is_token_ids", None) is not None:
        raise ValueError("LoRA and mixed embeddings require an explicit input fingerprint")
    prompt_tokens = _integer(getattr(request, "num_prompt_tokens", None), "prompt_tokens", 1)
    if value.get("kind") == "multimodal":
        identity = _multimodal_identity(request, value, prompt_tokens)
    elif value.get("kind") == "prompt_embeds":
        identity = _embedding_identity(request, value, prompt_tokens)
    else:
        raise ValueError("Unsupported structured input kind")
    if processor_fingerprint is not None:
        _text(processor_fingerprint, "configured processor_fingerprint")
        if identity.get("processor_fingerprint") != processor_fingerprint:
            raise ValueError("Structured input processor fingerprint differs from configured ABI")
    salt = getattr(request, "cache_salt", None)
    if salt is not None:
        _text(salt, "cache_salt")
    payload = dict(
        version=1, namespace=_text(namespace, "namespace"), cache_salt=salt, prompt_tokens=prompt_tokens, input=identity
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    result = hashlib.sha256(encoded).hexdigest()
    explicit = params.get("wan_pd_input_fingerprint")
    if explicit is not None and explicit != result:
        raise ValueError("Explicit and structured input fingerprints disagree")
    return result
