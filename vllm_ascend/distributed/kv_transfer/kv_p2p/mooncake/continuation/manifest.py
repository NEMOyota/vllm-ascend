# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Immutable request and state contracts for lossless PD handoff.

The manifest describes semantic coverage, not device addresses or Mooncake
handles. A digest detects inconsistent control messages; it does not authenticate
an untrusted peer or checksum device payloads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from .config import WanPDConfig

PROTOCOL_VERSION = 1
MAX_TEXT_LENGTH = 4096
MAX_STATE_TOKENS = 2**31 - 1
MAX_SHAPE_DIMS = 16


def _text(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or not (0 if allow_empty else 1) <= len(value) <= MAX_TEXT_LENGTH:
        raise ValueError(f"{name} must be a bounded {'possibly empty ' if allow_empty else ''}string")
    return value


def _integer(name: str, value: Any, *, maximum: int = MAX_STATE_TOKENS) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


def canonical_bytes(value: Any, *, max_bytes: int = 2 * 1024 * 1024) -> bytes:
    """Canonical JSON for small metadata only; never converts device tensors."""
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Continuation metadata must contain finite JSON primitives") from exc
    if len(result) > max_bytes:
        raise ValueError("Continuation metadata exceeds max_manifest_bytes")
    return result


def fingerprint(value: Any, *, max_bytes: int = 2 * 1024 * 1024) -> str:
    return hashlib.sha256(canonical_bytes(value, max_bytes=max_bytes)).hexdigest()


def _keys(cls: type, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = value.keys() - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(value)


@dataclass(frozen=True)
class RequestIdentity:
    namespace: str
    input_fingerprint: str
    prompt_tokens: int
    prefix_hashes: tuple[str, ...] = ()
    prefix_reusable: bool = False
    prefix_hash_block_size: int = 0

    def __post_init__(self) -> None:
        _text("namespace", self.namespace)
        _text("input_fingerprint", self.input_fingerprint)
        _integer("prompt_tokens", self.prompt_tokens)
        if type(self.prefix_reusable) is not bool:
            raise ValueError("prefix_reusable must be a boolean")
        _integer("prefix_hash_block_size", self.prefix_hash_block_size)
        if self.prefix_reusable and self.prefix_hash_block_size == 0:
            raise ValueError("A reusable prefix requires its hash block size")
        if not isinstance(self.prefix_hashes, tuple):
            raise ValueError("prefix_hashes must be an immutable tuple")
        for value in self.prefix_hashes:
            _text("prefix hash", value)

    def to_wire(self) -> dict[str, Any]:
        result = asdict(self)
        result["prefix_hashes"] = list(self.prefix_hashes)
        return result

    @classmethod
    def from_wire(cls, value: dict[str, Any], config: WanPDConfig | None = None) -> RequestIdentity:
        config = config or WanPDConfig()
        values = _keys(cls, value)
        hashes = values.get("prefix_hashes", [])
        if not isinstance(hashes, (list, tuple)) or len(hashes) > config.max_prefix_hashes:
            raise ValueError("Invalid or excessive prefix hashes")
        values["prefix_hashes"] = tuple(hashes)
        try:
            result = cls(**values)
        except TypeError as exc:
            raise ValueError("Incomplete request identity") from exc
        canonical_bytes(result.to_wire(), max_bytes=config.max_manifest_bytes)
        return result


def request_identity(
    request: Any,
    *,
    namespace: str,
    explicit_fingerprint: str | None = None,
    hash_block_size: int = 0,
    config: WanPDConfig | None = None,
) -> RequestIdentity:
    """Identify original input without copying embeddings from the device.

    Embedding, multimodal and LoRA requests require a caller-provided immutable
    fingerprint binding their content, adapter revision and position metadata.
    ``block_hashes`` are retained for *additional* prefix agreement, not used as
    evidence that a partial prompt hash identifies the entire input.
    Unknown hash granularity disables prefix reuse; scheduler block size must
    not be substituted for the framework's actual hash block size.
    """
    config = config or WanPDConfig()
    _text("namespace", namespace)
    _integer("hash_block_size", hash_block_size)
    params = getattr(request, "kv_transfer_params", None) or {}
    if explicit_fingerprint is None:
        explicit_fingerprint = params.get("wan_pd_input_fingerprint")
    tokens = getattr(request, "prompt_token_ids", None)
    prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if prompt_tokens is None:
        if tokens is not None:
            prompt_tokens = len(tokens)
        else:
            embeds = getattr(request, "prompt_embeds", None)
            if embeds is None:
                raise ValueError("Request has no prompt length")
            prompt_tokens = int(embeds.shape[0])
    _integer("prompt_tokens", prompt_tokens)
    salt = getattr(request, "cache_salt", None)
    if salt is not None:
        _text("cache_salt", salt)
    special_input = (
        getattr(request, "prompt_embeds", None) is not None
        or bool(getattr(request, "mm_features", None))
        or getattr(request, "lora_request", None) is not None
        or getattr(request, "prompt_is_token_ids", None) is not None
    )
    if explicit_fingerprint is not None:
        _text("wan_pd_input_fingerprint", explicit_fingerprint)
        input_hash = fingerprint(
            {"namespace": namespace, "input": explicit_fingerprint, "tokens": prompt_tokens, "cache_salt": salt}
        )
    else:
        if special_input or tokens is None:
            raise ValueError("Embedding, multimodal and LoRA PD require wan_pd_input_fingerprint")
        if len(tokens) != prompt_tokens:
            raise ValueError("Prompt token IDs do not match num_prompt_tokens")
        # Hash token IDs incrementally so long prompts do not become metadata.
        digest = hashlib.sha256(canonical_bytes({"namespace": namespace, "tokens": prompt_tokens}))
        for token in tokens:
            _integer("token ID", token, maximum=2**32 - 1)
            digest.update(token.to_bytes(4, "big"))
        if salt is not None:
            digest.update(canonical_bytes({"cache_salt": salt}))
        input_hash = digest.hexdigest()
    raw_hashes = getattr(request, "block_hashes", ())
    if len(raw_hashes) > config.max_prefix_hashes:
        raise ValueError("Request exceeds max_prefix_hashes")
    hashes: list[str] = []
    for block_hash in raw_hashes:
        if not isinstance(block_hash, bytes) or not 0 < len(block_hash) <= 128:
            raise ValueError("Request block_hashes must contain bounded byte digests")
        hashes.append(block_hash.hex())
    # Explicit whole-input identity alone does not establish compatibility of
    # intermediate embedding/MM/adapter checkpoints across requests.
    return RequestIdentity(
        namespace=namespace,
        input_fingerprint=input_hash,
        prompt_tokens=prompt_tokens,
        prefix_hashes=tuple(hashes),
        prefix_reusable=bool(hashes) and not special_input and hash_block_size > 0,
        prefix_hash_block_size=hash_block_size,
    )


@dataclass(frozen=True)
class StateComponent:
    component_id: str
    layer_name: str
    cache_group: int
    kind: str
    dtype: str
    shape: tuple[int, ...]
    token_start: int
    token_end: int
    checkpoint_token: int | None = None

    def __post_init__(self) -> None:
        for name in ("component_id", "layer_name", "kind", "dtype"):
            _text(name, getattr(self, name))
        _integer("cache_group", self.cache_group)
        if not isinstance(self.shape, tuple) or len(self.shape) > MAX_SHAPE_DIMS:
            raise ValueError("State component shape must be a bounded immutable tuple")
        for dimension in self.shape:
            _integer("shape dimension", dimension)
        _integer("token_start", self.token_start)
        _integer("token_end", self.token_end)
        if self.token_end < self.token_start:
            raise ValueError("State component token coverage is reversed")
        if self.checkpoint_token is not None:
            _integer("checkpoint_token", self.checkpoint_token)
            if self.checkpoint_token != self.token_end:
                raise ValueError("State checkpoint must match its covered token boundary")

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> StateComponent:
        values = _keys(cls, value)
        shape = values.get("shape")
        if not isinstance(shape, (tuple, list)):
            raise ValueError("State component shape must be a list")
        values["shape"] = tuple(shape)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Incomplete state component") from exc


@dataclass(frozen=True)
class ContinuationManifest:
    request_id: str
    attempt_id: str
    engine_epoch: str
    namespace: str
    layout_fingerprint: str
    input_fingerprint: str
    prompt_tokens: int
    state_position: int
    prefix_tokens: int
    components: tuple[StateComponent, ...]
    prefix_hashes: tuple[str, ...] = ()
    prefix_reusable: bool = False
    transfer_plan_fingerprint: str = ""
    prefix_hash_block_size: int = 0
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "attempt_id",
            "engine_epoch",
            "namespace",
            "layout_fingerprint",
            "input_fingerprint",
        ):
            _text(name, getattr(self, name))
        if type(self.protocol_version) is not int or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported continuation manifest version")
        for name in ("prompt_tokens", "state_position", "prefix_tokens"):
            _integer(name, getattr(self, name))
        if not self.prefix_tokens <= self.state_position <= self.prompt_tokens:
            raise ValueError("Manifest requires prefix_tokens <= state_position <= prompt_tokens")
        RequestIdentity(
            self.namespace,
            self.input_fingerprint,
            self.prompt_tokens,
            self.prefix_hashes,
            self.prefix_reusable,
            self.prefix_hash_block_size,
        )
        if not isinstance(self.components, tuple):
            raise ValueError("Manifest components must be an immutable tuple")
        seen = set()
        for component in self.components:
            if not isinstance(component, StateComponent):
                raise ValueError("Invalid state component")
            if component.component_id in seen:
                raise ValueError("Duplicate state component ID")
            seen.add(component.component_id)
            if component.token_end != self.state_position:
                raise ValueError("Mixed state components do not cover the same continuation boundary")

    def payload(self) -> dict[str, Any]:
        # JSON round-trip makes tuple/list representation identical on either peer.
        return json.loads(canonical_bytes(asdict(self), max_bytes=64 * 1024 * 1024))

    @property
    def digest(self) -> str:
        return fingerprint(self.payload(), max_bytes=64 * 1024 * 1024)

    def to_wire(self, config: WanPDConfig | None = None) -> dict[str, Any]:
        config = config or WanPDConfig()
        if len(self.components) > config.max_manifest_components or len(self.prefix_hashes) > config.max_prefix_hashes:
            raise ValueError("Manifest exceeds configured component or prefix limits")
        result = {"manifest": self.payload(), "digest": self.digest}
        canonical_bytes(result, max_bytes=config.max_manifest_bytes)
        return result

    @classmethod
    def from_wire(cls, value: dict[str, Any], config: WanPDConfig | None = None) -> ContinuationManifest:
        config = config or WanPDConfig()
        if not isinstance(value, dict) or set(value) != {"manifest", "digest"}:
            raise ValueError("Expected manifest and digest")
        canonical_bytes(value, max_bytes=config.max_manifest_bytes)
        values = _keys(cls, value["manifest"])
        components = values.get("components")
        hashes = values.get("prefix_hashes", [])
        if not isinstance(components, (list, tuple)) or len(components) > config.max_manifest_components:
            raise ValueError("Invalid or excessive state components")
        if not isinstance(hashes, (list, tuple)) or len(hashes) > config.max_prefix_hashes:
            raise ValueError("Invalid or excessive prefix hashes")
        values["components"] = tuple(StateComponent.from_wire(component) for component in components)
        values["prefix_hashes"] = tuple(hashes)
        try:
            result = cls(**values)
        except TypeError as exc:
            raise ValueError("Incomplete continuation manifest") from exc
        if result.digest != value["digest"]:
            raise ValueError("Continuation manifest digest mismatch")
        return result

    def validate_identity(self, identity: RequestIdentity) -> None:
        if (self.namespace, self.input_fingerprint, self.prompt_tokens) != (
            identity.namespace,
            identity.input_fingerprint,
            identity.prompt_tokens,
        ):
            raise ValueError("Continuation input or model namespace mismatch")

    def validate_prefix(self, identity: RequestIdentity, *, prefix_tokens: int, block_size: int) -> None:
        self.validate_identity(identity)
        _integer("prefix_tokens", prefix_tokens)
        if type(block_size) is not int or block_size <= 0 or prefix_tokens % block_size:
            raise ValueError("PD prefix must cover complete cache blocks")
        if prefix_tokens > self.state_position:
            raise ValueError("PD prefix exceeds the transferred state position")
        if prefix_tokens == 0:
            return
        if not self.prefix_reusable or not identity.prefix_reusable:
            raise ValueError("PD input has no reusable verified prefix")
        if (
            self.prefix_hash_block_size != identity.prefix_hash_block_size
            or prefix_tokens % self.prefix_hash_block_size
        ):
            raise ValueError("PD prefix hash block granularity mismatch")
        count = prefix_tokens // self.prefix_hash_block_size
        if len(self.prefix_hashes) < count or len(identity.prefix_hashes) < count:
            raise ValueError("PD prefix exceeds available block hashes")
        if self.prefix_hashes[:count] != identity.prefix_hashes[:count]:
            raise ValueError("PD prefix block identity mismatch")
