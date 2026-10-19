# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable shared-pool identity, independent of worker-local cache allocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _model_identity(model: Any) -> dict[str, Any]:
    hf_config = getattr(model, "hf_config", None)
    if hf_config is None:
        config = {}
    elif isinstance(hf_config, Mapping):
        config = dict(hf_config)
    else:
        config = hf_config.to_dict()
    return {
        "model": str(model.model).rstrip("/"),
        "revision": getattr(model, "revision", None),
        "code_revision": getattr(model, "code_revision", None),
        "hf_commit": getattr(hf_config, "_commit_hash", None),
        "dtype": str(getattr(model, "dtype", None)),
        "quantization": getattr(model, "quantization", None),
        "config": config,
    }


def build_cache_namespace(vllm_config: Any) -> str:
    """Bind keys to weights and representation without hashing runtime state.

    Group IDs, physical page sizes and PP coordinates belong to each wire key,
    not this identity: scheduler groups are global, worker groups are projected,
    and EngineCore rewrites ``cache_config.block_size`` after worker startup.
    Request prefix hashes (including their existing salt) remain unchanged.

    ``cache_namespace`` is an additional deployment identity, not an override.
    Change it when replacing weights in-place without a revision/commit change.
    New namespaces intentionally never probe the old unversioned key space.
    """
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None) or {}
    namespace = extra.get("cache_namespace", "")
    if not isinstance(namespace, str):
        raise TypeError("AscendStore cache_namespace must be a string")

    # The existing non-layerwise TP-mismatch path splits keys at the larger
    # P/D TP size. Both roles must retain the same namespace for those subkeys.
    tp_sizes = [parallel.tensor_parallel_size]
    for field in ("prefill_tp_size", "decode_tp_size"):
        value = extra.get(field)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"AscendStore {field} must be a positive integer")
            tp_sizes.append(value)
    identity = {
        "model": _model_identity(vllm_config.model_config),
        "cache": {
            field: getattr(cache, field, None)
            for field in (
                "cache_dtype",
                "kv_cache_layout",
                "kv_cache_dtype_skip_layers",
                "mamba_cache_dtype",
                "mamba_ssm_cache_dtype",
                "mamba_cache_mode",
                "use_replayssm",
                "replayssm_buffer_len",
            )
        },
        "parallel": {
            "effective_tp_size": max(tp_sizes),
            "dcp_size": getattr(parallel, "decode_context_parallel_size", 1),
            "pcp_size": getattr(parallel, "prefill_context_parallel_size", 1),
        },
        "deployment": namespace,
    }
    # No repr/default=str fallback: process-specific object reprs would silently
    # split reader/writer namespaces. HF to_dict() supplies JSON-compatible data.
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "ascend_store_v2:" + hashlib.sha256(encoded).hexdigest()
