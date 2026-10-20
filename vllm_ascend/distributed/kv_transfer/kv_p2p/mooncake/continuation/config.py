# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded, opt-in configuration for WAN PD continuation."""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class WanPDConfig:
    enabled: bool = False
    receiver_first: bool = False
    layerwise_publication: bool = False
    structured_input_identity: bool = False
    reservation_callback_url: str | None = None
    reservation_workers: int = 4
    processor_fingerprint: str | None = None
    max_pending_requests: int = 128
    max_active_requests: int = 8
    max_retained_source_requests: int = 1024
    staging_slots: int = 8
    chunk_bytes: int = 2 * 1024 * 1024
    max_inflight_bytes: int = 16 * 1024 * 1024
    max_descriptors_per_chunk: int = 256
    timeout_s: float = 300.0
    commit_retry_limit: int = 3
    commit_retry_interval_s: float = 0.1
    max_manifest_bytes: int = 2 * 1024 * 1024
    max_manifest_components: int = 4096
    max_manifest_ranges: int = 8192
    max_prefix_hashes: int = 65536
    max_terminal_requests: int = 1024
    cache_namespace: str = ""
    egress_endpoint: str | None = None

    def __post_init__(self) -> None:
        for name in ("enabled", "receiver_first", "layerwise_publication", "structured_input_identity"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"wan_pd.{name} must be a boolean")
        if not self.enabled and (self.receiver_first or self.layerwise_publication or self.structured_input_identity):
            raise ValueError("WAN PD components require wan_pd.enabled")
        if self.layerwise_publication and not self.receiver_first:
            raise ValueError("wan_pd.layerwise_publication requires receiver_first")
        limits = {
            "reservation_workers": 128,
            "max_pending_requests": 65536,
            "max_active_requests": 65536,
            "max_retained_source_requests": 1048576,
            "staging_slots": 65536,
            "chunk_bytes": 1024 * 1024 * 1024,
            "max_inflight_bytes": 1024**4,
            "max_descriptors_per_chunk": 65536,
            "commit_retry_limit": 128,
            "max_manifest_bytes": 64 * 1024 * 1024,
            "max_manifest_components": 65536,
            "max_manifest_ranges": 1048576,
            "max_prefix_hashes": 1048576,
            "max_terminal_requests": 1048576,
        }
        for name, maximum in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError(f"wan_pd.{name} must be an integer in [1, {maximum}]")
        if self.max_active_requests > self.max_pending_requests:
            raise ValueError("wan_pd.max_active_requests cannot exceed max_pending_requests")
        if not self.chunk_bytes <= self.max_inflight_bytes <= self.staging_slots * self.chunk_bytes:
            raise ValueError("wan_pd requires chunk_bytes <= max_inflight_bytes <= staging_slots * chunk_bytes")
        for name in ("timeout_s", "commit_retry_interval_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"wan_pd.{name} must be finite and positive")
        if not isinstance(self.cache_namespace, str) or len(self.cache_namespace) > 4096:
            raise ValueError("wan_pd.cache_namespace must be a string of at most 4096 characters")
        if self.egress_endpoint is not None and (
            not isinstance(self.egress_endpoint, str) or not 0 < len(self.egress_endpoint) <= 4096
        ):
            raise ValueError("wan_pd.egress_endpoint must be a nonempty string or null")
        if self.reservation_callback_url is not None:
            if not isinstance(self.reservation_callback_url, str) or len(self.reservation_callback_url) > 4096:
                raise ValueError("wan_pd.reservation_callback_url must be a bounded HTTP(S) URL")
            parsed = urlsplit(self.reservation_callback_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.fragment:
                raise ValueError("wan_pd.reservation_callback_url must be an absolute HTTP(S) URL")
        if self.processor_fingerprint is not None and (
            not isinstance(self.processor_fingerprint, str) or not 0 < len(self.processor_fingerprint) <= 4096
        ):
            raise ValueError("wan_pd.processor_fingerprint must be a bounded nonempty string")

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any] | None) -> "WanPDConfig":
        values = (extra_config or {}).get("wan_pd", {})
        if not isinstance(values, dict):
            raise ValueError("kv_connector_extra_config.wan_pd must be an object")
        unknown = values.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown wan_pd configuration fields: {sorted(unknown)}")
        return cls(**values)
