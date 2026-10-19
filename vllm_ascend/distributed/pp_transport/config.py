# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Configuration for the bounded pipeline transport."""

from dataclasses import field

from pydantic import ConfigDict, Field, model_validator

from vllm_ascend.config_utils import config


@config(config=ConfigDict(frozen=True))
class WanPPJointConfig:
    """Static compute/communication policy; sizes describe each PP lane."""

    enabled: bool = False
    compute_chunk_tokens: int = Field(default=1024, gt=0, strict=True)
    estimated_boundary_bytes_per_token: int = Field(default=0, ge=0, strict=True)
    boundary_fixed_bytes: int = Field(default=0, ge=0, strict=True)
    estimated_wire_bytes_per_token: int = Field(default=0, ge=0, strict=True)
    fragment_bytes: int = Field(default=1024 * 1024, gt=0, strict=True)
    window_fragments: int = Field(default=8, gt=0, strict=True)
    window_bytes: int = Field(default=8 * 1024 * 1024, gt=0, strict=True)
    estimated_link_bytes_per_s: int = Field(default=0, ge=0, strict=True)
    estimated_rtt_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    target_chunk_ms: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    egress_endpoint: str = ""
    egress_rank_endpoints: dict[int, str] = field(default_factory=dict)
    egress_rpc_timeout_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_window(self):
        if self.fragment_bytes > self.window_bytes:
            raise ValueError("wan_pp_config.joint_config requires fragment_bytes <= window_bytes.")
        if any(rank < 0 for rank in self.egress_rank_endpoints):
            raise ValueError("egress_rank_endpoints keys must be nonnegative global source ranks.")
        return self


@config(config=ConfigDict(frozen=True))
class WanPPConfig:
    """Per-worker staging capacity, configured with ``wan_pp_config``."""

    enabled: bool = False
    num_slots: int = Field(default=2, gt=0, strict=True)
    slot_bytes: int = Field(default=64 * 1024 * 1024, gt=0, strict=True)
    max_inflight_bytes: int = Field(default=128 * 1024 * 1024, gt=0, strict=True)
    timeout_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    max_metadata_bytes: int = Field(default=64 * 1024, gt=0, strict=True)
    joint_config: WanPPJointConfig = field(default_factory=WanPPJointConfig)

    @model_validator(mode="after")
    def _validate_capacity(self):
        if not self.slot_bytes <= self.max_inflight_bytes <= self.num_slots * self.slot_bytes:
            raise ValueError("wan_pp_config requires slot_bytes <= max_inflight_bytes <= num_slots * slot_bytes.")
        if self.joint_config.enabled and not self.enabled:
            raise ValueError("wan_pp_config.joint_config.enabled requires wan_pp_config.enabled.")
        if self.joint_config.enabled and self.joint_config.boundary_fixed_bytes >= self.slot_bytes:
            raise ValueError("wan_pp_config.joint_config.boundary_fixed_bytes must leave room in slot_bytes.")
        return self

    def validate_parallel_config(self, parallel_config) -> None:
        if not self.enabled:
            return
        if parallel_config.pipeline_parallel_size <= 1:
            raise ValueError("wan_pp_config.enabled requires pipeline_parallel_size > 1.")
        if parallel_config.distributed_executor_backend == "external_launcher":
            raise ValueError("wan_pp_config requires the built-in PP worker path; external_launcher is unsupported.")
