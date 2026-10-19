# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Configuration for the bounded pipeline transport."""

from pydantic import ConfigDict, Field, model_validator

from vllm_ascend.config_utils import config


@config(config=ConfigDict(frozen=True))
class WanPPConfig:
    """Per-worker staging capacity, configured with ``wan_pp_config``."""

    enabled: bool = False
    num_slots: int = Field(default=2, gt=0, strict=True)
    slot_bytes: int = Field(default=64 * 1024 * 1024, gt=0, strict=True)
    max_inflight_bytes: int = Field(default=128 * 1024 * 1024, gt=0, strict=True)
    timeout_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    max_metadata_bytes: int = Field(default=64 * 1024, gt=0, strict=True)

    @model_validator(mode="after")
    def _validate_capacity(self):
        if not self.slot_bytes <= self.max_inflight_bytes <= self.num_slots * self.slot_bytes:
            raise ValueError("wan_pp_config requires slot_bytes <= max_inflight_bytes <= num_slots * slot_bytes.")
        return self

    def validate_parallel_config(self, parallel_config) -> None:
        if not self.enabled:
            return
        if parallel_config.pipeline_parallel_size <= 1:
            raise ValueError("wan_pp_config.enabled requires pipeline_parallel_size > 1.")
        if parallel_config.distributed_executor_backend == "external_launcher":
            raise ValueError("wan_pp_config requires the built-in PP worker path; external_launcher is unsupported.")
