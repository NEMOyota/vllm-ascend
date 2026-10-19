# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compute chunk planning without model-specific tensor or scheduler rewrites."""

import math
from dataclasses import dataclass
from typing import Any

from vllm_ascend.distributed.pp_transport.config import WanPPConfig

# These implementations consume max_num_scheduled_tokens before allocating KV
# blocks. Unknown scheduler plugins must establish the same contract first.
_SUPPORTED_SCHEDULERS = frozenset(
    {
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_ascend.core.short_request_first_scheduler.ShortRequestFirstAsyncScheduler",
        "vllm_ascend.core.scheduler_profiling_chunk.ProfilingChunkScheduler",
        "vllm_ascend.core.scheduler_profiling_chunk.ProfilingChunkAsyncScheduler",
        "vllm_ascend.core.batch_job_aware_scheduler.BatchJobAwareScheduler",
        "vllm_ascend.core.batch_job_aware_scheduler.BatchJobAwareAsyncScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.AsyncDyntraLBScheduler",
        "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
        "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler",
        "vllm_ascend.core.recompute_scheduler.DyntraLBRecomputeScheduler",
        "vllm_ascend.core.recompute_scheduler.AsyncDyntraLBRecomputeScheduler",
    }
)


@dataclass(frozen=True)
class WanPPPlan:
    """Static estimates, separate from actual transport admission and DMA state."""

    compute_chunk_tokens: int
    estimated_boundary_bytes: int | None
    estimated_wire_bytes: int | None
    fragment_bytes: int
    window_fragments: int
    window_bytes: int
    bandwidth_delay_product_bytes: int


def plan_wan_pp_chunks(config: WanPPConfig, scheduler_token_limit: int) -> WanPPPlan:
    """Plan one compute batch; frames larger than the wire window may stream."""
    if scheduler_token_limit <= 0:
        raise ValueError("The scheduler token limit must be positive.")
    joint = config.joint_config
    token_cap = min(scheduler_token_limit, joint.compute_chunk_tokens)
    fixed_bytes = joint.boundary_fixed_bytes
    boundary_per_token = joint.estimated_boundary_bytes_per_token
    wire_per_token = joint.estimated_wire_bytes_per_token or boundary_per_token
    if boundary_per_token:
        token_cap = min(token_cap, (config.slot_bytes - fixed_bytes) // boundary_per_token)
    if joint.estimated_link_bytes_per_s and wire_per_token:
        target_wire_bytes = math.floor(joint.estimated_link_bytes_per_s * joint.target_chunk_ms / 1000)
        token_cap = min(token_cap, (target_wire_bytes - fixed_bytes) // wire_per_token)
    if token_cap <= 0:
        raise ValueError(
            "WAN PP compute budget cannot fit one estimated token. Increase slot_bytes/target_chunk_ms "
            "or correct the boundary/wire byte estimates."
        )
    return WanPPPlan(
        compute_chunk_tokens=token_cap,
        estimated_boundary_bytes=fixed_bytes + token_cap * boundary_per_token if boundary_per_token else None,
        estimated_wire_bytes=fixed_bytes + token_cap * wire_per_token if wire_per_token else None,
        fragment_bytes=joint.fragment_bytes,
        window_fragments=joint.window_fragments,
        window_bytes=min(joint.window_bytes, joint.window_fragments * joint.fragment_bytes),
        bandwidth_delay_product_bytes=math.ceil(joint.estimated_link_bytes_per_s * joint.estimated_rtt_ms / 1000),
    )


def apply_wan_pp_compute_budget(vllm_config: Any, config: WanPPConfig) -> WanPPPlan | None:
    """Install a shared scheduler cap before constructing the engine/workers.

    KV allocation, block alignment, multimodal boundaries and asynchronous
    scheduling stay in the selected scheduler. Graph padding and extra output
    tensors still require the transport's actual frame capacity validation.
    """
    if not config.joint_config.enabled:
        return None
    config.validate_parallel_config(vllm_config.parallel_config)
    scheduler = vllm_config.scheduler_config
    scheduler_cls = scheduler.scheduler_cls
    if isinstance(scheduler_cls, type):
        scheduler_cls = f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}"
    if scheduler_cls is not None and scheduler_cls not in _SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"WAN PP compute chunking has not validated scheduler_cls={scheduler_cls!r}. "
            "Use a supported scheduler that honors max_num_scheduled_tokens."
        )
    if getattr(vllm_config, "diffusion_config", None) is not None:
        raise ValueError("WAN PP compute chunking does not split indivisible diffusion canvases.")
    limit = min(
        scheduler.max_num_batched_tokens,
        scheduler.max_num_scheduled_tokens or scheduler.max_num_batched_tokens,
    )
    plan = plan_wan_pp_chunks(config, limit)
    model_config = vllm_config.model_config
    if not scheduler.enable_chunked_prefill and plan.compute_chunk_tokens < model_config.max_model_len:
        raise ValueError(
            "WAN PP compute chunking below max_model_len requires enable_chunked_prefill; "
            "otherwise a long prompt can never be admitted."
        )
    if (
        plan.compute_chunk_tokens < limit
        and getattr(model_config, "supports_multimodal_inputs", False)
        and scheduler.disable_chunked_mm_input
    ):
        raise ValueError(
            "WAN PP cannot lower the compute budget with disable_chunked_mm_input=True for a multimodal model: "
            "an indivisible multimodal item may never be admitted. Allow multimodal input chunking if the model "
            f"supports it, or keep the effective compute budget at the original {limit} tokens."
        )
    minimum_tokens = max(
        1 + (getattr(vllm_config, "num_speculative_tokens", 0) or 0),
        getattr(vllm_config, "num_prefill_lookahead_tokens", 0) or 0,
    )
    if plan.compute_chunk_tokens < minimum_tokens:
        raise ValueError(
            f"WAN PP compute cap {plan.compute_chunk_tokens} is smaller than the speculative/prefill "
            f"lookahead requirement {minimum_tokens}; increase the compute/byte budget."
        )
    scheduler.max_num_scheduled_tokens = plan.compute_chunk_tokens
    return plan
