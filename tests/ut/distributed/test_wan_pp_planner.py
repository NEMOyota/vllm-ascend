# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for static chunk planning and actual upstream scheduler admission."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ObservabilityConfig, ParallelConfig, SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.distributed.pp_transport.config import WanPPConfig
from vllm_ascend.distributed.pp_transport.planner import apply_wan_pp_compute_budget, plan_wan_pp_chunks


def _transport_config(**joint_overrides):
    return WanPPConfig(
        enabled=True,
        slot_bytes=4096,
        max_inflight_bytes=8192,
        joint_config={
            "enabled": True,
            "compute_chunk_tokens": 64,
            "fragment_bytes": 128,
            "window_bytes": 512,
            "window_fragments": 4,
            **joint_overrides,
        },
    )


def _engine_config(async_scheduling=False):
    cache_config = CacheConfig(block_size=16, enable_prefix_caching=False)
    cache_config.num_gpu_blocks = 32
    return SimpleNamespace(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=128,
            max_model_len=256,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
            watermark=0.0,
        ),
        parallel_config=ParallelConfig(pipeline_parallel_size=2),
        model_config=SimpleNamespace(
            max_model_len=256,
            uses_mrope=False,
            is_encoder_decoder=False,
            runner_type="generate",
            is_diffusion=False,
            supports_multimodal_inputs=False,
            return_sampling_mask=False,
        ),
        cache_config=cache_config,
        observability_config=ObservabilityConfig(),
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        aux_output_config=SimpleNamespace(enabled=False),
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        is_mm_encoder_only=False,
        num_speculative_tokens=0,
        num_lookahead_tokens=0,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=384,
        max_concurrent_batches=3,
        use_v2_model_runner=True,
    )


def test_static_plan_separates_materialization_wire_window_and_bdp():
    config = _transport_config(
        estimated_boundary_bytes_per_token=128,
        estimated_wire_bytes_per_token=32,
        boundary_fixed_bytes=128,
        estimated_link_bytes_per_s=100_000,
        estimated_rtt_ms=20,
        target_chunk_ms=10,
    )
    plan = plan_wan_pp_chunks(config, 128)
    assert plan.compute_chunk_tokens == 27
    assert plan.estimated_boundary_bytes == 3584
    assert plan.estimated_wire_bytes == 992
    assert plan.window_bytes == 512 < plan.estimated_wire_bytes
    assert plan.bandwidth_delay_product_bytes == 2000
    # A slow wire estimate limits compute, not the receiver's full tensor size.
    slot_limited = plan_wan_pp_chunks(_transport_config(estimated_boundary_bytes_per_token=128), 128)
    assert slot_limited.compute_chunk_tokens == 32


def test_apply_preserves_stricter_scheduler_cap_and_runner_capacity():
    engine = _engine_config()
    engine.scheduler_config.max_num_scheduled_tokens = 16
    engine.scheduler_config.scheduler_cls = AsyncScheduler
    apply_wan_pp_compute_budget(engine, _transport_config())
    assert engine.scheduler_config.max_num_scheduled_tokens == 16
    assert engine.scheduler_config.max_num_batched_tokens == 128
    assert engine.scheduler_config.scheduler_cls is AsyncScheduler
    assert apply_wan_pp_compute_budget(engine, WanPPConfig()) is None


@pytest.mark.parametrize("constraint", ["whole_prompt", "speculative_step", "unknown_scheduler", "atomic_mm"])
def test_unadmittable_or_unverified_scheduler_fails_before_mutating_budget(constraint):
    engine = _engine_config()
    if constraint == "whole_prompt":
        engine.scheduler_config.enable_chunked_prefill = False
        message = "requires enable_chunked_prefill"
    elif constraint == "speculative_step":
        engine.num_speculative_tokens = 64
        message = "lookahead requirement"
    elif constraint == "atomic_mm":
        engine.model_config.supports_multimodal_inputs = True
        engine.scheduler_config.disable_chunked_mm_input = True
        message = "indivisible multimodal item may never be admitted"
    else:
        engine.scheduler_config.scheduler_cls = "custom.scheduler.UnknownScheduler"
        message = "has not validated scheduler_cls"
    with pytest.raises(ValueError, match=message):
        apply_wan_pp_compute_budget(engine, _transport_config())
    assert engine.scheduler_config.max_num_scheduled_tokens is None


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_compute_cap_drives_real_prefill_chunks_and_kv_allocation(scheduler_cls):
    engine = _engine_config(async_scheduling=scheduler_cls is AsyncScheduler)
    apply_wan_pp_compute_budget(engine, _transport_config(compute_chunk_tokens=16))
    register_all_kvcache_specs(engine)
    kv_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    scheduler = scheduler_cls(
        vllm_config=engine,
        kv_cache_config=kv_config,
        block_size=16,
        structured_output_manager=SimpleNamespace(),
    )
    request = Request("long-prefill", list(range(48)), SamplingParams(max_tokens=4), pooling_params=None)
    scheduler.add_request(request)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert first.num_scheduled_tokens == second.num_scheduled_tokens == {"long-prefill": 16}
    assert request.num_computed_tokens == 32
    assert request.num_output_placeholders == 0
    assert len(scheduler.kv_cache_manager.get_blocks("long-prefill").blocks[0]) == 2


def test_rejects_joint_window_and_estimate_that_cannot_admit_one_token():
    with pytest.raises(ValueError, match="fragment_bytes <= window_bytes"):
        _transport_config(fragment_bytes=1024)
    with pytest.raises(ValueError, match="cannot fit one estimated token"):
        plan_wan_pp_chunks(_transport_config(estimated_boundary_bytes_per_token=4097), 128)
