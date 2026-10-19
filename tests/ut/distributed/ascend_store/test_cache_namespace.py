# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared-pool key identity checks without an NPU or Mooncake server."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    bind_layerwise_keys,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_namespace import build_cache_namespace
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
)


class HFConfig:
    def __init__(self):
        self._commit_hash = "commit-a"
        self.hidden_size = 128

    def to_dict(self):
        return {"model_type": "test", "hidden_size": self.hidden_size, "_commit_hash": self._commit_hash}


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="company-a/model",
            revision="revision-a",
            code_revision=None,
            hf_config=HFConfig(),
            dtype="bfloat16",
            quantization=None,
            compute_hash=lambda: "model-graph",
            get_layers_start_end_indices=lambda p: ((0, 2), (2, 4))[p.rank // p.tensor_parallel_size],
            get_total_num_hidden_layers=lambda: 4,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            rank=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=32, cache_dtype="auto", compute_hash=lambda: "cache-layout"),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        speculative_config=None,
    )


def test_namespace_isolates_model_revision_representation_and_explicit_generation():
    config = make_config()
    original = build_cache_namespace(config)
    assert original.startswith("ascend_store_v2:")
    for target, field, value in (
        ("model_config", "model", "company-b/model"),
        ("model_config", "revision", "revision-b"),
        ("model_config", "dtype", "float16"),
        ("model_config", "quantization", "fp8"),
        ("cache_config", "cache_dtype", "fp8"),
    ):
        other = deepcopy(config)
        setattr(getattr(other, target), field, value)
        assert build_cache_namespace(other) != original
    for field, value in (("_commit_hash", "commit-b"), ("hidden_size", 256)):
        other = deepcopy(config)
        setattr(other.model_config.hf_config, field, value)
        assert build_cache_namespace(other) != original
    config.kv_transfer_config.kv_connector_extra_config["cache_namespace"] = "weights-generation-2"
    assert build_cache_namespace(config) != original


def test_scheduler_worker_identity_ignores_runtime_projection_and_normalizes_tp_subkeys():
    scheduler = make_config()
    worker = deepcopy(scheduler)
    scheduler.cache_config.block_size = 2
    scheduler.cache_config.num_gpu_blocks = 1024
    worker.parallel_config.rank = 3
    worker.cache_config.num_gpu_blocks = 64
    assert build_cache_namespace(scheduler) == build_cache_namespace(worker)

    # Existing P4/D2 subkey sharing uses four effective heads on both sides.
    scheduler.parallel_config.tensor_parallel_size = 4
    scheduler.kv_transfer_config.kv_connector_extra_config["decode_tp_size"] = 2
    worker.kv_transfer_config.kv_connector_extra_config["prefill_tp_size"] = 4
    assert build_cache_namespace(scheduler) == build_cache_namespace(worker)

    scheduler.parallel_config.pipeline_parallel_size = worker.parallel_config.pipeline_parallel_size = 2
    worker.parallel_config.tensor_parallel_size = 4
    worker.parallel_config.rank = 4
    global_group = SimpleNamespace(layer_names=["layer0", "layer2"], kv_cache_spec=SimpleNamespace(block_size=16))
    local_group = SimpleNamespace(layer_names=["layer2"], kv_cache_spec=SimpleNamespace(block_size=16))
    scheduler_keys = bind_layerwise_keys(
        vllm_config=scheduler,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[global_group]),
        model_name="model",
        use_hybrid=True,
        grouped_block_size=[16],
    )
    worker_keys = bind_layerwise_keys(
        vllm_config=worker,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[local_group]),
        model_name="model",
        use_hybrid=True,
        grouped_block_size=[16],
    )
    assert scheduler_keys.make_full_key(0, "salted-prefix", 0, 1) == worker_keys.make_full_key(0, "salted-prefix", 0, 1)


def test_pool_keys_and_fast_token_prefix_preserve_namespace_and_page_size():
    metadata = KeyMetadata("model", 0, 0, 0, cache_namespace=build_cache_namespace(make_config()), block_size=16)
    key = PoolKey(metadata, "salted-prefix")
    database = ChunkedTokenDatabase([metadata], [16], None)
    assert database._get_key_prefix(0) + "salted-prefix" == key.to_string()
    assert database._make_key_by_hash("salted-prefix").to_string() == key.to_string()
    assert key.to_string() != PoolKey(replace(metadata, cache_namespace=""), "salted-prefix").to_string()
    different_page = PoolKey(replace(metadata, block_size=32), "salted-prefix")
    assert key.to_string() != different_page.to_string()
    assert key.split_layers(1)[0].to_string() != different_page.split_layers(1)[0].to_string()


def test_pp_one_layerwise_keys_snapshot_identity_and_do_not_reuse_legacy_objects():
    config = make_config()
    sizes = [16]

    def bind():
        return bind_layerwise_keys(
            vllm_config=config,
            kv_cache_config=None,
            model_name="model",
            use_hybrid=False,
            grouped_block_size=sizes,
        )

    builder = bind()
    key = builder.make_full_key(0, "salted-prefix", 0, 0)
    assert "ascend_store_v2:" in key
    assert "block_size:16" in key
    assert key.endswith("@salted-prefix@0")
    assert key != "model@salted-prefix@0"
    sizes[0] = 32
    config.model_config.revision = "revision-b"
    assert builder.make_full_key(0, "salted-prefix", 0, 0) == key
    assert bind().make_full_key(0, "salted-prefix", 0, 0) != key
