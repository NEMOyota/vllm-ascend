# WAN PD continuation

This experimental feature targets **E → cloud P → edge D**. Encoding remains
in the existing E path; this change strengthens the handoff from Prefill to
Decode in `MooncakeConnectorV2` / `MooncakePullConnector`. It is opt-in and
requires the managed asynchronous batch-ticket API from the matching Mooncake
`experiment/wan-pd-continuation` branch. Legacy V1/Hybrid connectors are unchanged.

The Ascend experiment starts at `a5c0a61d3` (WAN PP joint scheduling); the paired
Mooncake implementation is `d340e89`, based on `a2c6c5b701`. Both repositories
use `experiment/wan-pd-continuation`; the previous PP branch remains available.

## Ownership and progression

| Resource | Admission | Release |
| --- | --- | --- |
| P request and source KV pin | Before scheduling prefill; bounded source ledger | One valid COMMIT after all D workers finish |
| D request and completion outbox | Before advertising external tokens | Completion notification acknowledged by P |
| D registered staging slots | Fixed allocation at KV registration | Successful READ and local copy event complete |
| Network bytes and optional shared egress grant | Before each READ chunk | Managed batch completes with the full expected byte count |

The D progress owner advances a bounded number of active requests in rotation.
It splits physical ranges into network chunks, with byte and descriptor limits,
and polls tickets and copy events without a synchronous whole-request READ.
Chunking is independent of compute chunks; this version makes the request
decode-ready only after all its required chunks and state copies finish.
Metadata discovery and native transport submission may still block internally;
the bounds are on resources and progression work, not a real-time latency SLA.

READ destinations are always registered staging slots. Completed chunks are
copied into the reserved D KV allocation on a separate NPU stream, and their
events are polled before slots are recycled. An uncertain or late DMA can
therefore only touch an isolated staging slot, never a freed D cache block.
An incomplete request is not exposed as a valid cache hit. Device-copy failures
stop the worker's progress and surface through its result interface; they are
not reported as safely reusable destinations.

The worker's existing global transfer-engine owner retains KV and staging
allocations for the process lifetime. A connector object being destroyed does
not free registered memory. Replacing a registered pool requires a new worker
process. This version intentionally has no automatic drain/reconnect protocol.

## Manifest and commit

The P scheduler captures input identity **before** any hybrid-state prompt
truncation. After prefill it publishes an immutable manifest containing:

- Original input identity and length, actual state position, and model/cache
  namespace.
- Each layer's state kind, dtype, block shape, coverage interval and recurrent
  checkpoint position. Attention, convolution and SSM state must agree on the
  continuation boundary.
- Source engine epoch, unique attempt ID, registered layout fingerprint and a
  fingerprint binding the physical block table and source endpoint.
- Prefix hashes with their actual hash-block granularity, separate from the
  scheduler/cache allocation block size.

D checks the identity and prefix before claiming external tokens, then checks
the remote epoch, physical layout and complete local state set before posting
READs. Mismatched scheduler metadata falls back to local computation; it never
advertises that mismatched data as a remote hit. A failed worker follows the
existing vLLM KV-load failure policy. Local computation fallback may cost more
on an edge device and should be measured with the intended workload.

An application COMMIT contains `(engine_epoch, attempt_id, manifest_digest)`
and is sent only after vLLM aggregates successful completion from all D workers.
P releases its pin exactly once. Recent COMMIT duplicates are validated against a bounded terminal history.
A positive COMMIT for an absent attempt in the same engine epoch is acknowledged
as a no-op: outstanding source pins are never evicted from the active table,
so this handles lost ACKs after terminal-history eviction without releasing
another request's blocks. Stale epochs and mismatched active attempts are rejected. Failed transfers send an isolation outcome instead of the legacy DONE
notification. Timeout alone never frees P state or uncertain staging memory.

Completion RPCs use bounded reservations, finite retries and backoff. Exhausted
notifications retain their reservation; failed or abandoned source pins also
remain counted. Capacity exhaustion applies admission backpressure. Recovery
requires transport/process teardown before reclaiming uncertain memory. There
is no durable transaction log or transparent restart replay in this version.
The manifest digest detects control-plane inconsistency; it is not peer
authentication or a checksum over device payloads.

## Configuration

Use the same model and representation configuration on P and D. Add `wan_pd`
inside the existing **`kv_connector_extra_config`** field of
`--kv-transfer-config`. For P:

```json
{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_producer",
  "kv_port": 14579,
  "kv_connector_extra_config": {
    "cache_namespace": "enterprise-model-revision-1",
    "wan_pd": {
      "enabled": true,
      "max_pending_requests": 128,
      "max_active_requests": 8,
      "max_retained_source_requests": 1024,
      "staging_slots": 8,
      "chunk_bytes": 2097152,
      "max_inflight_bytes": 16777216,
      "max_descriptors_per_chunk": 256,
      "timeout_s": 300,
      "commit_retry_limit": 3,
      "commit_retry_interval_s": 0.1
    }
  }
}
```

Set `kv_role` to `kv_consumer` for D. The D staging allocation is
`staging_slots × chunk_bytes` bytes per worker, 16 MiB with these defaults.
It is reserved in automatic KV memory sizing. With explicit
`kv_cache_memory_bytes`, leave room for staging in addition to that explicit KV
allocation. Descriptor, manifest and prefix-hash limits are independently
configurable in `WanPDConfig`; long-context deployments should size them using
their actual layer/page layout.

Ordinary token requests derive identity from the complete prompt and cache salt.
For embedding, multimodal or LoRA input, the E/router path must provide the
same immutable `kv_transfer_params.wan_pd_input_fingerprint` on P and D, binding
content, positions and adapter revision. The response preserves this field and
`continuation_manifest`; proxies must forward them intact. This first version
does not reuse local prefixes for such special inputs. Missing input identity
declines the P handoff and returns `wan_pd_error`, without terminating unrelated
requests. It uses
`request.num_prompt_tokens`, including embedding-only input, rather than the
length of an absent token-ID list.

Current compatibility requires equal P/D TP, PCP and DCP sizes, matching cache
groups, dtypes, shapes, strides and block scales. Existing address mapping is
reused. Prefix negotiation uses a common recoverable hit across cache groups;
independently divergent group hits are disabled in this mode. PP boundaries
may differ when these per-layer contracts match. KVPP
scratch ownership and new model-specific state conversion are not enabled by
this feature. It is a framework protocol, not a claim of tested support for
every hybrid model.

## Shared egress and future communication work

Optional `wan_pd.egress_endpoint` uses the same service as
[WAN PP joint scheduling](wan_pipeline_joint_scheduling.md). PD READs consume
bulk-traffic credits; PP latency traffic can retain its configured reserve.
For a READ, the data exits **P**, although D submits the operation: configure
the service for that source egress and include every competing flow in the same
budget. A per-D service does not enforce a cloud-wide source limit. Control RPC
distance is an additional cost to measure. The framework cannot account for
traffic that bypasses this service.

Managed tickets provide a future integration point for HCCL/HCOMM/HIXL-backed
transfer work: the transport must preserve whole-batch completion, descriptor
lifetime and failure quiescence semantics. This branch changes Mooncake rather
than introducing another overlapping HCCL/HCOMM/HIXL scheduler. It does not add
PD state compression, cross-request deduplication or an eviction coordinator.

## Shared storage index identity

The accompanying AscendStore change isolates keys by full model identity,
revision/HF commit, model and KV dtypes, quantization and stable representation
configuration. Scheduler and worker use the same namespace despite local PP
group projection or engine block-size rewriting. Each non-layerwise group key
also includes its actual page size. Mooncake PP=1 keys are versioned too.

This intentionally cold-starts the cache: old and new versions do not share
legacy keys. Upgrade readers and writers together. Replacing weights in place
without a revision change requires a new `cache_namespace`. Changing an
unsupported non-layerwise PP partition also requires a namespace change.
Existing readable/commit checks remain in place; this is namespace correctness,
not a new global index-coherence or multi-tier eviction protocol.

## Focused validation

The CPU tests exercise production manifest, scheduler, progress and staging
code. They use controlled transport outcomes, real host byte copies and the
existing address mapper; they do not simulate NPU/RoCE performance.

```bash
python -m pytest --noconftest -q \
  tests/ut/distributed/kv_transfer/kv_p2p/test_wan_pd_manifest.py \
  tests/ut/distributed/mooncake/test_wan_pd_scheduler.py \
  tests/ut/distributed/mooncake/test_wan_pd_progress.py \
  tests/ut/distributed/ascend_store/test_cache_namespace.py \
  tests/ut/distributed/ascend_store/test_layerwise_keys.py
```

This focused set passed **27 tests and 10 subtests** in the CPU environment.
The five Mooncake C++ lifecycle cases also passed. Changed-file formatting and
lint checks passed. The connector's `get_wan_pd_stats()` provides local source,
outbox or worker progression counters for diagnostics; these are not new
Prometheus metrics.

Mooncake includes a hardware-free C++ lifecycle test and a separate native TCP
smoke script. The latter requires building its Python extension. The present
cloud environment has no NPU/CANN/RoCE stack, and lacks the native dependencies
for the full Mooncake build. Hardware correctness, payload visibility and
end-to-end latency/throughput remain laboratory validation items.

For a first hardware run, compare disabled/enabled PD with identical inputs and
prefix settings; verify tokens/logits, state bytes and no early D execution.
Then inject delayed/error completions and lost COMMIT replies, checking staging
occupancy, source pins and unrelated request progress. Finally sweep chunk/window
sizes under the intended RTT and competing PP traffic; report TTFT, queue time,
ITL and memory occupancy rather than only transfer bandwidth.
