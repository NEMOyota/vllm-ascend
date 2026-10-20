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
| P request admission | Before scheduling prefill; bounded source ledger | One valid COMMIT after all D workers finish |
| P source KV allocation | Retained at final publication; early publication additionally pins physical blocks | COMMIT; an aborted early publication retains its pins |
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
truncation. It publishes an immutable manifest containing:

- Original input identity and length, actual state position, and model/cache
  namespace.
- Each layer's state kind, dtype, block shape, coverage interval and recurrent
  checkpoint position. Attention, convolution and SSM state must agree on the
  continuation boundary.
- Source engine epoch, unique attempt ID, registered layout fingerprint and a
  fingerprint binding the physical block table and source endpoint.
- Prefix hashes with their actual hash-block granularity, separate from the
  scheduler/cache allocation block size.
- When receiver-first mode is used, a fingerprint of the exact D allocation
  ticket. A layerwise manifest also identifies its publication mode.

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

## Optional legacy-project components

These switches add reusable components to the existing bounded PD path:

| `wan_pd` option | Default | Purpose |
| --- | --- | --- |
| `receiver_first` | `false` | Allocate D blocks before asking P to compute |
| `layerwise_publication` | `false` | READ eligible layers while P computes later layers; requires `receiver_first` |
| `structured_input_identity` | `false` | Derive special-input identity from supplied content/position metadata |

Enable matching features on P and D. With all three switches disabled, the
existing final-manifest WAN path remains in use. With `enabled=false`, the
original connector path remains in use; enabling a subcomponent without it is
rejected. No model-name-specific scheduler or old LWD RPC runtime is installed.

### Receiver-first admission

D reserves callback and COMMIT capacity, then the native scheduler allocates
its actual cache-group blocks. A unique ticket binds the receiver epoch,
request/attempt IDs, input identity, state position, prefix boundary and
allocation fingerprint. D remains in `WAITING_FOR_REMOTE_KVS` while its callback
starts P. It advertises **expected** external tokens to reserve their storage;
it does not claim those bytes are already readable.

D validates the returned publication's ticket and manifest, then enters the
same READ → staging → copy-event → COMMIT path. Repeated callbacks reuse the
same attempt; a different allocation cannot replace it. Callback concurrency,
retained results, payload bytes and deadlines are bounded. Expired or cancelled
callbacks cannot publish into a newer allocation. Expired network operations
still follow the quarantine rules. A complete local-prefix hit bypasses P.

The single-process router
`examples/disaggregated_prefill_v1/wan_pd_receiver_first_proxy.py` accepts
single-prompt completions, token-ID prompts and chat completions with `n=1`;
multi-prompt batches are explicitly rejected. It keeps D's original sampling
request and does not append P's sampled token. Native D replay recomputes the
last prompt token when needed to produce the first output.

For example, use these P and D `--kv-transfer-config` JSON values.
P also needs `--enforce-eager --no-async-scheduling` for early layers:

```json
{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_producer",
  "kv_port": 14579,
  "kv_connector_extra_config": {
    "cache_namespace": "enterprise-model-revision-1",
    "wan_pd": {
      "enabled": true,
      "receiver_first": true,
      "layerwise_publication": true
    }
  }
}
```

```json
{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_consumer",
  "kv_port": 24579,
  "kv_connector_extra_config": {
    "cache_namespace": "enterprise-model-revision-1",
    "wan_pd": {
      "enabled": true,
      "receiver_first": true,
      "layerwise_publication": true,
      "reservation_workers": 4,
      "reservation_callback_url": "http://10.0.1.10:9000/v1/wan-pd/reservation"
    }
  }
}
```

Start the router on `10.0.1.10`, with P at `10.0.0.11:8000` and D at
`10.0.1.12:8000`. This example assumes P has DP=PP=PCP=TP=1:

```bash
python examples/disaggregated_prefill_v1/wan_pd_receiver_first_proxy.py \
  --host 0.0.0.0 --port 9000 \
  --prefill-url http://10.0.0.11:8000 \
  --decode-url http://10.0.1.12:8000 \
  --max-pending 128 --timeout-s 300 \
  --layerwise-publication --producer-control-address 10.0.0.11:14580
```

The scheduler control port is
`kv_port + dp_size * pp_size * pcp_size * tp_size + dp_rank`, after the worker
handshake port range. Use P's reachable scheduler address, not its HTTP port.
Both endpoints need matching model/weight revisions, cache namespace, hash
granularity and representation layout. The example routes one logical P/D
pair; DP routing requires a rendezvous-aware dispatcher per pair.

For D-first admission alone, set `layerwise_publication=false` at both engines
and omit the final two router options. The callback then returns after P's
final publication. Callback URL and producer control address are administrator
settings; incoming requests cannot choose them.

### Layer readiness and safe overlap

For eligible requests, P prepares one immutable plan during the **last prefill
compute chunk**, takes additional references to its actual block-pool blocks,
and exposes a `PREPARED` catalog entry. Layer-save hooks record device events;
only completed events make those layers readable. D can READ a ready layer
while later P layers execute, using the existing bounded window and staging
pool. The catalog must reach `SEALED`, every required layer must arrive, and
all local copy events must finish before D reports receive completion or sends
COMMIT. Preemption/abort invalidates the publication and retains source pins,
so a still-running READ cannot observe blocks recycled to another request.

The first early-publication path requires full-attention state, eager and
synchronous P execution, PCP=DCP=1, no MTP/speculation, no DBO/microbatch overlap,
and a one-token P response. It does not expose intermediate chunk state that
later computation can overwrite. Mixed Mamba/SSM, sliding/circular caches,
MTP and other ineligible configurations continue through **final publication**;
enabling the switch does not claim early overlap for those configurations.
Unknown layer hooks complete at the final worker fence rather than being
guessed from layer order.
Transfer ranges retain the registered layer order; a later layer at the head
of that order can delay ready layers behind it. Measure the first READ time and
actual overlap for each model rather than assuming registration order matches
execution order.

Native PP metadata merges layers across the logical P's workers, including an
empty E-stage layer table. This keeps E-edge/P-cloud wiring generic and permits
matching P/D layouts with different PP boundaries. Empty E stages combined
with PD have source-level support here; their NPU/WAN execution has not been
validated in this environment.

### Structured special-input identity

With `structured_input_identity=true`, the trusted E/ingestion path may provide
`kv_transfer_params.wan_pd_input_identity` on both requests. Version 1 supports:

- `kind="multimodal"`: `processor_fingerprint` and a `media` list containing
  each item's SHA256, modality, token offset and length. Prompt tokens also
  participate in the derived identity.
- `kind="prompt_embeds"`: SHA256, `[prompt_tokens, hidden_size]` shape, dtype
  and `encoder_fingerprint`; optionally `processor_fingerprint`.

Optional `wan_pd.processor_fingerprint` requires an exact preprocessing-ABI
match. Namespace and cache salt retain tenant/revision separation. This checks
metadata but does not hash device tensors or assume that a user-provided
multimodal cache key is a content digest. LoRA, mixed embedding inputs and
unsupported media-position semantics keep the explicit
`wan_pd_input_fingerprint` route. Special-input prefix reuse remains disabled.

The legacy PD demo supplied the E/P/D wiring and lifecycle lessons; its
layerwise transfers and several state mappers already came from the upstream
Mooncake layerwise connector. This port reuses those workflow semantics within
the bounded Pull protocol, rather than importing the model-specific launcher
or unbounded queue. Empty-batch progress, deferred MTP finalization and complete
hybrid D block tables were already present in the current baseline. The old
prefix/MM branches reused cloud-local KV; neither this identity component nor
the namespace changes implement a distributed multi-tier cache directory.

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
lifetime and failure quiescence semantics. The initial branch changes Mooncake;
the optional components reuse that managed-ticket implementation without
further Mooncake changes. They do not modify HCCL, HCOMM or HIXL or introduce
another transport scheduler. This does not add
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
  tests/ut/distributed/mooncake/test_wan_pd_reservation.py \
  tests/ut/distributed/mooncake/test_wan_pd_publication.py \
  tests/ut/distributed/mooncake/test_wan_pd_layer_events.py \
  tests/ut/distributed/mooncake/test_wan_pd_receiver_proxy.py \
  tests/ut/distributed/mooncake/test_wan_pd_receiver_scheduler.py \
  tests/ut/distributed/mooncake/test_wan_pd_input_identity.py \
  tests/ut/distributed/ascend_store/test_cache_namespace.py \
  tests/ut/distributed/ascend_store/test_layerwise_keys.py
```

The optional-component checks cover reservation replay/capacity, early layer
READs before the final seal, no early decode, source-pin ownership, event
ordering and the actual example HTTP router with controlled backends. They
provide correctness evidence, not NPU/RoCE performance measurements.
The connector's `get_wan_pd_stats()` provides local source,
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
