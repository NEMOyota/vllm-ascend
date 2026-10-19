# WAN PP joint scheduling

Joint scheduling combines a compute-token cap with network fragments, bounded
submission windows, and an optional shared egress service. It extends the
[bounded PP transport](wan_pipeline_transport.md) without splitting an already
scheduled batch inside a worker. Model Runner V1/V2 keep their existing forward
and KV ownership semantics.

## Three independent budgets

| Budget | Admission | Release |
| --- | --- | --- |
| Compute chunk | Central scheduler's `max_num_scheduled_tokens` before KV allocation | Existing scheduler execution/progress rules |
| Whole-frame staging | Source slot plus pregranted receiver capacity before forward | Source after its last fragment completes; receiver after model consumption |
| Network fragments | Local byte/operation window and shared egress grant before each submission | That fragment's backend completion and device event |

An activation frame can exceed the network window: completed fragments return
network capacity while the frame's source and receiver slots remain owned.
The receiver exposes tensors only after **all** fragments are visible. There
is no per-fragment model-consumption ACK, and fragments never reorder batches.
Empty frames use metadata and frame-credit handling without an egress grant.

## Configure compute chunks and the local window

Append the following object to the existing Ascend `--additional-config` JSON:

```json
{
  "wan_pp_config": {
    "enabled": true,
    "num_slots": 2,
    "slot_bytes": 67108864,
    "max_inflight_bytes": 134217728,
    "joint_config": {
      "enabled": true,
      "compute_chunk_tokens": 1024,
      "fragment_bytes": 1048576,
      "window_fragments": 8,
      "window_bytes": 8388608
    }
  }
}
```

These are illustrative starting values, not hardware-tuned defaults. Both WAN
transport and joint scheduling are disabled by default. All PP ranks must run
the same protocol version; the current transport version is 2.

The platform hook lowers `max_num_scheduled_tokens` before scheduler creation.
It retains a stricter existing cap and keeps `max_num_batched_tokens` unchanged
for runner buffers and graph capacity. The central scheduler broadcasts one
result to all PP ranks. Existing sync/async scheduling, request ordering, KV
allocation, block alignment and speculative state updates continue to apply.
The cap limits total scheduled tokens, not a separate prefill-only allowance;
it does not establish decode-first scheduling or increase executor depth.

Optional static estimates refine the cap:

| `joint_config` field | Meaning |
| --- | --- |
| `estimated_boundary_bytes_per_token` | Full logical tensor bytes/token for the limiting PP boundary; zero disables this estimate |
| `boundary_fixed_bytes` | Reserved estimate for fixed tensors, alignment and padding |
| `estimated_wire_bytes_per_token` | Per-lane wire bytes/token after slicing; zero uses the logical estimate |
| `estimated_link_bytes_per_s` | Expected available payload bandwidth for that lane, not a new rate limiter |
| `target_chunk_ms` | Estimated serialization-time target, default 10 ms |
| `estimated_rtt_ms` | Produces a bandwidth-delay-product reference in the startup plan |

With `C0` the existing scheduler limit and `C` the explicit chunk limit:

```text
compute cap = min(C0, C,
                  floor((slot_bytes - fixed_bytes) / logical_bytes_per_token),
                  floor((estimated_link_bytes_per_s * target_chunk_ms / 1000
                         - fixed_bytes) / wire_bytes_per_token))
```

Terms with disabled estimates are omitted. The bandwidth estimate must reflect
the flow's share of the exit, not the entire physical port rate for every TP
rank. RTT supplies a sizing reference; the implementation does not silently
expand the configured window to a BDP or measure congestion automatically.

Graph padding, sequence parallelism and auxiliary tensors can exceed these
estimates. Full-slot reservation and actual `FrameLayout` capacity checks
remain authoritative. No generic tensor-axis trimming is performed. Models
whose multimodal input cannot be chunked must retain a sufficient compute
budget: automatic reduction is rejected when `disable_chunked_mm_input` is
set, because an indivisible input could otherwise wait forever. Unknown custom
schedulers require explicit integration; whole-prompt and speculative-step
constraints are checked before installing the cap.

`fragment_bytes` is the requested maximum data piece. The negotiated size also
fits the local slot, peer capability and shared egress byte/burst limits.
`window_fragments` and `window_bytes` bound outstanding local network work.
The backend sends byte views of the existing packed slot, without allocating
another full buffer per fragment. TP gathers still run on the execution thread.

## Share one physical exit across processes and hosts

Start one budget service per physical exit on its **sending-side** control
network. For example, on that service host:

```bash
python -m vllm_ascend.distributed.pp_transport.egress_service \
  --host 0.0.0.0 --port 29590 \
  --max-bytes 16777216 --max-fragments 16 \
  --rate-bytes-per-s 1000000000 --burst-bytes 8388608 \
  --latency-burst-bytes 4194304 --max-flows 128
```

The server defaults to loopback unless an explicit host is supplied. It is a
control service for a trusted private network and does not provide application
authentication or TLS. It is started explicitly, not by racing worker imports.
It forwards no tensor payload. All participating PP/TP or future PD senders
share the same arbiter, even when they are separate processes on different hosts.

Set `joint_config.egress_endpoint` to the service's reachable address, for
example `"192.0.2.10:29590"` (replace this documentation address). Leave it
empty for local windows without shared arbitration. `egress_rpc_timeout_s`
bounds each short control RPC and defaults to 5 seconds.

For distinct exits/directions, `egress_rank_endpoints` overrides the default
endpoint by the **global source rank in the PP group**. An empty override
excludes a source using only local fabric from egress accounting:

```json
{
  "egress_endpoint": "",
  "egress_rank_endpoints": {
    "0": "192.0.2.10:29590",
    "1": "192.0.2.20:29590"
  }
}
```

This fragment belongs inside `joint_config`. Map all TP lanes sharing an exit
to the same service. Reverse-direction traffic usually uses the other site's
exit service. The RPCs should traverse the local control network: putting the
arbiter across the WAN adds a grant RTT for every fragment. No individual
worker may create a separate service and claim the full shared port budget.

The arbiter provides:

- One pending request per flow, strict request IDs, generation checks and
  idempotent pending polls. Flow count, messages and issued grants are bounded.
- Aggregate in-flight bytes and fragments. A persistent token bucket optionally
  controls aggregate payload admission rate; completion does not refund rate
  tokens or reset the bucket between frames.
- Byte-based fairness within a traffic class. Latency-class preference is
  bounded by `latency_burst_bytes` plus one fragment before waiting bulk work
  gets a turn. Already submitted transfers are not preempted.
- Automatic small-frame classification as latency traffic; larger frames are
  bulk. This is a size heuristic, not a decode detector. The public
  `send(..., traffic="latency" | "bulk")` interface can override it.

The limits count submitted **payload bytes**, including each participating TP
sender's actual wire slice. They do not include packet headers, transport ACKs,
retries or unregistered applications sharing the NIC. Reserve headroom in the
configured rate. This is cooperative application admission, not hardware QoS.

## Completion, timeouts and failure

A grant is returned only after `DeviceRuntime.finish()` proves the fragment's
completion. Source-frame credit waits for the final fragment; remote frame
credit still waits for the consumer event. A partial submission failure retains
the affected buffers and grants and stops new admission. Neither a timeout nor
a lost connection proves that an old DMA stopped.

Unissued waiting work and explicitly unposted grants can be cancelled. Healthy
client close cancels unobserved pregrants before disconnecting. A disconnect
with uncertain live grants, or `--lease-timeout-s` expiry after a grant, fails
the shared generation without refunding capacity. Stop/quiesce the affected
backends before recreating a failed budget scope; transparent recovery is not
implemented. Normal idle sessions remain connected. The socket timeout applies
to handshake/message I/O, not the interval between inference requests.

Pacing waits do not start a grant lease timer. They can still delay already
posted receiver operations: frame metadata is published before all pieces have
egress permission. Configure framework and HCCL/Gloo operation timeouts for
the expected rate and queueing delay. This implementation does not guarantee
unlimited pacing waits against a finite backend receive timeout.

## Reuse and future communication-library/PD integration

The design follows the HCCL experimental
[edge-cloud adapter](https://github.com/NEMOyota/hccl-pro/blob/experiment/a2-edge-cloud-adapter-v2/experimental/eco_system/edge_cloud/adapter/README.md)
and its
[validation record](https://github.com/NEMOyota/hccl-pro/blob/experiment/a2-edge-cloud-adapter-v2/experimental/eco_system/edge_cloud/adapter/VALIDATION.md):
shared grants, bounded foreground service, per-flow order and unresolved grant
retention. The earlier `dc_multitree/README_JOINT.md` separates producer-ready,
submission and retirement, but its FP32 training reduction is not a PP operator.

The current PP data path remains torch Gloo/HCCL P2P. The HIXL/HCOMM Python
object-file adapters use HOST staging or synchronous host/device copies; they
are not imported into this hot path. No communication-library source is changed.
HCOMM's public DEVICE-span Submit/Poll/Retire primitives and HIXL's device
registration/async operations can later implement a backend owner with the
same completion guarantees. HCOMM's session thread/context affinity must be
preserved; replacing one method on the current multithreaded adapter is unsafe.

A PD/EC backend can use `EgressClient` on the same exit, requesting a lease just
before a fragment submission and releasing it on its proven completion. It
must independently reserve destination KV/state storage and apply its state
commit protocol. Network admission is not a KV handoff ACK or model readiness.

## Basic experiments and observability

`NPUWorker.get_wan_pp_stats()` adds network reservations, peak window occupancy,
submitted/completed fragment counts, effective fragment size and network wait
time. The service's `stats()` reports global in-flight/peak bytes and fragments,
grant totals, pending flows and failed-generation state.

The focused CPU experiments cover real sync/async scheduler steps and KV block
allocation, shared-budget fairness/rate/failure handling, real TCP clients,
idle/resume, and Gloo PP transfers. A 672-byte frame crosses a 256-byte/two-piece
window with a final 32-byte fragment. Two PP groups share one TCP service and
transfer four frames, totaling 2688 bytes and 24 grants, without exceeding the
aggregate 256-byte/two-piece limit; final occupancy is zero.

```bash
python -m pytest --noconftest -q \
  tests/ut/distributed/test_wan_pp_planner.py \
  tests/ut/distributed/test_wan_pp_egress.py \
  tests/ut/distributed/test_wan_pp_egress_service.py \
  tests/ut/distributed/test_wan_pp_joint_transport.py
```

These establish control/lifetime correctness on CPU. They do not establish
NPU/HCCL behavior or WAN throughput gains. Hardware validation should compare
baseline, compute cap, fragmentation/window, and shared-egress configurations
on the same workload, measuring TTFT/token-gap tails, useful throughput,
communication waits and idle time. Dynamic congestion-feedback chunk tuning,
model-specific layout adaptation and PD state transfer remain separate work.
