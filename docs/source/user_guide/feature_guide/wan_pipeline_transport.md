# Bounded transport for long-distance pipeline parallelism

The opt-in WAN PP runtime manages intermediate tensors between existing
pipeline stages. It adds reliable FIFO publication, capacity reservations,
persistent device buffers, and separate source/consumer credits. Model Runner
V1 and V2 use the same worker adapter; no model-specific forward changes are
required by the transport.

The default PP path is unchanged. This feature does not create a layer
partition or change the model's existing PP support requirements.

## Enable

Add this configuration to an otherwise working PP deployment, on every rank:

```json
{
  "wan_pp_config": {
    "enabled": true,
    "num_slots": 2,
    "slot_bytes": 67108864,
    "max_inflight_bytes": 134217728,
    "timeout_s": 300,
    "max_metadata_bytes": 65536
  }
}
```

For example, append it using `--additional-config '<json>'`. Keep any existing
Ascend configuration keys in the same JSON object. Enable it consistently on
all members of each PP group. `pipeline_parallel_size` must exceed one; the
external-launcher path is not supported by this adapter.

| Field | Meaning |
| --- | --- |
| `enabled` | Defaults to false. |
| `num_slots` | Maximum concurrent source and receive slots, separately, per rank. |
| `slot_bytes` | Maximum aligned full tensor layout for one frame, before TP slicing. |
| `max_inflight_bytes` | Limit on reserved/aligned logical bytes in each direction. Must be between one slot and the sum of all slots. |
| `timeout_s` | Maximum framework wait for capacity, visibility, drain or cooperative close. This is not a packet retry timeout. |
| `max_metadata_bytes` | Maximum serialized control message, separately bounded from device payload. |

Size a slot for the largest scheduled PP output, including residuals and any
auxiliary tensors. Alignment between tensors also consumes capacity. An
oversized output raises before publication; the worker fails the execution
instead of dropping the batch. Reduce the scheduled chunk/batch or increase
capacity before restarting. The worker conservatively reserves a full slot
before forward, then shrinks the byte reservation to the actual layout.

## Reservation, publication and consumption

At startup the receiver reserves its pool and grants slots and byte capacity.
The sender reserves both a local source slot and remote consumption credit
before producing another output. Within this grant, publication needs no
additional per-frame reservation round trip.

1. The execution stream copies output tensors into an owned source slot and
   records a producer event. This copy precedes later reuse of model outputs.
2. A bounded outbox owns the frame. A single publisher sends metadata and data
   in sequence, with protocol version, receiver generation and batch identity.
3. The communication stream waits for the producer event. A completion owner
   waits for the transfer and a device completion event before returning the
   **source** slot.
4. A separate receive owner fills a reserved input slot. `receive()` returns
   immediately; its ticket becomes ready after data is device-visible.
5. The execution thread performs any TP gather into a persistent output area,
   then the model copies/consumes those tensors using its existing input path.
6. After model execution and the outgoing snapshot, the worker records the
   consumer event. Only after this event completes does the receiver return
   **consumption** credit to the sender.

A received frame is not reusable merely because its send completed. A queued
frame is never discarded because an outbox is full: each entry already owns a
reservation. Unpublished reservations can be cancelled without consuming a
sequence number. After any potentially partial publication, failure is fatal
to that channel; there is no blind application retry of an uncertain DMA.

The implementation preserves PP order. It does not increase the executor's
scheduling depth or allow ranks to select unrelated batches independently.
The worker checks the received batch identity before consumption. Ordinary
PP and head/tail PP use the same mechanism when their stages already execute
through this worker path. A single request still has its cross-stage causal
dependency; extra slots cannot remove propagation latency.

## Buffer and communication costs

For `N = num_slots` and `B = slot_bytes`, persistent allocations are:

| Rank role | Device staging memory |
| --- | --- |
| First stage | `N * B` for owned sends |
| Last stage, no TP gather | `N * B` for received frames |
| Middle stage, no TP gather | `2 * N * B` |
| A stage receiving TP-sliced tensors | Add `N * B` for persistent full-size gather outputs |

For example, the default two 64 MiB slots use 256 MiB on a middle stage
without TP gather, or 384 MiB with gather. Idle slots remain allocated and
count toward this total. Automatic KV sizing includes this memory. With an
explicit `--kv-cache-memory` override, the user remains responsible for the
combined memory budget. Sleep drains outstanding transfers but does not
offload the transport pools.

Frames contain packed, contiguous bytes and preserve each tensor's dtype and
bits. Packing is an additional device copy, not zero-copy transport. The
torch/HCCL backend manages its own communication buffers and registration;
those backend allocations are separate from the pool counters. TP slicing is
used for contiguous divisible tensors when the worker's usual gather policy
is enabled. Non-contiguous or indivisible tensors travel in full. TP gathers
stay on the execution thread, using reserved destination storage. Sequence
parallelism retains the existing worker policy that disables this gather.

The first backend uses the existing PP HCCL data group, with bounded msgpack
control messages on its Gloo CPU group. Dedicated tags isolate those messages
from ordinary object/CPU traffic. One runtime owns each PP data channel;
do not submit an additional untagged PP data protocol on that same channel.
The lower-level transport retains responsibility for reliable packet delivery.

## PD, encoder splits and future backend integration

This runtime transports PP intermediate tensors. It applies to the PP segment
of a combined deployment regardless of whether P and D are at the edge or in
the cloud. Existing V1 `kv_connector_output` forwarding and idle connector
steps are preserved. Existing connector/topology restrictions still apply.

An independent E-to-P encoder connector and a P-to-D state connector are not
automatically redirected through the new transport. Their cache/state
semantics remain with those connectors. The extension points are explicit:

- `CreditBudget` and its reservations can be shared within a process with a
  future PD/EC adapter. Source completion and persistent D KV ownership must
  still use different reservations. This is not yet cross-process NIC QoS.
- `try_reserve()` is a nonblocking admission interface. `receive()` tickets
  expose `ready()` for future data-ready scheduling. The current worker uses
  blocking bounded admission and preserves its batch order.
- `TorchPPBackend` separates control publication and data submission. A
  HIXL/HCOMM adapter must provide the same source-safe/target-visible
  completion guarantees and extend capability negotiation as needed. A
  returned Python handle alone is not a completion proof.
- `FrameLayout` describes generic typed tensors and bounded scalar/list/dict
  metadata. It does not assume hidden/residual names, Qwen layer types, or KV
  formats. Model and PD state adaptation can be implemented separately.

## Failure and shutdown

Device buffers remain owned after a failed or uncertain transfer. A timeout,
an epoch change or a process restart does not itself prove that old remote
writes have stopped. Teardown must obey the communication backend's lifetime
contract before reusing affected addresses. This version does not implement
transparent reconnect or in-place retransmission.

Healthy shutdown drains accepted sends, discards unread inputs only during
shutdown, and exchanges EOF after all consumption acknowledgements. Sleep and
weight updates drain without discarding scheduled inputs. A channel error is
reported on worker execution/sampling or a waiting transport call.

## Observability and basic validation

`NPUWorker.get_wan_pp_stats()` is available through executor collective RPC.
It reports source and remote frame/byte reservations, receive occupancy,
published/sent/received/consumed counts, wire bytes, credit wait time, persistent
pool bytes and failure state. These are per-rank counters, not a shared-NIC
bandwidth estimator. Compare consumer-visible completion across backends.

The focused CPU tests use actual Gloo processes for FIFO transfers, source
reuse, consumption backpressure, cancellation and PP2/TP2 materialization:

```bash
python -m pytest --noconftest -q \
  tests/ut/distributed/test_wan_pp_transport.py \
  tests/ut/worker/test_wan_pp_worker.py
```

Use the repository's prepared development environment. These independent
tests do not need the broad NPU mock setup in the standard UT conftest. They
also exercise the shared V1/V2 worker adapter and a controlled completion
event. They do not establish NPU/HCCL correctness or WAN performance.

On available NPU hardware, first compare an otherwise identical working PP
deployment with the feature disabled/enabled, then test a graph configuration
supported by that model. Confirm matching outputs, bounded pool/credit
occupancy, graph-buffer reuse and correct shutdown. For a RoCE deployment,
measure TTFT/token-gap tails, useful throughput and NPU idle time alongside
actual network counters. Network impairment must affect the real RDMA path.
No performance improvement is claimed from the CPU experiments.
