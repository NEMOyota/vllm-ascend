# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""FIFO PP publication with reserved buffers and consumer-owned receive credit.

The owned-source/FIFO/fail-closed design follows native vLLM's
distributed.pp_transport. This adapter adds NPU events, persistent pools, and
pregranted receiver credit, without depending on the native WAN fork's APIs.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from threading import Condition, Lock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

import msgpack
import torch

from .backend import TorchPPBackend
from .buffers import DeviceRuntime, FrameLayout, TensorBufferPool
from .config import WanPPConfig
from .credits import CreditBudget, CreditReservation

_PROTOCOL_VERSION = 1
_SHARED_BUDGET_POLL_S = 0.01


class PipelineTransportError(RuntimeError):
    """The channel must be torn down; an uncertain transfer is not retried."""


@dataclass
class _IncomingFrame:
    sequence: int
    slot: int
    layout: FrameLayout
    batch_id: int | str | None
    released: bool = False


class SendReservation:
    """Unpublished capacity may be cancelled; published work must be drained."""

    def __init__(
        self,
        transport: "WanPPTransport",
        slot: int,
        local: CreditReservation,
        remote: CreditReservation,
        shared: CreditReservation | None,
    ):
        self._transport = transport
        self.slot = slot
        self.local = local
        self.remote = remote
        self.shared = shared
        self.state = "reserved"
        self.sequence: int | None = None
        self.layout: FrameLayout | None = None
        self.header: dict[str, Any] | None = None
        self.ready: Any = None
        self.handles: list[Any] = []
        self.source_done = False
        self.consumer_done = False

    def publish(self, tensors: dict[str, Any], *, batch_id: int | str | None = None) -> None:
        self._transport._publish(self, tensors, batch_id)

    def cancel(self) -> None:
        self._transport._cancel(self)


class ReceiveTicket:
    """Lazy receive; wait makes data readable, release returns consumption credit."""

    def __init__(self, transport: "WanPPTransport", expected_batch_id: int | str | None = None):
        self._transport = transport
        self._expected_batch_id = expected_batch_id
        self._frame: _IncomingFrame | None = None
        self._result: dict[str, Any] | None = None
        self._released = False

    def wait(self) -> dict[str, Any]:
        transport = self._transport
        transport.check_error()
        if self._released:
            raise RuntimeError("Receive ticket has been released")
        if self._result is not None:
            return self._result
        self._frame = transport._take_frame(self)
        try:
            # TP collectives belong to the model execution thread, in its
            # existing order. Gather writes into the pool's persistent output.
            with transport._runtime.context():
                self._result = transport._recv_pool.materialize(
                    self._frame.slot, self._frame.layout, transport._all_gather_group
                )
        except Exception as exc:
            transport.fail(exc)
            transport.check_error()
        return self._result

    @property
    def batch_id(self) -> int | str | None:
        return None if self._frame is None else self._frame.batch_id

    def ready(self) -> bool:
        """Nonblocking visibility check for future data-ready schedulers."""
        with self._transport._condition:
            self._transport.check_error()
            return not self._released and (self._frame is not None or bool(self._transport._incoming))


class WanPPTransport:
    """One pipeline rank's fixed, directed receive/send channels.

    Admission reserves a source slot and *pregranted* destination capacity.
    Publication is a bounded, owned outbox. Data/control submission has one
    owner per direction, preserving FIFO; no per-frame reservation round trip
    is needed. Source retirement and receiver consumption are independent.

    ``backend`` is an adapter seam for HIXL/HCOMM. It must preserve the same
    completion/ordering contract; replacing it does not add PD state semantics.
    ``budget`` can coordinate local PP/PD outstanding work. Cross-process NIC
    arbitration and D-side persistent KV reservations remain separate layers.
    """

    def __init__(
        self,
        group,
        config: WanPPConfig,
        *,
        all_gather_group=None,
        on_fatal: Callable[[BaseException], None] | None = None,
        backend=None,
        device_runtime=None,
        budget: CreditBudget | None = None,
    ):
        if group.world_size <= 1:
            raise ValueError("WAN PP transport requires pipeline_parallel_size > 1")
        if getattr(group, "use_cpu_custom_send_recv", False):
            raise ValueError("WAN PP requires asynchronous tensor P2P")
        self.config = config
        self._group = group
        self._all_gather_group = all_gather_group
        self._gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        self._gather_rank = 0 if all_gather_group is None else all_gather_group.rank_in_group
        self._device = (
            torch.device("npu", group.device) if isinstance(group.device, int) else torch.device(group.device)
        )
        self._runtime = device_runtime or DeviceRuntime(self._device)
        self._backend = backend or TorchPPBackend(group, config)
        self._on_fatal = on_fatal
        self._shared_budget = budget
        if budget is not None and budget.max_bytes < config.slot_bytes:
            raise ValueError("Shared budget must accommodate one maximum-sized frame")
        rank = group.rank_in_group
        self._previous = None if rank == 0 else group.ranks[rank - 1]
        self._next = None if rank == group.world_size - 1 else group.ranks[rank + 1]
        self._condition = Condition()
        self._publication_lock = Lock()
        self._error: BaseException | None = None
        self._closing = False
        self._closed = False
        self._receive_epoch = uuid4().hex
        self._send_epoch: str | None = None
        self._remote_slot_bytes = 0
        self._remote_metadata_bytes = config.max_metadata_bytes
        self._local_budget = CreditBudget(config.num_slots, config.max_inflight_bytes)
        self._remote_budget: CreditBudget | None = None
        self._reservations: dict[int, SendReservation] = {}
        self._outbound: dict[int, SendReservation] = {}
        self._incoming: deque[_IncomingFrame] = deque()
        self._receive_leases: dict[int, _IncomingFrame] = {}
        self._active_ticket: ReceiveTicket | None = None
        self._next_sequence = 0
        self._received_sequence = 0
        self._acked_sequence = -1
        self._receive_bytes = 0
        self._published_frames = 0
        self._sent_frames = 0
        self._received_frames = 0
        self._consumed_frames = 0
        self._sent_wire_bytes = 0
        self._credit_wait_seconds = 0.0
        self._input_closed = self._previous is None
        self._output_closed = self._next is None
        self._send_queue: Queue[SendReservation | None] = Queue(config.num_slots + 1)
        self._completion_queue: Queue[SendReservation | None] = Queue(config.num_slots + 1)
        self._consume_queue: Queue[tuple[_IncomingFrame, Any] | None] = Queue(config.num_slots + 1)
        self._send_stream = self._runtime.new_stream()
        self._receive_stream = self._runtime.new_stream()
        self._completion_stream = self._runtime.new_stream()
        self._send_pool = (
            TensorBufferPool(config.num_slots, config.slot_bytes, self._device) if self._next is not None else None
        )
        self._recv_pool = (
            TensorBufferPool(config.num_slots, config.slot_bytes, self._device, materialize=self._gather_size > 1)
            if self._previous is not None
            else None
        )
        self._threads: list[Thread] = []
        # Credit receive must progress independently of waiting for input from
        # the previous rank; combining those owners can deadlock a pipeline.
        if self._next is not None:
            self._start(self._credit_loop, "wan-pp-credit")
            self._start(self._send_loop, "wan-pp-send")
            self._start(self._completion_loop, "wan-pp-retire")
        if self._previous is not None:
            self._start(self._consume_loop, "wan-pp-consume")
            self._start(self._receive_loop, "wan-pp-recv")

    @property
    def reserved_memory_bytes(self) -> int:
        return sum(pool.reserved_memory_bytes for pool in (self._send_pool, self._recv_pool) if pool is not None)

    def _start(self, target, name: str) -> None:
        thread = Thread(target=self._run_owner, args=(target,), name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _run_owner(self, target) -> None:
        try:
            with torch.inference_mode(), self._runtime.context():
                target()
        except Exception as exc:
            self.fail(exc)

    def check_error(self) -> None:
        with self._condition:
            if self._error is not None:
                raise PipelineTransportError("WAN PP channel failed; outstanding buffers remain owned") from self._error

    def fail(self, error: BaseException) -> None:
        with self._condition:
            first = self._error is None
            if first:
                self._error = error
            self._condition.notify_all()
        # No timeout or logical cancellation proves that a posted DMA stopped.
        # The worker retains this transport/pools until backend teardown.
        if first and self._on_fatal is not None:
            self._on_fatal(error)

    def _wait(self, predicate, description: str) -> None:
        """Wait with the condition held. Progress owners notify on every change."""
        deadline = monotonic() + self.config.timeout_s
        while not predicate():
            self.check_error()
            remaining = deadline - monotonic()
            if remaining <= 0:
                self.fail(TimeoutError(f"Timed out waiting for {description}"))
                self.check_error()
            self._condition.wait(remaining)
        self.check_error()

    def try_reserve(self, expected_bytes: int | None = None) -> SendReservation | None:
        """Side-effect-free capacity rejection before any publication or DMA."""
        size = self.config.slot_bytes if expected_bytes is None else expected_bytes
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= self.config.slot_bytes:
            raise ValueError("Frame reservation must fit slot_bytes")
        with self._condition:
            self.check_error()
            if self._closing or self._closed:
                raise RuntimeError("WAN PP transport is closing")
            if self._next is None:
                raise RuntimeError("Last PP rank has no outgoing pipeline channel")
            if self._remote_budget is None:
                return None
            if size > self._remote_slot_bytes:
                raise ValueError("Frame reservation exceeds the receiver's slot size")
            local = self._local_budget.try_reserve(size)
            if local is None:
                return None
            remote = self._remote_budget.try_reserve(size)
            if remote is None:
                local.release()
                return None
            shared = None if self._shared_budget is None else self._shared_budget.try_reserve(size)
            if self._shared_budget is not None and shared is None:
                local.release()
                remote.release()
                return None
            slot = self._send_pool.acquire()
            reservation = SendReservation(self, slot, local, remote, shared)
            self._reservations[id(reservation)] = reservation
            return reservation

    def reserve(self, expected_bytes: int | None = None) -> SendReservation:
        """Reserve before model execution, or use try_reserve in a future scheduler."""
        start = monotonic()
        deadline = start + self.config.timeout_s
        with self._condition:
            while (reservation := self.try_reserve(expected_bytes)) is None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self.fail(TimeoutError("Timed out waiting for PP capacity"))
                    self.check_error()
                # A shared local budget may be released by another transport.
                self._condition.wait(
                    min(remaining, _SHARED_BUDGET_POLL_S) if self._shared_budget is not None else remaining
                )
            self._credit_wait_seconds += monotonic() - start
            return reservation

    def _cancel(self, reservation: SendReservation) -> None:
        with self._condition:
            if reservation.state == "cancelled":
                return
            if reservation.state != "reserved" or reservation._transport is not self:
                raise RuntimeError("Only an unpublished reservation may be cancelled")
            # A reserved lease has never been used for copy or communication.
            # It may be settled after a worker compute failure; failed channels
            # still reject new reservations and never reuse published leases.
            reservation.local.release()
            reservation.remote.release()
            if reservation.shared is not None:
                reservation.shared.release()
            self._send_pool.release(reservation.slot)
            reservation.state = "cancelled"
            del self._reservations[id(reservation)]
            self._condition.notify_all()

    def send(self, tensors: dict[str, Any], *, batch_id: int | str | None = None) -> None:
        layout = FrameLayout.from_tensors(tensors, self._gather_size, self._gather_rank)
        reservation = self.reserve(layout.logical_bytes)
        try:
            reservation.publish(tensors, batch_id=batch_id)
        except Exception:
            if reservation.state == "reserved":
                reservation.cancel()
            raise

    def _publish(self, reservation: SendReservation, tensors: dict[str, Any], batch_id) -> None:
        # Packing runs outside the condition so completion owners can progress.
        # Serialize publishers across that interval to keep header/outbox
        # sequences identical, including for future scheduler-driven callers.
        with self._publication_lock:
            self._prepare_publication(reservation, tensors, batch_id)

    def _prepare_publication(self, reservation: SendReservation, tensors: dict[str, Any], batch_id) -> None:
        layout = FrameLayout.from_tensors(tensors, self._gather_size, self._gather_rank)
        if layout.logical_bytes > reservation.local.size:
            raise ValueError("Frame exceeds its reservation; increase slot_bytes or reduce the scheduled batch")
        if batch_id is not None and (isinstance(batch_id, bool) or not isinstance(batch_id, (str, int))):
            raise ValueError("batch_id must be an integer, string or None")
        with self._condition:
            self.check_error()
            if self._closing or reservation.state != "reserved" or reservation._transport is not self:
                raise RuntimeError("Reservation is not publishable")
            header = {
                "kind": "frame",
                "version": _PROTOCOL_VERSION,
                "epoch": self._send_epoch,
                "sequence": self._next_sequence,
                "batch_id": batch_id,
                "gather_size": self._gather_size,
                "layout": layout.to_dict(),
            }
            # Own the metadata as well as tensors before returning to a caller
            # that can mutate its output. Validate before copying or publishing.
            encoded = msgpack.packb(header, use_bin_type=True)
            if len(encoded) > min(self.config.max_metadata_bytes, self._remote_metadata_bytes):
                raise ValueError("PP frame metadata exceeds max_metadata_bytes")
            header = msgpack.unpackb(encoded, raw=False)
            reservation.state = "preparing"
        try:
            with torch.inference_mode(), self._runtime.context():
                self._send_pool.pack(reservation.slot, tensors, layout, self._gather_rank)
                reservation.ready = self._runtime.record_event()
            with self._condition:
                self.check_error()
                reservation.layout = layout
                reservation.header = header
                reservation.sequence = self._next_sequence
                reservation.local.shrink(layout.logical_bytes)
                reservation.remote.shrink(layout.logical_bytes)
                if reservation.shared is not None:
                    reservation.shared.shrink(layout.logical_bytes)
                self._outbound[self._next_sequence] = reservation
                self._next_sequence += 1
                reservation.state = "queued"
                # Every outbox entry owns a source slot, so this cannot fill
                # beyond its fixed capacity. Never drop a scheduled frame.
                self._send_queue.put_nowait(reservation)
                self._condition.notify_all()
        except Exception as exc:
            self.fail(exc)
            self.check_error()

    def _send_loop(self) -> None:
        while (frame := self._send_queue.get()) is not None:
            self.check_error()
            # A metadata error may occur after part of the message was sent.
            # It is fatal, not a retryable WOULD_BLOCK.
            self._backend.publish(frame.header, self._next)
            with self._runtime.context(self._send_stream):
                self._runtime.wait_event(frame.ready, self._send_stream)
                frame.handles = self._backend.submit_send(
                    self._send_pool.buffer(frame.slot, frame.layout.wire_bytes), self._next
                )
            with self._condition:
                frame.state = "published"
                self._published_frames += 1
                self._sent_wire_bytes += frame.layout.wire_bytes
                self._condition.notify_all()
            self._completion_queue.put_nowait(frame)
        self._backend.publish(
            {"kind": "close", "version": _PROTOCOL_VERSION, "epoch": self._send_epoch, "sequence": self._next_sequence},
            self._next,
        )
        self._completion_queue.put_nowait(None)

    def _completion_loop(self) -> None:
        while (frame := self._completion_queue.get()) is not None:
            self.check_error()
            self._runtime.finish(frame.handles, self._completion_stream)
            with self._condition:
                self.check_error()
                frame.source_done = True
                frame.local.release()
                if frame.shared is not None:
                    frame.shared.release()
                self._send_pool.release(frame.slot)
                self._sent_frames += 1
                self._retire_outbound(frame)
                self._condition.notify_all()

    def _retire_outbound(self, frame: SendReservation) -> None:
        if frame.source_done and frame.consumer_done:
            frame.state = "retired"
            del self._outbound[frame.sequence]
            del self._reservations[id(frame)]

    def _credit_loop(self) -> None:
        hello = self._backend.recv_ack(self._next)
        if (
            hello.get("kind") != "hello"
            or hello.get("version") != _PROTOCOL_VERSION
            or hello.get("gather_size") != self._gather_size
            or hello.get("device_type") != self._device.type
            or not isinstance(hello.get("epoch"), str)
        ):
            raise PipelineTransportError("Incompatible PP receiver capabilities")
        slots, size, limit = hello["num_slots"], hello["slot_bytes"], hello["max_inflight_bytes"]
        metadata_limit = hello["max_metadata_bytes"]
        if (
            not all(type(value) is int and value > 0 for value in (slots, size, limit, metadata_limit))
            or not size <= limit <= slots * size
        ):
            raise PipelineTransportError("Invalid receiver credit grant")
        with self._condition:
            self._send_epoch = hello["epoch"]
            self._remote_slot_bytes = size
            self._remote_metadata_bytes = metadata_limit
            self._remote_budget = CreditBudget(slots, limit)
            self._condition.notify_all()
        while True:
            ack = self._backend.recv_ack(self._next)
            if ack.get("epoch") != self._send_epoch:
                raise PipelineTransportError("Stale PP credit generation")
            with self._condition:
                self.check_error()
                if ack.get("kind") == "closed":
                    if not self._closing or self._outbound:
                        raise PipelineTransportError("Peer closed before consuming all published frames")
                    self._output_closed = True
                    self._condition.notify_all()
                    return
                if ack.get("kind") != "consumed" or type(ack.get("sequence")) is not int:
                    raise PipelineTransportError("Invalid PP consumption acknowledgement")
                sequence = ack["sequence"]
                if sequence <= self._acked_sequence:
                    # Duplicate notifications never grant credit twice.
                    continue
                if sequence != self._acked_sequence + 1 or sequence not in self._outbound:
                    raise PipelineTransportError("Out-of-order PP consumption acknowledgement")
                frame = self._outbound[sequence]
                if ack.get("logical_bytes") != frame.layout.logical_bytes:
                    raise PipelineTransportError("PP consumption size mismatch")
                frame.remote.release()
                frame.consumer_done = True
                self._acked_sequence = sequence
                self._retire_outbound(frame)
                self._condition.notify_all()

    def _receive_loop(self) -> None:
        self._backend.send_ack(
            {
                "kind": "hello",
                "version": _PROTOCOL_VERSION,
                "epoch": self._receive_epoch,
                "num_slots": self.config.num_slots,
                "slot_bytes": self.config.slot_bytes,
                "max_inflight_bytes": self.config.max_inflight_bytes,
                "max_metadata_bytes": self.config.max_metadata_bytes,
                "gather_size": self._gather_size,
                "device_type": self._device.type,
            },
            self._previous,
        )
        while True:
            header = self._backend.receive_header(self._previous)
            if (
                header.get("version") != _PROTOCOL_VERSION
                or header.get("epoch") != self._receive_epoch
                or type(header.get("sequence")) is not int
                or header["sequence"] != self._received_sequence
            ):
                raise PipelineTransportError("PP publication generation or sequence mismatch")
            if header.get("kind") == "close":
                with self._condition:
                    self._wait(lambda: not self._receive_leases, "remaining PP consumers")
                self._backend.send_ack({"kind": "closed", "epoch": self._receive_epoch}, self._previous)
                with self._condition:
                    self._input_closed = True
                    self._condition.notify_all()
                self._consume_queue.put_nowait(None)
                return
            if header.get("kind") != "frame" or header.get("gather_size") != self._gather_size:
                raise PipelineTransportError("Invalid PP frame or TP layout")
            layout = FrameLayout.from_dict(header["layout"])
            if (
                layout.logical_bytes > self.config.slot_bytes
                or layout.wire_bytes > layout.logical_bytes
                or layout.gather_size != self._gather_size
                or layout.device_kind not in (self._device.type, "none")
            ):
                raise PipelineTransportError("PP frame exceeds reserved receiver capacity")
            with self._condition:
                self.check_error()
                if (
                    len(self._receive_leases) >= self.config.num_slots
                    or self._receive_bytes + layout.logical_bytes > self.config.max_inflight_bytes
                ):
                    raise PipelineTransportError("Sender exceeded granted receive credit")
                slot = self._recv_pool.acquire()
                frame = _IncomingFrame(self._received_sequence, slot, layout, header.get("batch_id"))
                self._receive_leases[frame.sequence] = frame
                self._receive_bytes += layout.logical_bytes
                self._received_sequence += 1
            with self._runtime.context(self._receive_stream):
                handles = self._backend.submit_recv(self._recv_pool.buffer(slot, layout.wire_bytes), self._previous)
                self._runtime.finish(handles, self._receive_stream)
            with self._condition:
                self.check_error()
                self._received_frames += 1
                if self._closing:
                    self._queue_consumed(frame, None)
                else:
                    self._incoming.append(frame)
                self._condition.notify_all()

    def receive(self, *, expected_batch_id: int | str | None = None) -> ReceiveTicket:
        with self._condition:
            self.check_error()
            if self._previous is None:
                raise RuntimeError("First PP rank has no incoming pipeline channel")
            if self._closing or self._input_closed:
                raise RuntimeError("Incoming PP channel is closed")
            if self._active_ticket is not None:
                raise RuntimeError("Release the previous PP receive before requesting another")
            self._active_ticket = ReceiveTicket(self, expected_batch_id)
            return self._active_ticket

    def _take_frame(self, ticket: ReceiveTicket) -> _IncomingFrame:
        with self._condition:
            if ticket is not self._active_ticket:
                raise RuntimeError("Receive ticket is not active")
            self._wait(lambda: bool(self._incoming) or self._input_closed or self._closing, "PP frame visibility")
            if not self._incoming:
                raise PipelineTransportError("PP channel closed before the expected frame arrived")
            if ticket._expected_batch_id is not None and self._incoming[0].batch_id != ticket._expected_batch_id:
                self.fail(PipelineTransportError("PP frame does not match the scheduled batch"))
                self.check_error()
            return self._incoming.popleft()

    def release(self, ticket: ReceiveTicket) -> None:
        ticket.wait()
        with self._runtime.context():
            consumed = self._runtime.record_event()
        with self._condition:
            self.check_error()
            if ticket is not self._active_ticket or ticket._released:
                raise RuntimeError("Receive ticket is not active")
            ticket._released = True
            self._active_ticket = None
            self._queue_consumed(ticket._frame, consumed)

    def _queue_consumed(self, frame: _IncomingFrame, event) -> None:
        if frame.released:
            raise RuntimeError("Receiver lease released twice")
        frame.released = True
        self._consume_queue.put_nowait((frame, event))
        self._condition.notify_all()

    def _consume_loop(self) -> None:
        while (item := self._consume_queue.get()) is not None:
            frame, event = item
            if event is not None:
                event.synchronize()
            self.check_error()
            # ACK publication is serialized by this single consumption owner.
            # A receiver slot is no longer read at this point, but its credit is
            # not reusable by the sender until that sender receives the ACK.
            with self._condition:
                self._recv_pool.release(frame.slot)
                del self._receive_leases[frame.sequence]
                self._receive_bytes -= frame.layout.logical_bytes
                self._consumed_frames += 1
                self._condition.notify_all()
            self._backend.send_ack(
                {
                    "kind": "consumed",
                    "epoch": self._receive_epoch,
                    "sequence": frame.sequence,
                    "logical_bytes": frame.layout.logical_bytes,
                },
                self._previous,
            )

    def drain(self) -> None:
        """Wait for accepted sends and already scheduled consumers, not absent input."""
        with self._condition:
            self._wait(
                lambda: not self._reservations and not self._receive_leases,
                "published PP transfers and consumers",
            )

    def close(self) -> None:
        """Cooperative EOF; discard unread received frames only during shutdown.

        On failure, pools remain owned. Restarting a process or releasing a
        logical handle is not itself proof that remote writes have stopped.
        """
        with self._condition:
            if self._closed:
                self.check_error()
                return
            self.check_error()
            self._closing = True
            for reservation in list(self._reservations.values()):
                if reservation.state == "reserved":
                    self._cancel(reservation)
            if self._active_ticket is not None:
                ticket = self._active_ticket
                if ticket._frame is not None and not ticket._released:
                    with self._runtime.context():
                        self._queue_consumed(ticket._frame, self._runtime.record_event())
                ticket._released = True
                self._active_ticket = None
            while self._incoming:
                self._queue_consumed(self._incoming.popleft(), None)
            self._condition.notify_all()
        self.drain()
        if self._next is not None:
            with self._condition:
                self._wait(lambda: self._remote_budget is not None, "receiver capability handshake")
            self._send_queue.put_nowait(None)
        with self._condition:
            self._wait(lambda: self._input_closed and self._output_closed, "PP channel shutdown")
        deadline = monotonic() + self.config.timeout_s
        for thread in self._threads:
            thread.join(max(0, deadline - monotonic()))
            if thread.is_alive():
                self.fail(TimeoutError("PP progress owner did not shut down"))
                self.check_error()
        with self._condition:
            self._closed = True

    def stats(self) -> dict[str, int | float | bool]:
        with self._condition:
            local_count, local_bytes = self._local_budget.usage()
            remote_count, remote_bytes = (0, 0) if self._remote_budget is None else self._remote_budget.usage()
            return {
                "reserved_send_frames": local_count,
                "reserved_send_bytes": local_bytes,
                "remote_inflight_frames": remote_count,
                "remote_inflight_bytes": remote_bytes,
                "receive_frames": len(self._receive_leases),
                "receive_bytes": self._receive_bytes,
                "published_frames": self._published_frames,
                "sent_frames": self._sent_frames,
                "received_frames": self._received_frames,
                "consumed_frames": self._consumed_frames,
                "sent_wire_bytes": self._sent_wire_bytes,
                "credit_wait_seconds": self._credit_wait_seconds,
                "pool_bytes": self.reserved_memory_bytes,
                "failed": self._error is not None,
            }
