# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Finite-window READ progress, independent of model and NPU imports."""

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any

from .config import WanPDConfig


@dataclass(frozen=True)
class TransferRange:
    peer: str
    source: int
    destination: int
    nbytes: int
    layer_key: str = ""


@dataclass(frozen=True)
class ChunkPiece:
    source: int
    destination: int
    nbytes: int
    offset: int


@dataclass(frozen=True)
class TransferChunk:
    peer: str
    pieces: tuple[ChunkPiece, ...]
    nbytes: int
    layer_key: str = ""


def transfer_chunks(ranges: list[TransferRange], config: WanPDConfig) -> Iterator[TransferChunk]:
    """Split without copying payloads or building an unbounded chunk list."""
    if len(ranges) > config.max_manifest_ranges:
        raise ValueError("WAN PD transfer exceeds max_manifest_ranges")
    pieces: list[ChunkPiece] = []
    used = 0
    peer = ""
    layer_key = ""
    for item in ranges:
        if not item.peer or min(item.source, item.destination, item.nbytes) <= 0:
            raise ValueError("WAN PD transfer contains an invalid range")
        consumed = 0
        while consumed < item.nbytes:
            if pieces and (
                peer != item.peer
                or layer_key != item.layer_key
                or used == config.chunk_bytes
                or len(pieces) == config.max_descriptors_per_chunk
            ):
                yield TransferChunk(peer, tuple(pieces), used, layer_key)
                pieces, used = [], 0
            peer = item.peer
            layer_key = item.layer_key
            size = min(item.nbytes - consumed, config.chunk_bytes - used)
            pieces.append(ChunkPiece(item.source + consumed, item.destination + consumed, size, used))
            used += size
            consumed += size
    if pieces:
        yield TransferChunk(peer, tuple(pieces), used, layer_key)


@dataclass
class _Flight:
    slot: int
    chunk: TransferChunk
    ticket: Any
    lease: Any = None
    event: Any = None


@dataclass
class _Request:
    request_id: str
    planner: Callable[[], list[TransferRange]]
    deadline: float
    chunks: Iterator[TransferChunk] | None = None
    next_chunk: TransferChunk | None = None
    exhausted: bool = False
    flights: list[_Flight] = field(default_factory=list)
    error: str | None = None
    gate: Any = None


class BoundedReadProgress:
    """One progress owner polls each ticket at most once per advance call.

    enqueue/retire may run on the model thread. Planning, transfer publication,
    event queries and completion callbacks run on the progress owner. Failed
    DMA retains both the ticket and staging allocation; only staging addresses
    are ever submitted to Mooncake.
    """

    def __init__(
        self,
        config: WanPDConfig,
        engine: Any,
        pool: Any,
        completed: Callable[[str, str | None], None],
        *,
        egress: Any = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config, self.engine, self.pool = config, engine, pool
        self.completed, self.egress, self.clock = completed, egress, clock
        self._lock = Lock()
        self._pending: deque[_Request] = deque()
        self._owned: set[str] = set()
        self._active: deque[_Request] = deque()
        self._quarantined: list[_Flight] = []
        self._network_bytes = 0
        self._egress_waiter: str | None = None
        self.fatal_error: Exception | None = None
        engine.set_managed_batch_limit(config.staging_slots)

    def enqueue(self, request_id: str, planner: Callable[[], list[TransferRange]], *, gate: Any = None) -> bool:
        with self._lock:
            if request_id in self._owned:
                raise ValueError("WAN PD request is already pending or awaiting completion delivery")
            if len(self._owned) >= self.config.max_pending_requests:
                return False
            self._owned.add(request_id)
            self._pending.append(_Request(request_id, planner, self.clock() + self.config.timeout_s, gate=gate))
            return True

    def retire(self, request_id: str) -> None:
        with self._lock:
            self._owned.discard(request_id)

    def _fail(self, request: _Request, error: Any) -> None:
        if request.error is None:
            request.error = str(error)
        if self._egress_waiter == request.request_id:
            self._egress_waiter = None
            # A lost control connection cannot justify returning a grant.
            with suppress(Exception):
                self.egress.cancel_pending()
        for flight in request.flights[:]:
            if flight.event is None:
                self.pool.quarantine(flight.slot)
                self._quarantined.append(flight)
                request.flights.remove(flight)

    def _poll(self, request: _Request) -> None:
        for flight in request.flights[:]:
            if flight.event is not None:
                # Query failure means device work may still touch D KV. Let
                # the worker fail rather than report a reusable destination.
                if flight.event.query():
                    self.pool.release(flight.slot)
                    request.flights.remove(flight)
                continue
            try:
                status = flight.ticket.poll()
            except Exception as exc:
                self._fail(request, exc)
                return
            if status == "WAITING":
                continue
            if status != "COMPLETED":
                self._fail(request, getattr(flight.ticket, "error", status))
                return
            self._network_bytes -= flight.chunk.nbytes
            if flight.lease is not None:
                try:
                    flight.lease.release()
                except Exception as exc:
                    # Data is quiescent, but the shared arbiter is unavailable.
                    # Recycle safely and fail this request without more READs.
                    self.pool.release(flight.slot)
                    request.flights.remove(flight)
                    self._fail(request, exc)
                    return
            # copy() may have enqueued device work before raising. Such an
            # exception is fatal, never a request-local reusable-D failure.
            flight.event = self.pool.copy(flight.slot, flight.chunk.pieces)

    def _issue(self, request: _Request) -> None:
        if request.next_chunk is None and not request.exhausted:
            request.next_chunk = next(request.chunks, None)
            request.exhausted = request.next_chunk is None
        chunk = request.next_chunk
        if chunk is None or self._network_bytes + chunk.nbytes > self.config.max_inflight_bytes:
            return
        if request.gate is not None and not request.gate.permits(chunk.layer_key):
            return
        if self._egress_waiter not in (None, request.request_id):
            return
        slot = self.pool.acquire()
        if slot is None:
            return
        lease = None
        submit_attempted = False
        try:
            if self.egress is not None:
                lease = self.egress.try_acquire(chunk.nbytes, "bulk")
                if lease is None:
                    self._egress_waiter = request.request_id
                    self.pool.release(slot)
                    return
                self._egress_waiter = None
            # Native errors normally return an owned FAILED_UNDRAINED ticket.
            # The Python wrapper itself may still fail after publication, so
            # an exception at this boundary cannot prove quiescence.
            submit_attempted = True
            ticket = self.engine.batch_transfer_async_read_ticket(
                chunk.peer,
                [self.pool.address(slot, piece.offset) for piece in chunk.pieces],
                [piece.source for piece in chunk.pieces],
                [piece.nbytes for piece in chunk.pieces],
            )
        except Exception as exc:
            if submit_attempted:
                self._network_bytes += chunk.nbytes
                request.flights.append(_Flight(slot, chunk, None, lease))
            else:
                self.pool.release(slot)
            if lease is not None and not submit_attempted:
                with suppress(Exception):
                    lease.safe_cancel_before_submit()
            self._fail(request, exc)
            return
        self._network_bytes += chunk.nbytes
        request.flights.append(_Flight(slot, chunk, ticket, lease))
        request.next_chunk = None

    def advance(self) -> None:
        """Bounded polling and one fragment per active request per turn."""
        if self.fatal_error is not None:
            raise self.fatal_error
        try:
            # Poll existing copies before potentially slow metadata planning.
            for request in self._active:
                self._poll(request)
                if self.clock() >= request.deadline:
                    self._fail(request, "WAN PD continuation deadline exceeded")
            with self._lock:
                while self._pending and len(self._active) < self.config.max_active_requests:
                    self._active.append(self._pending.popleft())
            for _ in range(len(self._active)):
                request = self._active.popleft()
                if request.error is None and request.gate is not None:
                    try:
                        request.gate.advance()
                    except Exception as exc:
                        self._fail(request, exc)
                if request.error is None and request.chunks is None:
                    try:
                        if self.clock() >= request.deadline:
                            raise TimeoutError("WAN PD request expired before planning")
                        request.chunks = transfer_chunks(request.planner(), self.config)
                    except Exception as exc:
                        self._fail(request, exc)
                if request.error is None:
                    try:
                        self._issue(request)
                    except Exception as exc:
                        self._fail(request, exc)
                ready = request.exhausted and (request.gate is None or request.gate.sealed)
                if not request.flights and (request.error is not None or ready):
                    if request.gate is not None:
                        request.gate.close()
                    self.completed(request.request_id, request.error)
                else:
                    self._active.append(request)
        except Exception as exc:
            self.fatal_error = exc
            raise

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending_requests": len(self._pending),
                "active_requests": len(self._active),
                "retained_requests": len(self._owned),
                "quarantined_slots": len(self._quarantined),
                "network_inflight_bytes": self._network_bytes,
            }
