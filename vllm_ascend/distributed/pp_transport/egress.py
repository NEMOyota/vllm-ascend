# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded, process-independent arbitration for a shared network egress.

The arbiter owns control-plane state only. A service may expose it to several
hosts; it neither transfers tensors nor assumes that a disconnected sender's
DMA has stopped. All state transitions are protected by one short-lived lock.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from time import monotonic
from uuid import uuid4


class EgressError(RuntimeError):
    """Invalid admission or a failed egress generation."""


@dataclass(frozen=True)
class EgressLimits:
    max_bytes: int
    max_fragments: int
    rate_bytes_per_s: float = 0
    burst_bytes: int = 0
    max_flows: int = 1024
    latency_burst_bytes: int = 0
    lease_timeout_s: float = 300

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_fragments", "max_flows"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("burst_bytes", "latency_burst_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isfinite(self.rate_bytes_per_s) or self.rate_bytes_per_s < 0:
            raise ValueError("rate_bytes_per_s must be finite and nonnegative")
        if not isfinite(self.lease_timeout_s) or self.lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be finite and positive")
        if self.burst_bytes == 0:
            object.__setattr__(self, "burst_bytes", self.max_bytes)
        if self.latency_burst_bytes == 0:
            object.__setattr__(self, "latency_burst_bytes", self.burst_bytes)


@dataclass
class _Request:
    session: str
    request_id: int
    nbytes: int
    traffic: str
    finish: int
    order: int
    granted_at: float | None = None

    @property
    def lease_id(self) -> str:
        return f"{self.session}:{self.request_id}"


@dataclass
class _Flow:
    flow_id: str
    next_request_id: int = 0
    pending: _Request | None = None
    finish: dict[str, int] = field(default_factory=lambda: {"latency": 0, "bulk": 0})


class EgressArbiter:
    """Byte-fair fragment admission with bounded latency-class preference.

    Each flow may enqueue one pending fragment and own several granted ones.
    Within a traffic class, virtual finish tags charge actual bytes; FIFO is
    the tie breaker and each flow's own request sequence is strict. When bulk
    waits, latency may receive at most ``latency_burst_bytes`` plus one fragment
    before a bulk grant. A selected fragment is never bypassed just because a
    smaller fragment fits: its wait is bounded by completion/rate progress,
    rather than repeatedly losing to small requests.

    Request IDs start at zero and increase by one per flow. They provide
    retry-safe admission without an unbounded completed-request history.
    """

    def __init__(self, limits: EgressLimits, *, clock: Callable[[], float] = monotonic):
        self.limits = limits
        self.generation = uuid4().hex
        self._clock = clock
        self._lock = Lock()
        self._flows: dict[str, _Flow] = {}
        self._flow_ids: dict[str, str] = {}
        self._leases: dict[str, _Request] = {}
        self._used_bytes = 0
        self._tokens = float(limits.burst_bytes)
        self._last_refill = clock()
        self._virtual_time = {"latency": 0, "bulk": 0}
        self._latency_bytes = 0
        self._order = 0
        self._granted_fragments = 0
        self._granted_bytes = 0
        self._peak_inflight_fragments = 0
        self._peak_inflight_bytes = 0
        self._failure: str | None = None

    def _expire(self, now: float) -> None:
        if self._failure is None:
            for request in self._leases.values():
                if now - request.granted_at >= self.limits.lease_timeout_s:
                    self._failure = f"Lease {request.lease_id} expired without confirmed completion"
                    break

    def _check(self, now: float) -> None:
        self._expire(now)
        if self._failure is not None:
            raise EgressError(f"Egress generation {self.generation} failed: {self._failure}")

    def _flow(self, session: str) -> _Flow:
        if not isinstance(session, str) or session not in self._flows:
            raise EgressError("Unknown egress session")
        return self._flows[session]

    @staticmethod
    def _validate_request_id(request_id: int) -> None:
        if type(request_id) is not int or not 0 <= request_id < 2**63:
            raise EgressError("request_id must be a nonnegative signed 64-bit integer")

    def register(self, flow_id: str) -> str:
        if not isinstance(flow_id, str) or not flow_id or len(flow_id) > 256:
            raise EgressError("flow_id must contain between 1 and 256 characters")
        with self._lock:
            self._check(self._clock())
            if flow_id in self._flow_ids:
                raise EgressError("flow_id already has a live session")
            if len(self._flows) >= self.limits.max_flows:
                raise EgressError("Egress flow capacity exhausted")
            session = uuid4().hex
            self._flows[session] = _Flow(flow_id)
            self._flow_ids[flow_id] = session
            return session

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self.limits.burst_bytes, self._tokens + elapsed * self.limits.rate_bytes_per_s)
        self._last_refill = now

    def _pump(self, now: float) -> None:
        self._refill(now)
        while len(self._leases) < self.limits.max_fragments:
            pending = [flow.pending for flow in self._flows.values() if flow.pending is not None]
            if not pending:
                return
            latency = [request for request in pending if request.traffic == "latency"]
            bulk = [request for request in pending if request.traffic == "bulk"]
            if latency and (not bulk or self._latency_bytes < self.limits.latency_burst_bytes):
                candidates = latency
            else:
                candidates = bulk
            request = min(candidates, key=lambda item: (item.finish, item.order))
            if self._used_bytes + request.nbytes > self.limits.max_bytes:
                return
            if self.limits.rate_bytes_per_s and self._tokens < request.nbytes:
                return
            flow = self._flows[request.session]
            flow.pending = None
            flow.finish[request.traffic] = request.finish
            self._virtual_time[request.traffic] = request.finish
            request.granted_at = now
            self._leases[request.lease_id] = request
            self._used_bytes += request.nbytes
            self._peak_inflight_fragments = max(self._peak_inflight_fragments, len(self._leases))
            self._peak_inflight_bytes = max(self._peak_inflight_bytes, self._used_bytes)
            if self.limits.rate_bytes_per_s:
                self._tokens -= request.nbytes
            if request.traffic == "latency":
                self._latency_bytes += request.nbytes
            else:
                self._latency_bytes = 0
            self._granted_fragments += 1
            self._granted_bytes += request.nbytes

    def _result(self, session: str, request_id: int) -> dict:
        flow = self._flows[session]
        result = {"generation": self.generation, "request_id": request_id}
        if flow.pending is not None and flow.pending.request_id == request_id:
            return {**result, "status": "pending"}
        lease_id = f"{session}:{request_id}"
        if lease_id in self._leases:
            return {**result, "status": "granted", "lease_id": lease_id}
        return {**result, "status": "retired"}

    def request(self, session: str, request_id: int, nbytes: int, traffic: str = "bulk") -> dict:
        self._validate_request_id(request_id)
        if type(nbytes) is not int or not 0 < nbytes <= self.limits.max_bytes:
            raise EgressError("Fragment bytes must be positive and fit the shared byte budget")
        if self.limits.rate_bytes_per_s and nbytes > self.limits.burst_bytes:
            raise EgressError("Fragment exceeds the rate bucket's burst capacity")
        if traffic not in ("latency", "bulk"):
            raise EgressError("traffic must be latency or bulk")
        with self._lock:
            now = self._clock()
            self._check(now)
            flow = self._flow(session)
            if request_id > flow.next_request_id:
                raise EgressError("Request IDs must be consecutive")
            if request_id < flow.next_request_id:
                previous = self._leases.get(f"{session}:{request_id}")
                if flow.pending is not None and flow.pending.request_id == request_id:
                    previous = flow.pending
                if previous is not None and (previous.nbytes != nbytes or previous.traffic != traffic):
                    raise EgressError("Retry changed the pending/granted request")
            else:
                if flow.pending is not None:
                    raise EgressError("A flow may have only one pending request")
                request = _Request(
                    session,
                    request_id,
                    nbytes,
                    traffic,
                    max(flow.finish[traffic], self._virtual_time[traffic]) + nbytes,
                    self._order,
                )
                self._order += 1
                flow.pending = request
                flow.next_request_id += 1
            self._pump(now)
            return self._result(session, request_id)

    def complete(self, session: str, lease_id: str) -> dict:
        """Return capacity only after the sender has proved backend completion."""
        with self._lock:
            now = self._clock()
            self._check(now)
            flow = self._flow(session)
            if not isinstance(lease_id, str) or not lease_id.startswith(f"{session}:"):
                raise EgressError("Lease does not belong to this session")
            suffix = lease_id[len(session) + 1 :]
            if not suffix.isascii() or not suffix.isdigit() or len(suffix) > 20:
                raise EgressError("Invalid lease ID")
            request_id = int(suffix)
            if str(request_id) != suffix or request_id >= flow.next_request_id:
                raise EgressError("Unknown lease ID")
            if flow.pending is not None and flow.pending.request_id == request_id:
                raise EgressError("A pending request has no completion lease")
            request = self._leases.pop(lease_id, None)
            if request is not None:
                self._used_bytes -= request.nbytes
            self._pump(now)
            return {"generation": self.generation, "status": "completed" if request else "retired"}

    def cancel(self, session: str, request_id: int, *, unposted: bool = False) -> dict:
        """Cancel waiting work, or a grant whose caller proves it never posted."""
        self._validate_request_id(request_id)
        with self._lock:
            now = self._clock()
            self._check(now)
            flow = self._flow(session)
            if request_id >= flow.next_request_id:
                raise EgressError("Unknown request ID")
            if flow.pending is not None and flow.pending.request_id == request_id:
                flow.pending = None
            elif (request := self._leases.get(f"{session}:{request_id}")) is not None:
                if unposted is not True:
                    raise EgressError("A grant can only be cancelled with proof that it was never posted")
                del self._leases[request.lease_id]
                self._used_bytes -= request.nbytes
                # Grant tokens stay spent: refunding them could create bursts.
            else:
                return {"generation": self.generation, "status": "retired"}
            self._pump(now)
            return {"generation": self.generation, "status": "cancelled"}

    def disconnect(self, session: str) -> None:
        """A lost connection cannot prove that any granted DMA has stopped."""
        with self._lock:
            flow = self._flows.get(session)
            if flow is None:
                return
            if any(request.session == session for request in self._leases.values()):
                if self._failure is None:
                    self._failure = f"Flow {flow.flow_id} disconnected with live grants"
                return
            del self._flows[session]
            del self._flow_ids[flow.flow_id]
            now = self._clock()
            self._expire(now)
            if self._failure is None:
                self._pump(now)

    def fail(self, reason: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = str(reason)[:1024]

    def tick(self) -> None:
        with self._lock:
            now = self._clock()
            self._check(now)
            self._pump(now)

    def stats(self) -> dict:
        with self._lock:
            self._expire(self._clock())
            return {
                "generation": self.generation,
                "failed": self._failure is not None,
                "failure": self._failure,
                "flows": len(self._flows),
                "pending": sum(flow.pending is not None for flow in self._flows.values()),
                "inflight_fragments": len(self._leases),
                "inflight_bytes": self._used_bytes,
                "granted_fragments": self._granted_fragments,
                "granted_bytes": self._granted_bytes,
                "peak_inflight_fragments": self._peak_inflight_fragments,
                "peak_inflight_bytes": self._peak_inflight_bytes,
                "rate_tokens": self._tokens,
            }
