# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Receiver allocation tickets and a bounded final-publication callback.

Only control messages pass through this module. A completed callback does not
mean that remote state has been loaded: the scheduler must validate the returned
manifest and use the normal READ/copy/COMMIT lifecycle before releasing blocks.
"""

from __future__ import annotations

import http.client
import json
import socket
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, fields
from threading import Condition, Thread, Timer
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from .config import WanPDConfig
from .manifest import MAX_TEXT_LENGTH, RequestIdentity, _integer, _text, canonical_bytes, fingerprint

RESERVATION_PROTOCOL_VERSION = 1
MAX_TICKET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ReservationTicket:
    receiver_epoch: str
    reservation_id: str
    request_id: str
    rendezvous_id: str
    identity: RequestIdentity
    state_position: int
    prefix_tokens: int
    allocation_fingerprint: str
    protocol_version: int = RESERVATION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for name in ("receiver_epoch", "reservation_id", "request_id", "rendezvous_id"):
            _text(name, getattr(self, name))
        if not isinstance(self.identity, RequestIdentity):
            raise ValueError("Reservation identity must be a RequestIdentity")
        _integer("state_position", self.state_position)
        _integer("prefix_tokens", self.prefix_tokens)
        if not self.prefix_tokens <= self.state_position <= self.identity.prompt_tokens:
            raise ValueError("Reservation prefix/state position exceeds its input")
        if (
            not isinstance(self.allocation_fingerprint, str)
            or len(self.allocation_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.allocation_fingerprint)
        ):
            raise ValueError("Reservation allocation fingerprint must be a SHA256 hex digest")
        if type(self.protocol_version) is not int or self.protocol_version != RESERVATION_PROTOCOL_VERSION:
            raise ValueError("Unsupported receiver reservation protocol version")

    def to_wire(self) -> dict[str, Any]:
        return {
            "receiver_epoch": self.receiver_epoch,
            "reservation_id": self.reservation_id,
            "request_id": self.request_id,
            "rendezvous_id": self.rendezvous_id,
            "identity": self.identity.to_wire(),
            "state_position": self.state_position,
            "prefix_tokens": self.prefix_tokens,
            "allocation_fingerprint": self.allocation_fingerprint,
            "protocol_version": self.protocol_version,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_wire(), max_bytes=MAX_TICKET_BYTES)

    @classmethod
    def from_wire(cls, value: Any, config: WanPDConfig | None = None) -> ReservationTicket:
        config = config or WanPDConfig()
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ValueError("Receiver reservation must be an object")
        unknown = value.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError("Unknown receiver reservation fields")
        canonical_bytes(value, max_bytes=config.max_manifest_bytes)
        values = dict(value)
        values["identity"] = RequestIdentity.from_wire(values.get("identity"), config)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Incomplete receiver reservation") from exc


@dataclass(frozen=True)
class ReservationResult:
    request_id: str
    ticket: ReservationTicket
    transfer_params: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _Entry:
    ticket: ReservationTicket | None = None
    deadline: float = 0.0
    state: str = "reserved"
    result: ReservationResult | None = None
    reported: bool = False
    cancelled: bool = False


ReservationCallback = Callable[[dict[str, Any], float, int], dict[str, Any]]


class BoundedReservationDispatcher:
    """Reserve before D allocation, submit once, drain publication after P.

    At most ``min(reservation_workers, max_active_requests)`` callbacks execute
    concurrently. Reserved, queued, completed and cancelled-but-still-active
    entries all count against ``max_pending_requests``. There is no executor
    work queue. ``poll_completed`` reports deadlines even if an injected callback
    or an OS resolver is stuck; such a call keeps its worker/entry credit until
    it really exits. Its late result can never attach to another allocation.

    A callback is injected for CPU tests with signature
    ``callback({"ticket": ticket.to_wire()}, timeout_s, max_response_bytes)``.
    It must return ``{"transfer_params": {...}}``. Production HTTP uses only
    the administrator's configured URL and never follows redirects.
    """

    def __init__(
        self,
        config: WanPDConfig,
        receiver_epoch: str,
        callback: ReservationCallback | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config = config
        self.receiver_epoch = _text("receiver_epoch", receiver_epoch)
        self._clock = clock
        self._callback = callback or self._http_callback
        self._url = None
        if callback is None:
            url = config.reservation_callback_url
            if not isinstance(url, str) or not 0 < len(url) <= MAX_TEXT_LENGTH:
                raise ValueError("Receiver-first PD requires an administrator callback URL")
            parsed = urlsplit(url)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.fragment
            ):
                raise ValueError("Reservation callback URL must be HTTP(S), without credentials or fragment")
            # Validate malformed ports at configuration time, not after allocating D blocks.
            _ = parsed.port
            self._url = parsed
        self._condition = Condition()
        self._entries: dict[str, _Entry] = {}
        self._closed = False
        self._threads: list[Thread] = []

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Receiver reservation dispatcher is closed")
            if self._threads:
                return
            for index in range(min(self.config.reservation_workers, self.config.max_active_requests)):
                thread = Thread(target=self._run, name=f"MooncakePDReservation-{index}", daemon=True)
                self._threads.append(thread)
                thread.start()

    def reserve(self, request_id: str) -> bool:
        _text("request_id", request_id)
        with self._condition:
            if self._closed:
                return False
            previous = self._entries.get(request_id)
            if previous is not None:
                return not previous.cancelled
            if len(self._entries) >= self.config.max_pending_requests:
                return False
            self._entries[request_id] = _Entry()
            return True

    def submit(
        self,
        request_id: str,
        *,
        rendezvous_id: str,
        identity: RequestIdentity,
        state_position: int,
        prefix_tokens: int,
        allocation_fingerprint: str,
    ) -> ReservationTicket:
        """Bind the actual allocation once; identical retries do not dispatch P twice."""
        with self._condition:
            entry = self._entries.get(request_id)
            if self._closed or entry is None or entry.cancelled:
                raise ValueError("Receiver callback has no live admission reservation")
            ticket = ReservationTicket(
                receiver_epoch=self.receiver_epoch,
                reservation_id=entry.ticket.reservation_id if entry.ticket else uuid.uuid4().hex,
                request_id=request_id,
                rendezvous_id=rendezvous_id,
                identity=identity,
                state_position=state_position,
                prefix_tokens=prefix_tokens,
                allocation_fingerprint=allocation_fingerprint,
            )
            ReservationTicket.from_wire(ticket.to_wire(), self.config)
            canonical_bytes({"ticket": ticket.to_wire()}, max_bytes=self.config.max_manifest_bytes)
            if entry.ticket is not None:
                if entry.ticket != ticket:
                    raise ValueError("Receiver allocation or input changed after callback submission")
                return entry.ticket
            entry.ticket = ticket
            entry.deadline = self._clock() + self.config.timeout_s
            entry.state = "queued"
            self._condition.notify_all()
            return ticket

    def _expire_locked(self) -> None:
        now = self._clock()
        for request_id, entry in self._entries.items():
            if entry.ticket and entry.result is None and not entry.cancelled and now >= entry.deadline:
                entry.result = ReservationResult(request_id, entry.ticket, error="Receiver publication timed out")
                if entry.state != "active":
                    entry.state = "completed"

    def poll_completed(self) -> list[ReservationResult]:
        """Drain results once; keep each admission until ``retire``/``cancel``."""
        with self._condition:
            self._expire_locked()
            results = []
            for entry in self._entries.values():
                if entry.result is not None and not entry.reported and not entry.cancelled:
                    results.append(entry.result)
                    entry.reported = True
            return results

    def cancel(self, request_id: str) -> None:
        """Discard publication; this does not cancel READs or release device blocks."""
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            if entry.state == "active":
                entry.cancelled = True
                entry.result = None
            else:
                del self._entries[request_id]
            self._condition.notify_all()

    def retire(self, request_id: str) -> None:
        """Release admission after the caller has resolved its block ownership."""
        self.cancel(request_id)

    def close(self, timeout: float = 0.0) -> None:
        """Reject new work and late publications, with a bounded optional join."""
        if timeout < 0:
            raise ValueError("Close timeout cannot be negative")
        with self._condition:
            self._closed = True
            for request_id, entry in list(self._entries.items()):
                if entry.state == "active":
                    entry.cancelled = True
                    entry.result = None
                else:
                    del self._entries[request_id]
            self._condition.notify_all()
        deadline = monotonic() + timeout
        for thread in self._threads:
            thread.join(timeout=max(deadline - monotonic(), 0))

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "reserved": len(self._entries),
                "active": sum(entry.state == "active" for entry in self._entries.values()),
                "queued": sum(entry.state == "queued" for entry in self._entries.values()),
                "cancelled_active": sum(entry.cancelled for entry in self._entries.values()),
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                self._expire_locked()
                candidate = next(((key, item) for key, item in self._entries.items() if item.state == "queued"), None)
                if candidate is None:
                    self._condition.wait(timeout=min(self.config.timeout_s, 1.0))
                    continue
                request_id, entry = candidate
                ticket = entry.ticket
                assert ticket is not None
                entry.state = "active"
                timeout = max(entry.deadline - self._clock(), 0)
            try:
                if timeout <= 0:
                    raise TimeoutError("Receiver publication timed out")
                response = self._callback({"ticket": ticket.to_wire()}, timeout, self.config.max_manifest_bytes)
                # Snapshot the response: a callback must not mutate a publication after validation.
                response = json.loads(canonical_bytes(response, max_bytes=self.config.max_manifest_bytes))
                if not isinstance(response, dict) or set(response) != {"transfer_params"}:
                    raise ValueError("Receiver callback must return a transfer_params object")
                params = response["transfer_params"]
                if not isinstance(params, dict):
                    raise ValueError("Receiver callback transfer_params must be an object")
                result = ReservationResult(request_id, ticket, transfer_params=params)
            except Exception as exc:
                result = ReservationResult(request_id, ticket, error=str(exc)[:MAX_TEXT_LENGTH] or type(exc).__name__)
            with self._condition:
                # Entry identity is checked in addition to request ID. A cancelled callback
                # can never publish into a newly allocated request with the same ID.
                if self._entries.get(request_id) is not entry:
                    continue
                if entry.cancelled or self._closed:
                    del self._entries[request_id]
                    self._condition.notify_all()
                    continue
                self._expire_locked()
                if entry.result is None:
                    entry.result = result
                entry.state = "completed"
                self._condition.notify_all()

    def _http_callback(self, payload: dict[str, Any], timeout_s: float, max_response_bytes: int) -> dict[str, Any]:
        assert self._url is not None
        url = self._url
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(url.hostname, port=url.port, timeout=timeout_s)
        body = canonical_bytes(payload, max_bytes=max_response_bytes)
        path = (url.path or "/") + ("?" + url.query if url.query else "")
        deadline = monotonic() + timeout_s
        try:
            connection.connect()
        except Exception:
            connection.close()
            raise
        connected_socket = connection.sock

        def stop_io() -> None:
            # A total callback deadline also interrupts a peer trickling response bytes.
            # DNS cannot always be interrupted; its fixed worker retains its credit.
            if connected_socket is not None:
                with suppress(OSError):
                    connected_socket.shutdown(socket.SHUT_RDWR)
            connection.close()

        timer = Timer(max(deadline - monotonic(), 0), stop_io)
        timer.daemon = True
        timer.start()
        try:
            if monotonic() >= deadline:
                raise TimeoutError("Receiver publication timed out")
            connection.request(
                "POST", path, body=body, headers={"Content-Type": "application/json", "Connection": "close"}
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f"Receiver callback returned HTTP {response.status}")
            content_length = response.getheader("Content-Length")
            if content_length is not None and not 0 <= int(content_length) <= max_response_bytes:
                raise ValueError("Receiver callback response exceeds max_manifest_bytes")
            data = response.read(max_response_bytes + 1)
            if len(data) > max_response_bytes:
                raise ValueError("Receiver callback response exceeds max_manifest_bytes")
            if monotonic() >= deadline:
                raise TimeoutError("Receiver publication timed out")
            return json.loads(data)
        finally:
            timer.cancel()
            connection.close()
