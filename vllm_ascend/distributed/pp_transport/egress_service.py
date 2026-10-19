# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Explicitly started, bounded TCP admission service for a shared WAN exit.

One service owns one physical-budget scope across worker processes and hosts.
This control protocol is for a trusted private network; it has no authentication
or encryption. The service never transfers tensors or infers DMA completion.
"""

import argparse
import json
import math
import select
import socket
import socketserver
import struct
import threading
from contextlib import suppress
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from .egress import EgressArbiter, EgressError, EgressLimits

_MAX_MESSAGE_BYTES = 64 * 1024
_PROTOCOL_VERSION = 1
_DEFAULT_PORT = 29590
_IDLE_POLL_INTERVAL_S = 0.1


def _timeout(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("socket timeout must be finite and positive")
    return float(value)


def _address(endpoint: str | tuple[str, int]) -> tuple[str, int]:
    if isinstance(endpoint, str):
        parsed = urlsplit(endpoint if "://" in endpoint else f"tcp://{endpoint}")
        if parsed.scheme != "tcp" or parsed.username is not None or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("egress endpoint must be host:port or tcp://host:port")
        address = (parsed.hostname, parsed.port)
    else:
        address = endpoint
    if (
        not isinstance(address, tuple)
        or len(address) != 2
        or not isinstance(address[0], str)
        or not address[0]
        or type(address[1]) is not int
        or not 0 < address[1] <= 65535
    ):
        raise ValueError("egress endpoint requires a host and a port in [1, 65535]")
    return address


def _read_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise EOFError("egress control connection closed")
        result.extend(chunk)
    return bytes(result)


def _read_message(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _read_exact(connection, 4))[0]
    if not 0 < size <= _MAX_MESSAGE_BYTES:
        raise ValueError("invalid egress control message size")
    value = json.loads(_read_exact(connection, size))
    if not isinstance(value, dict):
        raise ValueError("egress control message must be an object")
    return value


def _write_message(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(payload) <= _MAX_MESSAGE_BYTES:
        raise ValueError("egress control message exceeds its size limit")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


class _EgressHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        session = None
        self.request.settimeout(self.server.socket_timeout_s)
        try:
            hello = _read_message(self.request)
            if hello.get("op") != "hello" or hello.get("version") != _PROTOCOL_VERSION:
                raise ValueError("unsupported egress control handshake")
            session = self.server.arbiter.register(hello.get("flow_id"))
            _write_message(
                self.request,
                {
                    "ok": True,
                    "session": session,
                    "version": _PROTOCOL_VERSION,
                    "generation": self.server.arbiter.generation,
                    "limits": asdict(self.server.limits),
                },
            )
            while not self.server.closing.is_set():
                # No request is not a stalled request: enterprise workers may
                # sit idle for hours between batches. Apply socket deadlines
                # only after readability indicates a message (or peer EOF).
                readable, _, _ = select.select([self.request], [], [], _IDLE_POLL_INTERVAL_S)
                if not readable:
                    continue
                message = _read_message(self.request)
                operation = message.get("op")
                if operation == "acquire":
                    result = self.server.arbiter.request(
                        session, message.get("request_id"), message.get("nbytes"), message.get("traffic")
                    )
                elif operation == "complete":
                    result = self.server.arbiter.complete(session, message.get("lease_id"))
                elif operation == "cancel":
                    unposted = message.get("unposted", False)
                    if type(unposted) is not bool:
                        raise ValueError("unposted must be a boolean")
                    result = self.server.arbiter.cancel(session, message.get("request_id"), unposted=unposted)
                elif operation == "stats":
                    result = self.server.arbiter.stats()
                elif operation == "close":
                    # disconnect retains all uncertain issued work and poisons
                    # the shared scope if the caller still owns a grant.
                    self.server.arbiter.disconnect(session)
                    session = None
                    _write_message(
                        self.request,
                        {"ok": True, "result": {"status": "closed", "generation": self.server.arbiter.generation}},
                    )
                    return
                else:
                    raise ValueError("unknown egress control operation")
                _write_message(self.request, {"ok": True, "result": result})
        except (EOFError, OSError):
            pass
        except (EgressError, ValueError, TypeError, KeyError) as error:
            with suppress(OSError, ValueError):
                _write_message(self.request, {"ok": False, "error": str(error)[:1024]})
        finally:
            if session is not None:
                self.server.arbiter.disconnect(session)


class EgressBudgetServer(socketserver.ThreadingTCPServer):
    """One arbiter, at most ``limits.max_flows`` connection-owner threads.

    Construction binds the listener but never starts a background service.
    Call ``serve_forever`` explicitly and stop it before ``server_close``.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], limits: EgressLimits, *, socket_timeout_s: float = 300.0) -> None:
        self.limits = limits
        self.arbiter = EgressArbiter(limits)
        self.socket_timeout_s = _timeout(socket_timeout_s)
        self.closing = threading.Event()
        self._connection_slots = threading.BoundedSemaphore(limits.max_flows)
        self._connections_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _EgressHandler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        # Gate before ThreadingMixIn creates a thread, including connections
        # that never send a handshake. The kernel listen backlog is bounded too.
        if not self._connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        with self._connections_lock:
            self._connections.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._remove_connection(request)
            raise

    def _remove_connection(self, request: socket.socket) -> None:
        with self._connections_lock:
            self._connections.discard(request)
        self._connection_slots.release()

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._remove_connection(request)

    def service_actions(self) -> None:
        # Keep the failed generation available for diagnostics. Restarting
        # this listener cannot establish quiescence of old device writes.
        with suppress(EgressError):
            self.arbiter.tick()

    def server_close(self) -> None:
        self.closing.set()
        with self._connections_lock:
            connections = tuple(self._connections)
        for connection in connections:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()
        super().server_close()


class EgressLease:
    """A grant belongs to the client until proven complete or never posted."""

    def __init__(self, client: "EgressClient", request_id: int, lease_id: str) -> None:
        self._client = client
        self.request_id = request_id
        self.lease_id = lease_id

    def release(self) -> None:
        """Release only after the backend proves transfer completion."""
        self._client._retire(self, unposted=False)

    def safe_cancel_before_submit(self) -> None:
        """Cancel only when no backend call could have submitted this fragment."""
        self._client._retire(self, unposted=True)

    def cancel(self) -> None:
        self.safe_cancel_before_submit()


class EgressClient:
    """Short serialized RPCs; no lock or socket call waits for a grant.

    Polling the same pending acquisition is idempotent. Source completion can
    release older leases concurrently with submission polling. Broken control
    connections never reconnect silently or refund potentially submitted work.
    """

    def __init__(self, endpoint: str | tuple[str, int], flow_id: str, *, timeout_s: float = 5.0) -> None:
        self._lock = threading.RLock()
        self._error: str | None = None
        self._closed = False
        self._pending: tuple[int, int, str] | None = None
        self._next_request = 0
        self._issued: set[str] = set()
        self._connection = socket.create_connection(_address(endpoint), timeout=_timeout(timeout_s))
        try:
            _write_message(self._connection, {"op": "hello", "version": _PROTOCOL_VERSION, "flow_id": flow_id})
            hello = _read_message(self._connection)
            if not hello.get("ok"):
                raise EgressError(hello.get("error", "egress registration failed"))
            if (
                hello.get("version") != _PROTOCOL_VERSION
                or not isinstance(hello.get("session"), str)
                or not isinstance(hello.get("generation"), str)
            ):
                raise EgressError("invalid egress registration response")
            self.session = hello["session"]
            self.generation = hello["generation"]
            self.limits = EgressLimits(**hello["limits"])
        except BaseException:
            self._disconnect()
            raise

    def _disconnect(self) -> None:
        with suppress(OSError):
            self._connection.shutdown(socket.SHUT_RDWR)
        self._connection.close()
        self._closed = True

    def check_error(self) -> None:
        if self._error is not None:
            raise EgressError(self._error)
        if self._closed:
            raise EgressError("egress client is closed")

    def _rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        self.check_error()
        try:
            _write_message(self._connection, message)
            response = _read_message(self._connection)
            if not response.get("ok"):
                raise EgressError(response.get("error", "egress operation failed"))
            result = response.get("result")
            if not isinstance(result, dict) or result.get("generation") != self.generation:
                raise EgressError("invalid egress response")
            return result
        except (EOFError, OSError, ValueError, EgressError) as error:
            self._error = str(error)
            self._disconnect()
            raise EgressError(f"egress control failed: {error}") from error

    def try_acquire(self, nbytes: int, traffic: str = "bulk") -> EgressLease | None:
        with self._lock:
            if self._pending is None:
                self._pending = (self._next_request, nbytes, traffic)
                self._next_request += 1
            request_id, pending_bytes, pending_traffic = self._pending
            if (nbytes, traffic) != (pending_bytes, pending_traffic):
                raise EgressError("finish or cancel the pending fragment before changing its size or traffic")
            result = self._rpc({"op": "acquire", "request_id": request_id, "nbytes": nbytes, "traffic": traffic})
            if result.get("status") == "pending":
                return None
            if result.get("status") != "granted" or not isinstance(result.get("lease_id"), str):
                self.fail("invalid grant response")
                raise EgressError("invalid grant response")
            self._pending = None
            lease = EgressLease(self, request_id, result["lease_id"])
            self._issued.add(lease.lease_id)
            return lease

    def _retire(self, lease: EgressLease, *, unposted: bool) -> None:
        with self._lock:
            if lease.lease_id not in self._issued:
                raise EgressError("lease is not live or was already released")
            if unposted:
                self._rpc({"op": "cancel", "request_id": lease.request_id, "unposted": True})
            else:
                self._rpc({"op": "complete", "lease_id": lease.lease_id})
            self._issued.remove(lease.lease_id)

    def cancel_pending(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._rpc({"op": "cancel", "request_id": self._pending[0], "unposted": True})
                self._pending = None

    def fail(self, reason: Any) -> None:
        """Disconnect immediately, without waiting for cancellation RPCs.

        Server-side live grants, including an unobserved pending grant, poison
        the scope. Normal teardown uses ``close`` to cancel unposted work first.
        """
        with self._lock:
            if self._error is None:
                self._error = str(reason)
            self._disconnect()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                # The arbiter may already have granted the pending request
                # after our last poll. No lease was returned to submitters,
                # so explicitly cancelling it is safe before disconnecting.
                self.cancel_pending()
                self._rpc({"op": "close"})
            finally:
                self._disconnect()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"failed": self._error is not None, "closed": True, "client_live_leases": len(self._issued)}
            return self._rpc({"op": "stats"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--max-fragments", type=int, required=True)
    parser.add_argument("--rate-bytes-per-s", type=float, default=0)
    parser.add_argument("--burst-bytes", type=int, default=0)
    parser.add_argument("--latency-burst-bytes", type=int, default=0)
    parser.add_argument("--max-flows", type=int, default=128)
    parser.add_argument("--lease-timeout-s", type=float, default=300)
    parser.add_argument(
        "--socket-timeout-s", type=float, default=300, help="Handshake/message I/O timeout; not an idle-session limit."
    )
    args = parser.parse_args()
    limits = EgressLimits(
        max_bytes=args.max_bytes,
        max_fragments=args.max_fragments,
        rate_bytes_per_s=args.rate_bytes_per_s,
        burst_bytes=args.burst_bytes,
        latency_burst_bytes=args.latency_burst_bytes,
        max_flows=args.max_flows,
        lease_timeout_s=args.lease_timeout_s,
    )
    with EgressBudgetServer((args.host, args.port), limits, socket_timeout_s=args.socket_timeout_s) as server:
        print(f"Shared egress budget listening on {server.server_address}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
