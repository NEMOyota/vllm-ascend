# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded layer-event publication and nonblocking source-readiness polling.

Only control metadata travels here. Payloads continue through the managed
Mooncake READ/staging path. A layer event permits a READ; only a sealed source
publication permits the receiver to report decode-ready.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import msgspec
import zmq

from .config import WanPDConfig
from .manifest import ContinuationManifest
from .publication import GET_READY_MSG, UPDATE_READY_MSG


class _Exchange:
    """One outstanding REQ, finite retries, and no blocking send/receive."""

    def __init__(self, host: str, port: int, config: WanPDConfig, *, clock=monotonic) -> None:
        self.endpoint = f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"
        self.config, self.clock = config, clock
        self.context: Any = None
        self.socket: Any = None
        self.sent_at: float | None = None
        self.started_at: float | None = None
        self.attempts = 0

    def _reset_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.sent_at = None

    def advance(self, payload: bytes) -> bytes | None:
        now = self.clock()
        if self.started_at is None:
            self.started_at = now
        if now - self.started_at >= self.config.timeout_s:
            raise TimeoutError("WAN PD readiness exchange expired")
        if self.socket is None:
            if self.context is None:
                self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDHWM, 1)
            self.socket.setsockopt(zmq.RCVHWM, 1)
            self.socket.setsockopt(zmq.IMMEDIATE, 1)
            self.socket.setsockopt(zmq.MAXMSGSIZE, self.config.max_manifest_bytes)
            self.socket.connect(self.endpoint)
        if self.sent_at is None:
            try:
                self.socket.send(payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
            self.sent_at = now
            self.attempts += 1
            return None
        try:
            result = self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            if now - self.sent_at >= min(1.0, self.config.timeout_s):
                self._reset_socket()
                if self.attempts >= self.config.commit_retry_limit:
                    raise TimeoutError("WAN PD readiness exchange exhausted retries")
            return None
        self.sent_at = self.started_at = None
        self.attempts = 0
        if len(result) > self.config.max_manifest_bytes:
            raise ValueError("WAN PD readiness reply exceeds metadata limit")
        return result

    def close(self) -> None:
        self._reset_socket()
        if self.context is not None:
            self.context.term()
            self.context = None


class ReadinessGate:
    """A request-scoped source catalog client owned by the progress thread."""

    def __init__(
        self,
        config: WanPDConfig,
        manifest: ContinuationManifest,
        host: str,
        port: int,
        *,
        exchange: Any = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config, self.clock = config, clock
        self.exchange = exchange or _Exchange(host, port, config, clock=clock)
        self.payload = msgspec.msgpack.encode(
            (GET_READY_MSG, manifest.engine_epoch, manifest.attempt_id, manifest.digest)
        )
        self.ready_layers: set[str] = set()
        self.sealed = False
        self.next_query = 0.0

    def advance(self) -> None:
        if self.sealed or self.clock() < self.next_query:
            return
        response = self.exchange.advance(self.payload)
        if response is None:
            return
        value = msgspec.msgpack.decode(response)
        if not isinstance(value, dict) or value.get("status") not in ("PREPARED", "SEALED", "FAILED"):
            raise ValueError("WAN PD source rejected layer readiness query")
        if value["status"] == "FAILED":
            raise RuntimeError(f"WAN PD source publication failed: {value.get('error')}")
        layers = value.get("ready_layers")
        if not isinstance(layers, list) or len(layers) > self.config.max_manifest_ranges:
            raise ValueError("WAN PD source returned an invalid readiness set")
        if any(not isinstance(layer, str) or not 0 < len(layer) <= 8192 for layer in layers):
            raise ValueError("WAN PD source returned an invalid layer identity")
        ready = set(layers)
        if not self.ready_layers <= ready:
            raise ValueError("WAN PD source readiness regressed within one attempt")
        self.ready_layers = ready
        self.sealed = value["status"] == "SEALED"
        self.next_query = self.clock() + 0.002

    def permits(self, layer_key: str) -> bool:
        if not layer_key:
            raise ValueError("Early PD transfer range has no source layer identity")
        return layer_key in self.ready_layers

    def close(self) -> None:
        self.exchange.close()


@dataclass
class _Publication:
    request_id: str
    manifest: ContinuationManifest
    layers: frozenset[str]
    events: dict[str, Any] = field(default_factory=dict)
    ready: set[str] = field(default_factory=set)
    final_event: Any = None
    finalized: bool = False
    acknowledged: tuple[frozenset[str], bool] = (frozenset(), False)
    rejected: bool = False


class LayerEventPublisher:
    """Publish completed device events without blocking the model thread.

    The scheduler admits only eager, non-speculative final-prefill steps with
    stable full-attention state. Entries and their events survive until an
    explicit scheduler retirement; a deadline never grants permission to reuse
    source memory.
    """

    def __init__(
        self,
        config: WanPDConfig,
        worker_key: str,
        layer_names: list[str],
        *,
        event_factory: Callable[[], Any],
        initialize_thread: Callable[[], None] = lambda: None,
        exchange_factory: Callable[..., Any] = _Exchange,
    ) -> None:
        self.config, self.worker_key = config, worker_key
        self.layer_names = frozenset(layer_names)
        self.event_factory, self.initialize_thread = event_factory, initialize_thread
        self.exchange_factory = exchange_factory
        self._lock = Lock()
        self._entries: dict[str, _Publication] = {}
        self._order: deque[str] = deque()
        self._current: set[str] = set()
        self._pending: tuple[str, bytes, tuple[frozenset[str], bool], str] | None = None
        self._endpoint: tuple[str, int] | None = None
        self._exchange: Any = None
        self._wake, self._stop = Event(), Event()
        self.failure: Exception | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("WAN PD layer publisher is already running")
        self._thread = Thread(target=self._run, name="WanPDLayerPublisher", daemon=True)
        self._thread.start()

    def begin(self, publications: Mapping[str, dict[str, Any]], finished: set[str]) -> None:
        if self.failure is not None:
            raise RuntimeError("WAN PD layer publisher failed") from self.failure
        with self._lock:
            for request_id in finished:
                self._entries.pop(request_id, None)
            self._order = deque(request_id for request_id in self._order if request_id in self._entries)
            self._current = set()
            for request_id, params in publications.items():
                manifest = ContinuationManifest.from_wire(params.get("continuation_manifest"), self.config)
                if manifest.publication_mode != "layerwise" or manifest.request_id != request_id:
                    raise ValueError("Invalid producer layer publication metadata")
                owned = frozenset(component.layer_name for component in manifest.components) & self.layer_names
                if any(component.kind != "full_attention" for component in manifest.components):
                    raise ValueError("Early publication requires stable full-attention state")
                if not owned:
                    # Embedding/output-only PP endpoints own no source state.
                    continue
                endpoint = (params["remote_host"], params["remote_port"])
                if self._endpoint is not None and self._endpoint != endpoint:
                    raise ValueError("Producer layer publisher changed scheduler endpoint")
                self._endpoint = endpoint
                existing = self._entries.get(request_id)
                if existing is not None:
                    if existing.manifest.digest != manifest.digest:
                        raise ValueError("Producer request ID already owns another publication")
                    # Replaying scheduler metadata cannot start another writer.
                    raise ValueError("Producer final-prefill publication was scheduled twice")
                if len(self._entries) >= self.config.max_retained_source_requests:
                    raise ValueError("Producer layer publication capacity exhausted")
                self._entries[request_id] = _Publication(request_id, manifest, owned)
                self._order.append(request_id)
                self._current.add(request_id)
        self._wake.set()

    def save_layer(self, layer_name: str) -> None:
        # Empty/unrecognized hooks are covered only by final stream completion.
        with self._lock:
            selected = [
                self._entries[rid]
                for rid in self._current
                if layer_name in self._entries[rid].layers and not self._entries[rid].rejected
            ]
            if not selected:
                return
            if any(layer_name in entry.events or layer_name in entry.ready for entry in selected):
                raise ValueError("Repeated layer hook cannot prove immutable final-prefill state")
            event = self.event_factory()
            for entry in selected:
                entry.events[layer_name] = event
        self._wake.set()

    def finalize(self) -> None:
        with self._lock:
            if not self._current:
                return
            event = self.event_factory()
            for request_id in self._current:
                self._entries[request_id].final_event = event
            self._current = set()
        self._wake.set()

    def advance(self) -> None:
        """One nonblocking control exchange; directly drivable in CPU tests."""
        with self._lock:
            pending_entry = self._entries.get(self._pending[0]) if self._pending is not None else None
            if self._pending is not None and (
                pending_entry is None or pending_entry.manifest.digest != self._pending[3]
            ):
                # Retirement is handed to this thread; a REQ socket must not
                # be closed concurrently by the model thread.
                if self._exchange is not None:
                    self._exchange.close()
                    self._exchange = None
                self._pending = None
            for entry in self._entries.values():
                if entry.rejected:
                    continue
                for layer, event in tuple(entry.events.items()):
                    if event.query():
                        entry.ready.add(layer)
                        del entry.events[layer]
                if entry.final_event is not None and entry.final_event.query():
                    entry.ready.update(entry.layers)
                    entry.events.clear()
                    entry.final_event = None
                    entry.finalized = True
            if self._pending is None:
                for _ in range(len(self._order)):
                    request_id = self._order.popleft()
                    self._order.append(request_id)
                    entry = self._entries[request_id]
                    state = (frozenset(entry.ready), entry.finalized)
                    if entry.rejected or state == entry.acknowledged:
                        continue
                    manifest = entry.manifest
                    payload = msgspec.msgpack.encode(
                        (
                            UPDATE_READY_MSG,
                            manifest.engine_epoch,
                            manifest.attempt_id,
                            manifest.digest,
                            self.worker_key,
                            sorted(entry.ready),
                            entry.finalized,
                        )
                    )
                    if len(payload) > self.config.max_manifest_bytes:
                        raise ValueError("WAN PD layer publication exceeds metadata limit")
                    self._pending = (request_id, payload, state, manifest.digest)
                    break
            pending = self._pending
        if pending is None:
            return
        if self._exchange is None:
            assert self._endpoint is not None
            self._exchange = self.exchange_factory(*self._endpoint, self.config)
        try:
            response = self._exchange.advance(pending[1])
        except TimeoutError:
            # Losing a control ACK grants no source reuse permission. Keep
            # this entry bounded and let the receiver fail its own attempt.
            self._exchange.close()
            self._exchange = None
            response = b"REJECTED"
        if response is None:
            return
        with self._lock:
            entry = self._entries.get(pending[0])
            if entry is not None and entry.manifest.digest == pending[3]:
                if response != b"ACK":
                    entry.rejected = True
                    logging.getLogger(__name__).warning(
                        "Source catalog did not accept layer events for %s; retaining publication", pending[0]
                    )
                else:
                    entry.acknowledged = pending[2]
            self._pending = None

    def _run(self) -> None:
        try:
            self.initialize_thread()
            while not self._stop.is_set():
                self.advance()
                self._wake.wait(0.002)
                self._wake.clear()
        except Exception as exc:
            self.failure = exc
        finally:
            if self._exchange is not None:
                self._exchange.close()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
