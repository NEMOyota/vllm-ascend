# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded source pin ownership. A timeout is not a remote DMA fence."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from .config import WanPDConfig
from .manifest import ContinuationManifest


@dataclass
class SourceAttempt:
    manifest: ContinuationManifest
    published_at: float
    quarantined: bool = False


class SourceLedger:
    def __init__(self, config: WanPDConfig, engine_epoch: str):
        self.config = config
        self.engine_epoch = engine_epoch
        self._requests: dict[str, SourceAttempt | None] = {}
        self._attempts: dict[str, str] = {}
        self._committed: OrderedDict[str, str] = OrderedDict()
        self._lock = Lock()

    def reserve(self, request_id: str) -> bool:
        """Reserve before P compute, including requests not yet published."""
        with self._lock:
            if request_id in self._requests:
                return self._requests[request_id] is None
            if len(self._requests) >= self.config.max_retained_source_requests:
                return False
            self._requests[request_id] = None
            return True

    def cancel_unpublished(self, request_id: str) -> None:
        with self._lock:
            if request_id in self._requests and self._requests[request_id] is None:
                del self._requests[request_id]

    def publish(self, manifest: ContinuationManifest) -> None:
        with self._lock:
            if manifest.engine_epoch != self.engine_epoch or manifest.request_id not in self._requests:
                raise ValueError("PD source publication has no matching reservation")
            previous = self._requests[manifest.request_id]
            if previous is not None:
                if previous.manifest.digest == manifest.digest:
                    return
                raise ValueError("PD source request already published")
            if manifest.attempt_id in self._attempts or manifest.attempt_id in self._committed:
                raise ValueError("PD source attempt ID reused")
            self._requests[manifest.request_id] = SourceAttempt(manifest, monotonic())
            self._attempts[manifest.attempt_id] = manifest.request_id

    def finish(self, epoch: str, attempt_id: str, digest: str, committed: bool) -> str | None:
        """Return a request to unpin exactly once, after all D workers commit."""
        with self._lock:
            if epoch != self.engine_epoch:
                raise ValueError("Stale PD engine epoch")
            if attempt_id in self._committed:
                if self._committed[attempt_id] != digest or not committed:
                    raise ValueError("Conflicting PD terminal message")
                return None
            request_id = self._attempts.get(attempt_id)
            attempt = self._requests.get(request_id) if request_id is not None else None
            if attempt is None and committed:
                # Terminal history is bounded, whereas outstanding pins are
                # never evicted. An absent attempt in this live epoch owns no
                # source pin, so acknowledging it is a safe idempotent no-op.
                # This permits late retries after a lost ACK without freeing a
                # new request that happens to reuse the old request ID.
                return None
            if attempt is None or attempt.manifest.digest != digest:
                raise ValueError("Unknown or mismatched PD attempt")
            if not committed:
                attempt.quarantined = True
                return None
            # A later genuine commit still proves all READs and copies completed,
            # including when a control timeout previously quarantined this pin.
            del self._requests[request_id]
            del self._attempts[attempt_id]
            self._committed[attempt_id] = digest
            while len(self._committed) > self.config.max_terminal_requests:
                self._committed.popitem(last=False)
            return request_id

    def stats(self) -> dict[str, int]:
        with self._lock:
            now = monotonic()
            for attempt in self._requests.values():
                if attempt is not None and now - attempt.published_at > self.config.timeout_s:
                    attempt.quarantined = True
            return {
                "reserved": len(self._requests),
                "published": len(self._attempts),
                "quarantined": sum(item is not None and item.quarantined for item in self._requests.values()),
                "terminal": len(self._committed),
            }
