# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reserved, bounded completion outbox with fair delayed retries."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic

from .config import WanPDConfig

COMMIT_MSG = b"wan_pd_commit_v1"


@dataclass(frozen=True)
class CommitRecord:
    host: str
    port: int
    epoch: str
    attempt_id: str
    digest: str
    committed: bool


@dataclass
class Delivery:
    record: CommitRecord
    attempts: int = 0
    due: float = 0.0
    failed: bool = False


class BoundedCommitSender(Thread):
    def __init__(self, config: WanPDConfig, send: Callable[[CommitRecord], None]):
        super().__init__(daemon=True, name="MooncakePDCommitSender")
        self.config = config
        self.send = send
        self._entries: dict[str, Delivery | None] = {}
        self._condition = Condition()

    def reserve(self, request_id: str) -> bool:
        with self._condition:
            if request_id in self._entries:
                return True
            if len(self._entries) >= self.config.max_pending_requests:
                return False
            self._entries[request_id] = None
            return True

    def cancel_unsubmitted(self, request_id: str) -> None:
        with self._condition:
            if request_id in self._entries and self._entries[request_id] is None:
                del self._entries[request_id]

    def enqueue(self, request_id: str, record: CommitRecord) -> None:
        with self._condition:
            if request_id not in self._entries:
                raise ValueError("PD completion has no outbox reservation")
            previous = self._entries[request_id]
            if previous is not None:
                if previous.record != record:
                    raise ValueError("Conflicting PD completion record")
                return
            self._entries[request_id] = Delivery(record, due=monotonic())
            self._condition.notify()

    def progress(self, now: float | None = None) -> bool:
        """One attempt at most; a failed peer does not spin or block its queue."""
        now = monotonic() if now is None else now
        with self._condition:
            candidates = [
                (key, item)
                for key, item in self._entries.items()
                if item is not None and not item.failed and item.due <= now
            ]
            if not candidates:
                return False
            request_id, delivery = min(candidates, key=lambda pair: pair[1].due)
            delivery.attempts += 1
        try:
            self.send(delivery.record)
        except Exception as exc:
            with self._condition:
                delivery.failed = delivery.attempts >= self.config.commit_retry_limit
                delivery.due = now + self.config.commit_retry_interval_s * 2 ** (delivery.attempts - 1)
            if delivery.failed:
                logging.getLogger(__name__).error(
                    "PD completion delivery exhausted retries for attempt %s at %s:%s; retaining reservation: %s",
                    delivery.record.attempt_id,
                    delivery.record.host,
                    delivery.record.port,
                    exc,
                )
        else:
            with self._condition:
                del self._entries[request_id]
        return True

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "reserved": len(self._entries),
                "failed": sum(item is not None and item.failed for item in self._entries.values()),
            }

    def run(self) -> None:
        while True:
            if not self.progress():
                with self._condition:
                    self._condition.wait(timeout=min(self.config.commit_retry_interval_s, 1.0))
