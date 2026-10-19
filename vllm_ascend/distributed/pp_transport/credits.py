# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Thread-safe byte/operation reservations, independent of PP or KV semantics."""

from threading import Lock


class CreditBudget:
    """A local budget that may be shared by PP and a future PD adapter.

    This is not a cross-process NIC scheduler. A caller owns every returned
    reservation until its particular resource's completion condition is met.
    Source completion and destination consumption use *different* budgets.
    """

    def __init__(self, max_frames: int, max_bytes: int):
        if max_frames <= 0 or max_bytes <= 0:
            raise ValueError("Credit limits must be positive")
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self._frames = 0
        self._bytes = 0
        self._lock = Lock()

    def try_reserve(self, size: int) -> "CreditReservation | None":
        if size < 0 or size > self.max_bytes:
            raise ValueError("Reservation exceeds the byte budget")
        with self._lock:
            if self._frames == self.max_frames or self._bytes + size > self.max_bytes:
                return None
            self._frames += 1
            self._bytes += size
            return CreditReservation(self, size)

    def usage(self) -> tuple[int, int]:
        with self._lock:
            return self._frames, self._bytes


class CreditReservation:
    def __init__(self, budget: CreditBudget, size: int):
        self._budget = budget
        self.size = size
        self._released = False

    def shrink(self, size: int) -> None:
        with self._budget._lock:
            if self._released or not 0 <= size <= self.size:
                raise ValueError("Only a live reservation can shrink")
            self._budget._bytes -= self.size - size
            self.size = size

    def release(self) -> None:
        with self._budget._lock:
            if self._released:
                raise RuntimeError("Credit already released")
            self._released = True
            self._budget._frames -= 1
            self._budget._bytes -= self.size
