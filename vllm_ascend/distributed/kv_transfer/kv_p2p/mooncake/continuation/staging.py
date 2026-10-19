# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Registered staging owns every destination that Mooncake may write late."""

from bisect import bisect_right
from contextlib import nullcontext
from typing import Any

import torch

from .config import WanPDConfig


class _ImmediateEvent:
    def query(self) -> bool:
        return True


class StagingPool:
    """Fixed byte slots and persistent byte views of the local KV allocations.

    A quarantined slot is deliberately never returned to the free list. The
    owner must outlive the transfer engine; timeout does not unregister memory.
    """

    def __init__(self, config: WanPDConfig, kv_caches: dict[str, Any], *, device: Any = None) -> None:
        tensors = [
            tensor
            for value in kv_caches.values()
            for tensor in (value if isinstance(value, (tuple, list)) else (value,))
        ]
        if device is None:
            if not tensors:
                raise ValueError("WAN PD staging requires a cache device")
            device = tensors[0].device
        self.device = torch.device(device)
        self.slots = [
            torch.empty(config.chunk_bytes, dtype=torch.uint8, device=self.device) for _ in range(config.staging_slots)
        ]
        self._free = list(reversed(range(config.staging_slots)))
        self._leased: set[int] = set()
        self._quarantined: set[int] = set()
        self.reserved_memory_bytes = config.staging_slots * config.chunk_bytes
        self._stream = torch.npu.Stream(device=self.device) if self.device.type == "npu" else None
        storages: dict[int, torch.Tensor] = {}
        for tensor in tensors:
            if tensor.device != self.device:
                raise ValueError("WAN PD KV allocations must reside on one device")
            storage = tensor.untyped_storage()
            base, nbytes = storage.data_ptr(), storage.nbytes()
            if base not in storages:
                storages[base] = torch.empty(0, dtype=torch.uint8, device=self.device).set_(storage, 0, (nbytes,), (1,))
        self._bases = sorted(storages)
        self._views = [storages[base] for base in self._bases]

    def register(self, engine: Any) -> None:
        for slot in self.slots:
            if engine.register_memory(slot.data_ptr(), slot.numel()) != 0:
                raise RuntimeError("Mooncake WAN PD staging registration failed")

    def acquire(self) -> int | None:
        if not self._free:
            return None
        slot = self._free.pop()
        self._leased.add(slot)
        return slot

    def release(self, slot: int) -> None:
        if slot not in self._leased or slot in self._quarantined:
            raise RuntimeError("Cannot recycle an unowned or quarantined staging slot")
        self._leased.remove(slot)
        self._free.append(slot)

    def quarantine(self, slot: int) -> None:
        if slot not in self._leased:
            raise RuntimeError("Cannot quarantine an unowned staging slot")
        self._quarantined.add(slot)

    def address(self, slot: int, offset: int = 0) -> int:
        return self.slots[slot].data_ptr() + offset

    def destination(self, address: int, nbytes: int) -> torch.Tensor:
        index = bisect_right(self._bases, address) - 1
        if index < 0 or nbytes <= 0:
            raise ValueError("PD copy destination is outside local KV storage")
        offset = address - self._bases[index]
        view = self._views[index]
        if offset + nbytes > view.numel():
            raise ValueError("PD copy destination crosses local KV storage")
        return view[offset : offset + nbytes]

    def copy(self, slot: int, pieces: Any) -> Any:
        # Resolve every range before enqueueing any operation. A malformed
        # destination therefore cannot leave a half-issued device copy.
        copies = [(self.destination(piece.destination, piece.nbytes), piece) for piece in pieces]
        context = torch.npu.stream(self._stream) if self._stream is not None else nullcontext()
        with torch.inference_mode(), context:
            for destination, piece in copies:
                destination.copy_(self.slots[slot][piece.offset : piece.offset + piece.nbytes], non_blocking=True)
            if self._stream is None:
                return _ImmediateEvent()
            event = torch.npu.Event()
            event.record(self._stream)
            return event
