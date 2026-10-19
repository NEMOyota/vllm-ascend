# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fixed PP staging slots and device completion boundaries.

The transport owns a slot until its last device reader has completed. This
module deliberately does not infer ownership from Python object lifetimes.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from threading import Lock
from typing import Any

import msgpack
import torch
import torch.distributed as dist

_ALIGNMENT = 16
_DTYPE_NAMES = (
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "uint16",
    "uint32",
    "uint64",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
)


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_NAMES or not hasattr(torch, name):
        raise ValueError(f"Unsupported pipeline tensor dtype: {name!r}")
    return getattr(torch, name)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _copy_constants(value: Any, depth: int = 0) -> Any:
    """Own metadata and reject objects that require executable deserialization."""
    if depth > 32:
        raise ValueError("Pipeline metadata nesting exceeds 32 levels")
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, (list, tuple)):
        return [_copy_constants(item, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _copy_constants(item, depth + 1) for key, item in value.items()}
    raise ValueError("Pipeline metadata supports only scalar, list and string-keyed dict values")


@dataclass(frozen=True)
class TensorLayout:
    key: str
    shape: tuple[int, ...]
    dtype: str
    wire_offset: int
    wire_numel: int
    full_offset: int
    gather: bool

    @property
    def torch_dtype(self) -> torch.dtype:
        return _dtype(self.dtype)

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.torch_dtype).element_size()

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "wire_offset": self.wire_offset,
            "wire_numel": self.wire_numel,
            "full_offset": self.full_offset,
            "gather": self.gather,
        }


@dataclass(frozen=True)
class FrameLayout:
    tensors: tuple[TensorLayout, ...]
    constants: dict[str, Any]
    gather_size: int
    wire_bytes: int
    logical_bytes: int
    device_kind: str

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, Any], gather_size: int = 1, gather_rank: int = 0) -> "FrameLayout":
        _integer(gather_size, "gather_size", 1)
        _integer(gather_rank, "gather_rank")
        if gather_rank >= gather_size:
            raise ValueError("gather_rank must be smaller than gather_size")
        entries = []
        constants = {}
        wire_bytes = logical_bytes = 0
        device_kind = None
        for key, tensor in tensors.items():
            if not isinstance(key, str):
                raise ValueError("Pipeline tensor keys must be strings")
            if not isinstance(tensor, torch.Tensor):
                constants[key] = _copy_constants(tensor)
                continue
            if tensor.layout != torch.strided:
                raise ValueError("Pipeline transport requires strided tensors")
            if tensor.ndim > 32:
                raise ValueError("Pipeline transport supports at most 32 tensor dimensions")
            kind = tensor.device.type
            if kind not in ("cpu", "npu") or device_kind not in (None, kind):
                raise ValueError("Pipeline frames require tensors on one CPU or NPU device kind")
            device_kind = kind
            dtype_name = str(tensor.dtype).removeprefix("torch.")
            _dtype(dtype_name)
            numel = tensor.numel()
            gather = gather_size > 1 and numel > 0 and numel % gather_size == 0 and tensor.is_contiguous()
            wire_numel = numel // gather_size if gather else numel
            wire_bytes, logical_bytes = _align(wire_bytes), _align(logical_bytes)
            entries.append(
                TensorLayout(key, tuple(tensor.shape), dtype_name, wire_bytes, wire_numel, logical_bytes, gather)
            )
            wire_bytes += wire_numel * tensor.element_size()
            logical_bytes += numel * tensor.element_size()
        # Check that even large Python integers are representable by the wire codec.
        msgpack.packb(constants, use_bin_type=True)
        return cls(tuple(entries), constants, gather_size, wire_bytes, logical_bytes, device_kind or "none")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "constants": _copy_constants(self.constants),
            "gather_size": self.gather_size,
            "wire_bytes": self.wire_bytes,
            "logical_bytes": self.logical_bytes,
            "device_kind": self.device_kind,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrameLayout":
        expected = {"tensors", "constants", "gather_size", "wire_bytes", "logical_bytes", "device_kind"}
        if not isinstance(value, dict) or value.keys() != expected:
            raise ValueError("Invalid pipeline frame layout fields")
        gather_size = _integer(value["gather_size"], "gather_size", 1)
        if value["device_kind"] not in ("cpu", "npu", "none"):
            raise ValueError("Invalid pipeline device kind")
        if not isinstance(value["tensors"], list) or not isinstance(value["constants"], dict):
            raise ValueError("Invalid pipeline tensor or constant collection")
        constants = _copy_constants(value["constants"])
        entries = []
        keys = set(constants)
        wire_bytes = logical_bytes = 0
        fields = {"key", "shape", "dtype", "wire_offset", "wire_numel", "full_offset", "gather"}
        for entry in value["tensors"]:
            if not isinstance(entry, dict) or entry.keys() != fields:
                raise ValueError("Invalid pipeline tensor layout fields")
            key = entry["key"]
            if not isinstance(key, str) or key in keys:
                raise ValueError("Pipeline tensor keys must be unique strings")
            keys.add(key)
            if not isinstance(entry["shape"], list) or len(entry["shape"]) > 32:
                raise ValueError("Invalid pipeline tensor shape")
            shape = tuple(_integer(dim, "shape dimension") for dim in entry["shape"])
            dtype = _dtype(entry["dtype"])
            itemsize = torch.empty((), dtype=dtype).element_size()
            numel = prod(shape)
            gather = entry["gather"]
            if type(gather) is not bool or (gather and (gather_size <= 1 or numel == 0 or numel % gather_size)):
                raise ValueError("Invalid pipeline tensor gather layout")
            wire_numel = numel // gather_size if gather else numel
            wire_bytes, logical_bytes = _align(wire_bytes), _align(logical_bytes)
            canonical = (wire_bytes, wire_numel, logical_bytes)
            actual = tuple(_integer(entry[name], name) for name in ("wire_offset", "wire_numel", "full_offset"))
            if actual != canonical:
                raise ValueError("Pipeline offsets and sizes must use the canonical non-overlapping layout")
            entries.append(TensorLayout(key, shape, entry["dtype"], wire_bytes, wire_numel, logical_bytes, gather))
            wire_bytes += wire_numel * itemsize
            logical_bytes += numel * itemsize
        if _integer(value["wire_bytes"], "wire_bytes") != wire_bytes:
            raise ValueError("Pipeline wire byte count does not match its tensor layout")
        if _integer(value["logical_bytes"], "logical_bytes") != logical_bytes or wire_bytes > logical_bytes:
            raise ValueError("Pipeline logical byte count does not match its tensor layout")
        if bool(entries) != (value["device_kind"] != "none"):
            raise ValueError("Pipeline device kind does not match its tensor layout")
        return cls(tuple(entries), constants, gather_size, wire_bytes, logical_bytes, value["device_kind"])


class _ImmediateEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class DeviceRuntime:
    """Bind every progress thread to its device and an explicit NPU stream."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "npu"):
            raise ValueError("Ascend pipeline transport supports CPU and NPU devices")

    @contextmanager
    def context(self, stream: Any = None) -> Iterator[None]:
        with torch.inference_mode():
            if self.device.type == "npu":
                with torch.npu.device(self.device):
                    if stream is None:
                        yield
                    else:
                        with torch.npu.stream(stream):
                            yield
            else:
                yield

    def new_stream(self) -> Any:
        with self.context():
            return torch.npu.Stream(device=self.device) if self.device.type == "npu" else None

    def record_event(self) -> Any:
        if self.device.type == "cpu":
            return _ImmediateEvent()
        with self.context():
            event = torch.npu.Event()
            event.record(torch.npu.current_stream(self.device))
            return event

    def wait_event(self, event: Any, stream: Any) -> None:
        if self.device.type == "cpu":
            event.synchronize()
        else:
            with self.context(stream):
                torch.npu.current_stream(self.device).wait_event(event)

    def finish(self, handles: Sequence[Any], stream: Any) -> None:
        with self.context(stream):
            for handle in handles:
                if handle is not None:
                    handle.wait()
            # HCCL Work.wait establishes a stream dependency. It is not proof
            # that a host-owned pool slot may already be overwritten.
            self.record_event().synchronize()

    def wait_on_consumer(self, event: Any) -> None:
        if self.device.type == "cpu":
            event.synchronize()
        else:
            with self.context():
                torch.npu.current_stream(self.device).wait_event(event)


class TensorBufferPool:
    """A fixed-size pool; idle storage remains inside the reserved budget."""

    def __init__(
        self, num_slots: int, slot_bytes: int, device: torch.device | str, *, materialize: bool = False
    ) -> None:
        self.num_slots = _integer(num_slots, "num_slots", 1)
        self.slot_bytes = _integer(slot_bytes, "slot_bytes", 1)
        self.device = torch.device(device)
        self.runtime = DeviceRuntime(self.device)
        with self.runtime.context():
            # Separate allocations keep every slot aligned even when its size
            # is not divisible by the widest supported dtype.
            self._storage = [torch.empty(slot_bytes, dtype=torch.uint8, device=self.device) for _ in range(num_slots)]
            self._full_storage = (
                [torch.empty(slot_bytes, dtype=torch.uint8, device=self.device) for _ in range(num_slots)]
                if materialize
                else None
            )
        self.reserved_memory_bytes = num_slots * slot_bytes * (2 if materialize else 1)
        self._leased: set[int] = set()
        self._lock = Lock()

    @staticmethod
    def describe(tensors: Mapping[str, Any], gather_size: int = 1, gather_rank: int = 0) -> FrameLayout:
        return FrameLayout.from_tensors(tensors, gather_size, gather_rank)

    def acquire(self) -> int:
        with self._lock:
            for slot in range(self.num_slots):
                if slot not in self._leased:
                    self._leased.add(slot)
                    return slot
        raise RuntimeError("Pipeline buffer pool has no available slot")

    def release(self, slot: int) -> None:
        with self._lock:
            if slot not in self._leased:
                raise RuntimeError("Pipeline buffer slot is not leased")
            self._leased.remove(slot)

    def _check_slot(self, slot: int) -> None:
        with self._lock:
            if slot not in self._leased:
                raise RuntimeError("Pipeline buffer slot is not leased")

    def buffer(self, slot: int, nbytes: int) -> torch.Tensor:
        self._check_slot(slot)
        _integer(nbytes, "nbytes")
        if nbytes > self.slot_bytes:
            raise ValueError("Pipeline frame exceeds its reserved buffer slot")
        return self._storage[slot][:nbytes]

    def _check_layout(self, layout: FrameLayout) -> None:
        if layout.logical_bytes > self.slot_bytes or layout.wire_bytes > layout.logical_bytes:
            raise ValueError("Pipeline frame exceeds its reserved buffer slot")
        if layout.device_kind not in (self.device.type, "none"):
            raise ValueError("Pipeline tensor device kind differs from the local device")

    @staticmethod
    def _view(storage: torch.Tensor, offset: int, numel: int, dtype: torch.dtype) -> torch.Tensor:
        size = torch.empty((), dtype=dtype).element_size()
        return storage[offset : offset + numel * size].view(dtype)

    def pack(self, slot: int, tensors: Mapping[str, Any], layout: FrameLayout, gather_rank: int = 0) -> None:
        self._check_layout(layout)
        self._check_slot(slot)
        _integer(gather_rank, "gather_rank")
        if gather_rank >= layout.gather_size:
            raise ValueError("gather_rank must be smaller than gather_size")
        with self.runtime.context():
            previous_end = 0
            for entry in layout.tensors:
                source = tensors[entry.key]
                if not isinstance(source, torch.Tensor) or source.device != self.device:
                    raise ValueError("Pipeline source tensor must be on the local transport device")
                if tuple(source.shape) != entry.shape or source.dtype != entry.torch_dtype:
                    raise ValueError("Pipeline source tensor differs from its published layout")
                # Alignment padding is part of the transferred byte range.
                # Do not expose bytes left by a different frame in this slot.
                self._storage[slot][previous_end : entry.wire_offset].zero_()
                target = self._view(self._storage[slot], entry.wire_offset, entry.wire_numel, entry.torch_dtype)
                previous_end = entry.wire_offset + entry.wire_numel * entry.element_size
                if entry.gather:
                    if not source.is_contiguous():
                        raise ValueError("Pipeline TP slicing requires a contiguous source tensor")
                    start = gather_rank * entry.wire_numel
                    target.copy_(source.view(-1)[start : start + entry.wire_numel])
                else:
                    target.view(entry.shape).copy_(source)

    def materialize(self, slot: int, layout: FrameLayout, all_gather_group: Any = None) -> dict[str, Any]:
        self._check_layout(layout)
        self._check_slot(slot)
        result = _copy_constants(layout.constants)
        with self.runtime.context():
            for entry in layout.tensors:
                wire = self._view(self._storage[slot], entry.wire_offset, entry.wire_numel, entry.torch_dtype)
                if entry.gather:
                    if self._full_storage is None or all_gather_group is None:
                        raise ValueError("Pipeline TP gather requires a reserved materialization buffer and TP group")
                    if all_gather_group.world_size != layout.gather_size:
                        raise ValueError("Pipeline TP size differs from its published layout")
                    full = self._view(self._full_storage[slot], entry.full_offset, entry.numel, entry.torch_dtype)
                    group = all_gather_group.cpu_group if self.device.type == "cpu" else all_gather_group.device_group
                    # Preserve dtype bits while avoiding backend-specific
                    # support requirements for float8 and metadata dtypes.
                    dist.all_gather_into_tensor(full.view(torch.uint8), wire.view(torch.uint8), group=group)
                    result[entry.key] = full.view(entry.shape)
                else:
                    result[entry.key] = wire.view(entry.shape)
        return result
