# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded Gloo control messages and replaceable torch PP data operations."""

from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import torch.distributed as dist

# Existing GroupCoordinator object/tensor traffic uses tag zero. These tags
# belong exclusively to one WAN PP transport on each existing PP CPU group.
_HEADER_LENGTH_TAG = 31201
_HEADER_PAYLOAD_TAG = 31202
_ACK_LENGTH_TAG = 31203
_ACK_PAYLOAD_TAG = 31204
_CPU_DATA_TAG = 31205


@dataclass(frozen=True)
class BackendCapabilities:
    """Completion requires DeviceRuntime.finish, not merely Work.wait."""

    asynchronous_data: bool = True
    source_safe_after_finish: bool = True
    target_visible_after_finish: bool = True
    consumer_release_required: bool = True
    cancel_proves_quiescence: bool = False
    control_replay: bool = False


class TorchPPBackend:
    """Use the PP data channel without changing the HCCL or Gloo ABI.

    Rank arguments are global ranks. Control calls are blocking and must run
    only on their dedicated progress owners; partial publication errors escape
    to the transport, which retains affected data slots until teardown.
    """

    capabilities = BackendCapabilities()

    def __init__(self, group: Any, config: Any) -> None:
        self.group = group
        # Ascend's GroupCoordinator stores the local NPU ordinal as an int.
        self.device = torch.device("npu", group.device) if isinstance(group.device, int) else torch.device(group.device)
        self.max_metadata_bytes = config.max_metadata_bytes
        if type(self.max_metadata_bytes) is not int or self.max_metadata_bytes < 1:
            raise ValueError("max_metadata_bytes must be a positive integer")
        self._cpu_group = group.cpu_group
        self._data_group = group.cpu_group if self.device.type == "cpu" else group.device_group

    def _send(self, value: dict[str, Any], dst: int, length_tag: int, payload_tag: int) -> None:
        if not isinstance(value, dict):
            raise ValueError("Pipeline control messages must be dictionaries")
        payload = msgpack.packb(value, use_bin_type=True)
        if not 0 < len(payload) <= self.max_metadata_bytes:
            raise ValueError("Pipeline metadata exceeds max_metadata_bytes")
        length = torch.tensor([len(payload)], dtype=torch.int64, device="cpu")
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        dist.send(length, dst=dst, group=self._cpu_group, tag=length_tag)
        dist.send(data, dst=dst, group=self._cpu_group, tag=payload_tag)

    def _recv(self, src: int, length_tag: int, payload_tag: int) -> dict[str, Any]:
        length = torch.empty(1, dtype=torch.int64, device="cpu")
        dist.recv(length, src=src, group=self._cpu_group, tag=length_tag)
        size = int(length.item())
        if not 0 < size <= self.max_metadata_bytes:
            raise ValueError("Invalid pipeline metadata length")
        data = torch.empty(size, dtype=torch.uint8, device="cpu")
        dist.recv(data, src=src, group=self._cpu_group, tag=payload_tag)
        value = msgpack.unpackb(data.numpy().tobytes(), raw=False, strict_map_key=True)
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("Invalid pipeline control message")
        return value

    def publish(self, header: dict[str, Any], dst: int) -> None:
        self._send(header, dst, _HEADER_LENGTH_TAG, _HEADER_PAYLOAD_TAG)

    def receive_header(self, src: int) -> dict[str, Any]:
        return self._recv(src, _HEADER_LENGTH_TAG, _HEADER_PAYLOAD_TAG)

    def send_ack(self, ack: dict[str, Any], dst: int) -> None:
        self._send(ack, dst, _ACK_LENGTH_TAG, _ACK_PAYLOAD_TAG)

    def recv_ack(self, src: int) -> dict[str, Any]:
        return self._recv(src, _ACK_LENGTH_TAG, _ACK_PAYLOAD_TAG)

    def submit_send(self, buffer: torch.Tensor, dst: int) -> list[Any]:
        if buffer.numel() == 0:
            return []
        kwargs = {"tag": _CPU_DATA_TAG} if self.device.type == "cpu" else {}
        return [dist.isend(buffer, dst=dst, group=self._data_group, **kwargs)]

    def submit_recv(self, buffer: torch.Tensor, src: int) -> list[Any]:
        if buffer.numel() == 0:
            return []
        kwargs = {"tag": _CPU_DATA_TAG} if self.device.type == "cpu" else {}
        return [dist.irecv(buffer, src=src, group=self._data_group, **kwargs)]
