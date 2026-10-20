# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-side logic for Mooncake pull transfers."""

import queue
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.commit import SourceLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.config import WanPDConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.control import (
    COMMIT_MSG,
    BoundedCommitSender,
    CommitRecord,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import (
    ContinuationManifest,
    RequestIdentity,
    StateComponent,
    fingerprint,
    request_identity,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.publication import (
    GET_PUBLICATION_MSG,
    GET_READY_MSG,
    UPDATE_READY_MSG,
    SourcePublicationManager,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.reservation import (
    BoundedReservationDispatcher,
    ReservationTicket,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_namespace import build_cache_namespace

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
ACK_MSG = b"ACK"


class MooncakeSchedulerSendingThread(threading.Thread):
    """Serve worker metadata and collect D-side completion messages."""

    def __init__(
        self,
        host: str,
        port: int,
        engine_id: str,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        use_kv_pp: bool,
        ready_event: threading.Event,
        source_ledger: SourceLedger | None = None,
        namespace: str = "",
        publications: SourcePublicationManager | None = None,
    ) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerSendingThread")
        encoder = msgspec.msgpack.Encoder()
        self.host = host
        self.port = port
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.engine_id = engine_id
        self.use_kv_pp = use_kv_pp
        if self.use_kv_pp and self.dcp_size != 1:
            raise ValueError(f"Mooncake KVPP cannot be combined with DCP, got dcp_size={self.dcp_size}")
        self.source_ledger = source_ledger
        self.publications = publications
        self.namespace = namespace
        self.engine_epoch = source_ledger.engine_epoch if source_ledger else ""
        metadata_by_pp_rank = self._merge_metadata_by_pp_rank(metadata)
        self.metadata_by_pp_rank = metadata_by_pp_rank
        self.layout_fingerprint = (
            fingerprint(
                {
                    str(pp): {
                        key: getattr(data, key)
                        for key in (
                            "layer_names",
                            "layer_block_sizes",
                            "group_indices",
                            "block_strides",
                            "block_lens",
                            "block_shapes",
                            "block_size_scales",
                            "state_kinds",
                            "state_dtypes",
                        )
                    }
                    for pp, data in sorted(metadata_by_pp_rank.items())
                }
            )
            if source_ledger
            else ""
        )
        self.encoded_metadata = encoder.encode(
            MooncakeTransferMetadataGroups(
                engine_id=engine_id,
                scheduler_host=host,
                scheduler_port=port,
                pp_size=pp_size,
                pcp_size=pcp_size,
                dcp_size=dcp_size,
                tp_size=tp_size,
                use_kv_pp=self.use_kv_pp,
                metadata_by_pp_rank=metadata_by_pp_rank,
                engine_epoch=self.engine_epoch,
                namespace=namespace,
                layout_fingerprint=self.layout_fingerprint,
            )
        )
        logger.info(
            "Initialized Mooncake scheduler handshake metadata: engine_id=%s, "
            "worker_count=%d, pp_ranks=%s, tp_size=%d, pp_size=%d, "
            "pcp_size=%d, dcp_size=%d, use_kv_pp=%s",
            engine_id,
            len(metadata),
            sorted(metadata_by_pp_rank),
            tp_size,
            pp_size,
            pcp_size,
            dcp_size,
            self.use_kv_pp,
        )
        self.ready_event = ready_event
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.early_finished_requests: set[str] = set()
        self.finished_request_ids: set[str] = set()
        self.state_lock = threading.Lock()

    def _merge_metadata_by_pp_rank(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> dict[int, MooncakePPTransferMetadata]:
        """Merge worker metadata into one PP-wide layer table.

        LayerSplit may assign different layers to any PCP/TP worker. Per-layer
        fields are therefore aligned by layer name instead of requiring every
        worker to expose the same layer list. Worker-private address tables are
        padded to the PP-union layer order, while ``layer_indices`` records the
        entries physically owned by each PCP/TP worker.
        """
        # pp_rank -> pcp_rank -> tp_rank -> worker handshake metadata.
        workers_by_pp_rank: dict[int, dict[int, dict[int, MooncakeTransferMetadata]]] = {}
        for metadata_key, rank_metadata in metadata.items():
            if isinstance(metadata_key, int):
                pp_rank, pcp_rank, tp_rank = 0, 0, metadata_key
            elif len(metadata_key) == 2:
                pp_rank, tp_rank = metadata_key
                pcp_rank = 0
            elif len(metadata_key) == 3:
                pp_rank, pcp_rank, tp_rank = metadata_key
            else:
                raise ValueError(
                    "Mooncake handshake metadata key must be tp_rank, "
                    f"(pp_rank, tp_rank), or (pp_rank, pcp_rank, tp_rank), got {metadata_key!r}"
                )

            if not isinstance(rank_metadata, MooncakeTransferMetadata):
                raise ValueError(
                    "Mooncake scheduler expects MooncakeTransferMetadata, "
                    f"got {type(rank_metadata).__name__} for key {metadata_key!r}"
                )
            if rank_metadata.engine_id != self.engine_id:
                raise ValueError(
                    "Mooncake worker metadata engine ID mismatch: expected "
                    f"{self.engine_id!r}, got {rank_metadata.engine_id!r}"
                )

            workers_by_tp_rank = workers_by_pp_rank.setdefault(pp_rank, {}).setdefault(pcp_rank, {})
            if tp_rank in workers_by_tp_rank:
                raise ValueError(
                    f"Duplicate Mooncake metadata for PP rank {pp_rank}, PCP rank {pcp_rank}, TP rank {tp_rank}"
                )
            workers_by_tp_rank[tp_rank] = rank_metadata

        expected_pp_ranks = set(range(self.pp_size))
        if set(workers_by_pp_rank) != expected_pp_ranks:
            raise ValueError(
                "Mooncake worker metadata has incomplete PP ranks: expected "
                f"{sorted(expected_pp_ranks)}, got {sorted(workers_by_pp_rank)}"
            )

        expected_pcp_ranks = set(range(self.pcp_size))
        expected_tp_ranks = set(range(self.tp_size))
        merged: dict[int, MooncakePPTransferMetadata] = {}
        for pp_rank, workers_by_pcp_rank in sorted(workers_by_pp_rank.items()):
            if set(workers_by_pcp_rank) != expected_pcp_ranks:
                raise ValueError(
                    "Mooncake worker metadata has incomplete PCP ranks for "
                    f"PP rank {pp_rank}: expected {sorted(expected_pcp_ranks)}, "
                    f"got {sorted(workers_by_pcp_rank)}"
                )
            for pcp_rank, workers_by_tp_rank in workers_by_pcp_rank.items():
                if set(workers_by_tp_rank) != expected_tp_ranks:
                    raise ValueError(
                        "Mooncake worker metadata has incomplete TP ranks for "
                        f"PP rank {pp_rank}, PCP rank {pcp_rank}: "
                        f"expected {sorted(expected_tp_ranks)}, got {sorted(workers_by_tp_rank)}"
                    )

            reference_pcp_rank = min(workers_by_pcp_rank)
            reference_tp_rank = min(workers_by_pcp_rank[reference_pcp_rank])
            reference = workers_by_pcp_rank[reference_pcp_rank][reference_tp_rank]
            for pcp_rank, workers_by_tp_rank in workers_by_pcp_rank.items():
                for tp_rank, worker_metadata in workers_by_tp_rank.items():
                    mismatched_fields = [
                        field_name
                        for field_name in ("block_size", "num_blocks")
                        if getattr(worker_metadata, field_name) != getattr(reference, field_name)
                    ]
                    if mismatched_fields:
                        raise ValueError(
                            "Mooncake worker metadata differs across PCP/TP ranks for "
                            f"PP rank {pp_rank}: PCP/TP {reference_pcp_rank}/{reference_tp_rank} and "
                            f"{pcp_rank}/{tp_rank} mismatch in {mismatched_fields}"
                        )

            # layer_name -> (first owning worker metadata, its local layer index).
            # The source supplies PP-shared fields after all duplicate owners
            # have been checked against the corresponding structural signature.
            layer_source_by_name: dict[str, tuple[MooncakeTransferMetadata, int]] = {}
            # layer_name -> (group index, block size, strides, lengths, shapes,
            #                block-size scales).
            layer_signature_by_name: dict[str, tuple[object, ...]] = {}
            reference_layer_names: set[str] | None = None
            for pcp_rank, workers_by_tp_rank in sorted(workers_by_pcp_rank.items()):
                for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items()):
                    worker_layer_names = set(worker_metadata.layer_names)
                    if len(worker_layer_names) != len(worker_metadata.layer_names):
                        raise ValueError(
                            f"Mooncake worker metadata for PP rank {pp_rank}, PCP rank {pcp_rank}, "
                            f"TP rank {tp_rank} contains duplicate layers"
                        )
                    if reference_layer_names is None:
                        reference_layer_names = worker_layer_names
                    elif not self.use_kv_pp and worker_layer_names != reference_layer_names:
                        raise ValueError(
                            "Mooncake workers expose different KV-cache layers while KVPP is disabled: "
                            f"PP rank {pp_rank}, PCP/TP rank {pcp_rank}/{tp_rank}"
                        )
                    for local_layer_index, layer_name in enumerate(worker_metadata.layer_names):
                        layer_signature = (
                            worker_metadata.group_indices[local_layer_index],
                            worker_metadata.layer_block_sizes[local_layer_index],
                            worker_metadata.block_strides[local_layer_index],
                            worker_metadata.block_lens[local_layer_index],
                            worker_metadata.block_shapes[local_layer_index],
                            worker_metadata.block_size_scales[local_layer_index],
                        )
                        if self.source_ledger is not None:
                            if not worker_metadata.state_kinds or not worker_metadata.state_dtypes:
                                raise ValueError("WAN PD requires state kind/dtype metadata on all P workers")
                            layer_signature += (
                                worker_metadata.state_kinds[local_layer_index],
                                worker_metadata.state_dtypes[local_layer_index],
                            )
                        previous_signature = layer_signature_by_name.get(layer_name)
                        if previous_signature is not None and previous_signature != layer_signature:
                            layer_field_names = (
                                "group_indices",
                                "layer_block_sizes",
                                "block_strides",
                                "block_lens",
                                "block_shapes",
                                "block_size_scales",
                                "state_kinds",
                                "state_dtypes",
                            )
                            mismatched_layer_fields = [
                                field_name
                                for field_name, previous_value, value in zip(
                                    layer_field_names, previous_signature, layer_signature
                                )
                                if previous_value != value
                            ]
                            raise ValueError(
                                "Mooncake worker metadata differs across PCP/TP ranks for "
                                f"PP rank {pp_rank}, layer {layer_name!r}: mismatch in {mismatched_layer_fields}"
                            )
                        if previous_signature is None:
                            layer_source_by_name[layer_name] = (worker_metadata, local_layer_index)
                            layer_signature_by_name[layer_name] = layer_signature
            layer_names = sorted(layer_source_by_name)
            layer_block_sizes: list[int] = []
            group_indices: list[int] = []
            block_strides: list[list[int]] = []
            block_lens: list[list[int]] = []
            block_shapes: list[list[tuple[int, ...]]] = []
            block_size_scales: list[list[int]] = []
            for layer_name in layer_names:
                worker_metadata, local_layer_index = layer_source_by_name[layer_name]
                layer_block_sizes.append(worker_metadata.layer_block_sizes[local_layer_index])
                group_indices.append(worker_metadata.group_indices[local_layer_index])
                block_strides.append(worker_metadata.block_strides[local_layer_index])
                block_lens.append(worker_metadata.block_lens[local_layer_index])
                block_shapes.append(worker_metadata.block_shapes[local_layer_index])
                block_size_scales.append(worker_metadata.block_size_scales[local_layer_index])

            layer_index_by_name = {layer_name: layer_index for layer_index, layer_name in enumerate(layer_names)}
            metadata_by_pcp_rank: dict[int, MooncakePCPTransferMetadata] = {}
            for pcp_rank, workers_by_tp_rank in sorted(workers_by_pcp_rank.items()):
                metadata_by_tp_rank: dict[int, MooncakeTPTransferMetadata] = {}
                for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items()):
                    layer_indices = sorted(
                        layer_index_by_name[layer_name] for layer_name in worker_metadata.layer_names
                    )
                    # PP-union layer index -> this worker's per-cache-tensor
                    # addresses; unowned entries remain empty.
                    aligned_base_addrs: list[list[int]] = [[] for _ in layer_names]
                    for local_layer_index, layer_name in enumerate(worker_metadata.layer_names):
                        layer_index = layer_index_by_name[layer_name]
                        aligned_base_addrs[layer_index] = worker_metadata.kv_caches_base_addr[local_layer_index]
                    metadata_by_tp_rank[tp_rank] = MooncakeTPTransferMetadata(
                        te_rpc_port=worker_metadata.te_rpc_port,
                        layer_indices=layer_indices,
                        kv_caches_base_addr=aligned_base_addrs,
                        local_ip=worker_metadata.local_ip,
                        handshake_port=worker_metadata.handshake_port,
                    )
                metadata_by_pcp_rank[pcp_rank] = MooncakePCPTransferMetadata(metadata_by_tp_rank=metadata_by_tp_rank)

            merged[pp_rank] = MooncakePPTransferMetadata(
                block_size=reference.block_size,
                num_blocks=reference.num_blocks,
                layer_names=layer_names,
                layer_block_sizes=layer_block_sizes,
                group_indices=group_indices,
                block_strides=block_strides,
                block_lens=block_lens,
                block_shapes=block_shapes,
                block_size_scales=block_size_scales,
                metadata_by_pcp_rank=metadata_by_pcp_rank,
                state_kinds=[
                    layer_source_by_name[name][0].state_kinds[layer_source_by_name[name][1]] for name in layer_names
                ]
                if self.source_ledger
                else [],
                state_dtypes=[
                    layer_source_by_name[name][0].state_dtypes[layer_source_by_name[name][1]] for name in layer_names
                ]
                if self.source_ledger
                else [],
            )
        return merged

    def add_delayed_request(self, request_id: str, delay_start_time: float) -> None:
        with self.state_lock:
            self.delayed_free_requests[request_id] = delay_start_time
            if request_id in self.early_finished_requests:
                self.early_finished_requests.remove(request_id)
                self._mark_finished_locked(request_id)

    def get_and_clear_finished_requests(self) -> set[str]:
        finished: set[str] = set()
        with self.state_lock:
            self._retrieve_expired_requests_locked()
        while True:
            try:
                finished.add(self.finished_requests.get_nowait())
            except queue.Empty:
                return finished

    def run(self) -> None:
        path = make_zmq_path("tcp", self.host, self.port)
        try:
            logger.info("Mooncake scheduler sending thread listening on %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore[attr-defined]
                sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
                self.ready_event.set()
                self._run_busy_loop(sock)
        except Exception:
            self.ready_event.set()
            logger.exception("Mooncake scheduler sending thread failed on %s", path)

    def _run_busy_loop(self, sock: Any) -> None:
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
            except zmq.Again:  # type: ignore[attr-defined]
                continue

            identity = frames[0]
            payload = [frame for frame in frames[1:] if frame]
            if len(payload) != 1:
                logger.warning("Invalid Mooncake scheduler control frames: %s", frames)
                continue

            try:
                if self.source_ledger is not None and len(payload[0]) > self.source_ledger.config.max_manifest_bytes:
                    sock.send_multipart((identity, b"", b"REJECTED"))
                    continue
                msg = decoder.decode(payload[0])
                if msg and msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", self.encoded_metadata))
                elif msg and msg[0] in (GET_PUBLICATION_MSG, GET_READY_MSG, UPDATE_READY_MSG):
                    try:
                        if self.publications is None:
                            result = (
                                None if msg[0] == GET_PUBLICATION_MSG else {"status": "FAILED", "error": "disabled"}
                            )
                        elif msg[0] == GET_PUBLICATION_MSG and len(msg) == 2:
                            result = self.publications.get_publication(msg[1])
                        elif msg[0] == GET_READY_MSG and len(msg) == 4:
                            result = self.publications.get_ready(*msg[1:])
                        elif msg[0] == UPDATE_READY_MSG and len(msg) == 7:
                            self.publications.update_ready(*msg[1:])
                            result = ACK_MSG
                        else:
                            raise ValueError("Invalid PD publication message")
                    except (ValueError, KeyError, TypeError) as exc:
                        result = b"REJECTED" if msg[0] == UPDATE_READY_MSG else {"status": "FAILED", "error": str(exc)}
                    response = result if isinstance(result, bytes) else msgspec.msgpack.encode(result)
                    sock.send_multipart((identity, b"", response))
                elif msg and msg[0] == COMMIT_MSG and len(msg) == 5 and self.source_ledger is not None:
                    try:
                        if type(msg[4]) is not bool:
                            raise ValueError("Invalid PD commit outcome")
                        if self.publications is not None:
                            self.publications.validate_commit(*msg[1:])
                        request_id = self.source_ledger.finish(msg[1], msg[2], msg[3], msg[4])
                        if request_id is not None:
                            self.finished_requests.put(request_id)
                        response = ACK_MSG
                    except ValueError:
                        response = b"REJECTED"
                    sock.send_multipart((identity, b"", response))
                elif msg and msg[0] == DONE_RECVING_MSG and len(msg) == 2 and self.source_ledger is None:
                    self._handle_finished_request(str(msg[1]))
                    sock.send_multipart((identity, b"", ACK_MSG))
                else:
                    logger.warning("Unexpected Mooncake scheduler control message: %s", msg)
            except Exception:
                logger.exception("Failed to handle Mooncake scheduler control message")

    def _handle_finished_request(self, request_id: str) -> None:
        with self.state_lock:
            if request_id in self.finished_request_ids:
                return
            if request_id in self.delayed_free_requests:
                self._mark_finished_locked(request_id)
            else:
                self.early_finished_requests.add(request_id)

    def _mark_finished_locked(self, request_id: str) -> None:
        if request_id in self.finished_request_ids:
            return
        self.finished_request_ids.add(request_id)
        self.delayed_free_requests.pop(request_id, None)
        self.finished_requests.put(request_id)

    def _retrieve_expired_requests_locked(self) -> None:
        if self.source_ledger is not None:
            # Expiry never proves the remote device has stopped reading P memory.
            self.source_ledger.stats()
            return
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time <= envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                break

            self._mark_finished_locked(request_id)
            logger.error(
                "Force freed expired Mooncake request %s after %s seconds",
                request_id,
                envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
            )


class MooncakeSchedulerRecvingThread(threading.Thread):
    """Send scheduler-level completion messages from D to P."""

    def __init__(self, ready_event: threading.Event) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerRecvingThread")
        self.ready_event = ready_event
        self.request_queue: queue.Queue[tuple[str, int, str]] = queue.Queue()
        self.encoder = msgspec.msgpack.Encoder()
        self.remote_sockets: dict[str, deque[Any]] = defaultdict(deque)
        self.remote_sockets_lock = threading.Lock()
        self.zmq_context: Any | None = None

    def add_request(self, remote_host: str, remote_port: int, request_id: str) -> None:
        self.request_queue.put((remote_host, remote_port, request_id))

    def run(self) -> None:
        self.ready_event.set()
        while True:
            request = self.request_queue.get()

            try:
                self._send_done_recving(*request)
            except Exception:
                logger.exception(
                    "Failed to send Mooncake scheduler completion for request %s",
                    request[2],
                )
                self.request_queue.put(request)
            finally:
                self.request_queue.task_done()

    def _send_done_recving(self, remote_host: str, remote_port: int, request_id: str) -> None:
        path = make_zmq_path("tcp", remote_host, remote_port)
        sock = self._get_remote_socket(path)
        try:
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_RECVING_MSG, request_id)),
                path,
            )
            response = ensure_zmq_recv(sock, path)
            if response != ACK_MSG:
                raise RuntimeError(f"Unexpected Mooncake scheduler completion response: {response!r}")
        except Exception:
            sock.close(linger=0)
            raise
        self._return_remote_socket(path, sock)

    def _get_remote_socket(self, path: str) -> Any:
        """Borrow a persistent REQ socket for one producer scheduler."""
        with self.remote_sockets_lock:
            sockets = self.remote_sockets[path]
            if sockets:
                return sockets.popleft()

            if self.zmq_context is None:
                self.zmq_context = zmq.Context()  # type: ignore[attr-defined]
            sock = make_zmq_socket(
                ctx=self.zmq_context,
                path=path,
                socket_type=zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )
            sock.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
            return sock

    def _return_remote_socket(self, path: str, sock: Any) -> None:
        """Return a healthy REQ socket to its endpoint pool."""
        with self.remote_sockets_lock:
            self.remote_sockets[path].append(sock)


class MooncakePullConnectorScheduler(MooncakeBaseConnectorScheduler):
    """Scheduler-side Mooncake pull connector implementation."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.wan_pd = WanPDConfig.from_extra_config(self.kv_transfer_config.kv_connector_extra_config)
        self._namespace = ""
        self._source_ledger: SourceLedger | None = None
        self._commit_sender: BoundedCommitSender | None = None
        self._reservations: BoundedReservationDispatcher | None = None
        self._publications: SourcePublicationManager | None = None
        self._producer_requests: dict[str, Request] = {}
        self._prepared: dict[str, dict[str, Any]] = {}
        self._producer_finished: set[str] = set()
        self._awaiting_reservations: dict[str, tuple[Request, BlockIds, BlockIds, int, ReservationTicket]] = {}
        self._reservation_failures: dict[str, BlockIds] = {}
        self._recv_manifests: dict[str, ContinuationManifest] = {}
        self._hash_block_size = getattr(kv_cache_config, "hash_block_size", None) or 0
        if self.wan_pd.enabled:
            self._namespace = fingerprint(
                {"cache": build_cache_namespace(vllm_config), "deployment": self.wan_pd.cache_namespace}
            )
            if self.kv_transfer_config.is_kv_producer:
                self._source_ledger = SourceLedger(self.wan_pd, uuid.uuid4().hex)
                if self.wan_pd.layerwise_publication:
                    self._publications = SourcePublicationManager(self.wan_pd, self._source_ledger)
            else:
                self._commit_sender = BoundedCommitSender(self.wan_pd, self._send_commit)
                self._commit_sender.start()
                if self.wan_pd.receiver_first:
                    if not self.wan_pd.reservation_callback_url:
                        raise ValueError(
                            "Receiver-first D requires an administrator-configured reservation_callback_url"
                        )
                    self._reservations = BoundedReservationDispatcher(self.wan_pd, uuid.uuid4().hex)
                    self._reservations.start()

        # Requests waiting for the worker to start a READ transfer.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        # Producer requests whose blocks must remain allocated until read.
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()
        # D request -> (P scheduler host, port, P request id).
        self._reqs_recv_info: dict[str, tuple[str, int, str]] = {}
        self._sending_thread: MooncakeSchedulerSendingThread | None = None
        self._recving_thread: MooncakeSchedulerRecvingThread | None = None

        if self.kv_transfer_config.is_kv_consumer and not self.wan_pd.enabled:
            recving_ready_event = threading.Event()
            self._recving_thread = MooncakeSchedulerRecvingThread(recving_ready_event)
            self._recving_thread.start()
            recving_ready_event.wait()

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        if not self.kv_transfer_config.is_kv_producer or not metadata or self._sending_thread is not None:
            return

        ready_event = threading.Event()
        self._sending_thread = MooncakeSchedulerSendingThread(
            self.side_channel_host,
            self.side_channel_port,
            self.engine_id,
            metadata,
            self.tp_size,
            self.pp_size,
            self.pcp_size,
            self.dcp_size,
            self.ascend_config.kvpp_config.size > 1,
            ready_event,
            self._source_ledger,
            self._namespace,
            self._publications,
        )
        self._sending_thread.start()
        if not ready_event.wait(timeout=10):
            raise RuntimeError("Timed out starting Mooncake scheduler sending thread")
        if not self._sending_thread.is_alive():
            raise RuntimeError("Mooncake scheduler sending thread failed to start")

    def on_new_request(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if self.wan_pd.enabled and params and params.get("do_remote_decode"):
            # The base class can truncate the P prompt for recurrent state.
            # Preserve the original input identity before that mutation.
            try:
                identity = self._identity(request)
                params["_wan_pd_original_identity"] = identity.to_wire()
                if "wan_pd_reservation" in params:
                    if not self.wan_pd.receiver_first:
                        raise ValueError("Receiver-first PD is disabled on P")
                    ticket = ReservationTicket.from_wire(params["wan_pd_reservation"], self.wan_pd)
                    if (
                        ticket.identity.namespace,
                        ticket.identity.input_fingerprint,
                        ticket.identity.prompt_tokens,
                        ticket.state_position,
                    ) != (
                        identity.namespace,
                        identity.input_fingerprint,
                        identity.prompt_tokens,
                        self._state_prefill_token_count(request.num_prompt_tokens),
                    ):
                        raise ValueError("P input does not match D reservation")
            except ValueError as exc:
                # add_request() does not catch connector-hook errors. Decline
                # this handoff without terminating unrelated queued requests.
                params["_wan_pd_identity_error"] = str(exc)
                params["do_remote_decode"] = False
                logger.warning("Declining PD handoff for %s: %s", request.request_id, exc)
        super().on_new_request(request)

    def _identity(self, request: "Request") -> RequestIdentity:
        return request_identity(
            request, namespace=self._namespace, config=self.wan_pd, hash_block_size=self._hash_block_size
        )

    def get_wan_pd_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        if self._source_ledger is not None:
            stats.update({f"source_{key}": value for key, value in self._source_ledger.stats().items()})
        if self._commit_sender is not None:
            stats.update({f"completion_{key}": value for key, value in self._commit_sender.stats().items()})
        if self._reservations is not None:
            stats.update({f"reservation_{key}": value for key, value in self._reservations.stats().items()})
        if self._publications is not None:
            stats.update({f"publication_{key}": value for key, value in self._publications.stats().items()})
        return stats

    def bind_kv_cache_manager(self, manager: Any) -> None:
        if self._publications is not None:
            self._publications.bind_kv_cache_manager(manager)

    def shutdown(self) -> None:
        if self._reservations is not None:
            self._reservations.close()

    @staticmethod
    def _send_commit(record: CommitRecord) -> None:
        path = make_zmq_path("tcp", record.host, record.port)
        # The bounded sender owns socket use; no per-peer socket cache can grow.
        with zmq.Context() as context:
            sock = make_zmq_socket(context, path, zmq.REQ, bind=False)
            try:
                sock.setsockopt(zmq.SNDTIMEO, 1000)
                sock.setsockopt(zmq.RCVTIMEO, 1000)
                ensure_zmq_send(
                    sock,
                    msgspec.msgpack.encode(
                        (COMMIT_MSG, record.epoch, record.attempt_id, record.digest, record.committed)
                    ),
                    path,
                )
                if ensure_zmq_recv(sock, path) != ACK_MSG:
                    raise RuntimeError("PD source rejected completion")
            finally:
                sock.close(linger=0)

    def _queue_commit(self, request_id: str, remote: tuple[str, int, str], committed: bool) -> None:
        assert self._commit_sender is not None
        manifest = self._recv_manifests.pop(request_id)
        self._commit_sender.enqueue(
            request_id,
            CommitRecord(remote[0], remote[1], manifest.engine_epoch, manifest.attempt_id, manifest.digest, committed),
        )

    def _validate_receive_publication(
        self, request: "Request", params: dict[str, Any], prefix_tokens: int, ticket: ReservationTicket | None = None
    ) -> ContinuationManifest:
        manifest = ContinuationManifest.from_wire(params.get("continuation_manifest"), self.wan_pd)
        if manifest.transfer_plan_fingerprint != fingerprint(
            {
                key: params.get(key)
                for key in (
                    "remote_block_ids",
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_num_prompt_tokens",
                    "remote_host",
                    "remote_port",
                )
            },
            max_bytes=self.wan_pd.max_manifest_bytes,
        ):
            raise ValueError("PD physical transfer plan does not match publication")
        manifest.validate_prefix(self._identity(request), prefix_tokens=prefix_tokens, block_size=self.block_size)
        actual = self._state_prefill_token_count(request.num_prompt_tokens)
        if (
            manifest.state_position != actual
            or manifest.request_id != params.get("remote_request_id")
            or params.get("remote_num_prompt_tokens") != actual
        ):
            raise ValueError("PD state position or producer request mismatch")
        for name in ("remote_block_ids", "remote_engine_id", "remote_host", "remote_port"):
            if name not in params:
                raise ValueError(f"Missing PD field {name}")
        if manifest.reservation_fingerprint != (ticket.fingerprint if ticket is not None else ""):
            raise ValueError("PD publication belongs to another receiver reservation")
        if manifest.publication_mode == "layerwise" and not self.wan_pd.layerwise_publication:
            raise ValueError("Layerwise PD publication is disabled on D")
        return manifest

    def _reserve_receiver_first(self, request: "Request", prefix_tokens: int) -> tuple[int | None, bool]:
        params = request.kv_transfer_params
        assert params is not None
        count = max(self._state_prefill_token_count(request.num_prompt_tokens) - prefix_tokens, 0)
        if count == 0:
            # Full local hit needs no P work or asynchronous load. The proxy
            # simply forwards the local D response without waiting for a callback.
            params["do_remote_prefill"] = False
            if self._reservations is not None:
                self._reservations.cancel(request.request_id)
            if self._commit_sender is not None:
                self._commit_sender.cancel_unsubmitted(request.request_id)
            return 0, False
        try:
            if self._reservations is None:
                raise ValueError("Receiver-first PD is disabled on D")
            rendezvous = params.get("wan_pd_rendezvous_id")
            if not isinstance(rendezvous, str) or not 0 < len(rendezvous) <= 4096:
                raise ValueError("Receiver-first PD requires a bounded rendezvous ID")
            self._identity(request)
        except ValueError as exc:
            params["do_remote_prefill"] = False
            if self._reservations is not None:
                self._reservations.cancel(request.request_id)
            if self._commit_sender is not None:
                self._commit_sender.cancel_unsubmitted(request.request_id)
            logger.warning("Declining receiver-first PD for %s: %s", request.request_id, exc)
            return 0, False
        assert self._commit_sender is not None
        if not self._reservations.reserve(request.request_id):
            return None, False
        if not self._commit_sender.reserve(request.request_id):
            self._reservations.cancel(request.request_id)
            return None, False
        params["num_computed_tokens"] = prefix_tokens
        return count, True

    def _allocate_receiver_first(self, request: "Request", blocks: "KVCacheBlocks", external: int) -> None:
        assert self._reservations is not None and self._commit_sender is not None
        params = request.kv_transfer_params
        assert params is not None
        local, full = blocks.get_unhashed_block_ids_all_groups(), blocks.get_block_ids()
        try:
            ticket = self._reservations.submit(
                request.request_id,
                rendezvous_id=params["wan_pd_rendezvous_id"],
                identity=self._identity(request),
                state_position=self._state_prefill_token_count(request.num_prompt_tokens),
                prefix_tokens=params["num_computed_tokens"],
                allocation_fingerprint=fingerprint({"local": local, "full": full}),
            )
            self._awaiting_reservations[request.request_id] = (request, local, full, external, ticket)
        except ValueError as exc:
            self._reservations.cancel(request.request_id)
            self._commit_sender.cancel_unsubmitted(request.request_id)
            self._reservation_failures[request.request_id] = local
            logger.warning("PD receiver allocation publication failed for %s: %s", request.request_id, exc)
        params["do_remote_prefill"] = False

    def _drain_receiver_reservations(self) -> None:
        if self._reservations is None:
            return
        assert self._commit_sender is not None
        for result in self._reservations.poll_completed():
            allocation = self._awaiting_reservations.pop(result.request_id, None)
            if allocation is None:
                continue
            request, local, full, external, ticket = allocation
            try:
                if result.error is not None:
                    raise ValueError(result.error)
                if result.ticket != ticket or result.transfer_params is None:
                    raise ValueError("PD callback does not match the live receiver allocation")
                remote = result.transfer_params
                manifest = self._validate_receive_publication(request, remote, ticket.prefix_tokens, ticket)
                assert request.kv_transfer_params is not None
                request.kv_transfer_params.update(remote)
                request.kv_transfer_params["do_remote_prefill"] = False
                request.kv_transfer_params["num_computed_tokens"] = ticket.prefix_tokens
                self._recv_manifests[request.request_id] = manifest
                self._reqs_need_recv[request.request_id] = (request, local, full, external)
                self._reqs_recv_info[request.request_id] = (
                    remote["remote_host"],
                    remote["remote_port"],
                    remote["remote_request_id"],
                )
            except ValueError as exc:
                self._commit_sender.cancel_unsubmitted(request.request_id)
                self._reservations.retire(request.request_id)
                self._reservation_failures[request.request_id] = local
                logger.warning("PD reservation failed for %s: %s", request.request_id, exc)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        """Return prompt tokens that will be loaded from a remote producer."""
        params = request.kv_transfer_params
        if request.request_id in self._prepared and params and params.get("do_remote_decode"):
            # Re-admission of a prepared request follows preemption. Its old
            # source pages stay pinned, but a replay must not be advertised as
            # the same immutable attempt or wait forever on its ledger entry.
            self._abort_layerwise_replay(request.request_id)
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if self.wan_pd.enabled and params and params.get("do_remote_decode"):
            assert self._source_ledger is not None
            if not self._source_ledger.reserve(request.request_id):
                return None, False
            self._producer_requests[request.request_id] = request
        if params is not None and params.get("do_remote_prefill"):
            if params.get("wan_pd_receiver_first"):
                return self._reserve_receiver_first(request, num_computed_tokens)
            actual = self._state_prefill_token_count(request.num_prompt_tokens)
            if self.wan_pd.enabled:
                assert self._commit_sender is not None
                if not self._commit_sender.reserve(request.request_id):
                    return None, False
                try:
                    manifest = self._validate_receive_publication(request, params, num_computed_tokens)
                except ValueError:
                    # Keep this failure request-local; never advertise invalid external tokens.
                    self._commit_sender.cancel_unsubmitted(request.request_id)
                    params["do_remote_prefill"] = False
                    logger.exception("Rejected PD continuation for %s; recomputing locally", request.request_id)
                    return 0, False
                self._recv_manifests[request.request_id] = manifest
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)

        if params is None or not params.get("do_remote_prefill"):
            return

        if self.wan_pd.receiver_first and params.get("wan_pd_receiver_first"):
            if num_external_tokens > 0:
                self._allocate_receiver_first(request, blocks, num_external_tokens)
            else:
                # Native scheduling can shorten an advertised external hit to
                # zero after admission (local tail, replay or cache policy).
                assert self._reservations is not None and self._commit_sender is not None
                self._reservations.cancel(request.request_id)
                self._commit_sender.cancel_unsubmitted(request.request_id)
                params["do_remote_prefill"] = False
            return

        if params.get("remote_block_ids"):
            required_remote_fields = (
                "remote_engine_id",
                "remote_host",
                "remote_port",
                "remote_request_id",
                "remote_num_prompt_tokens",
            )
            if all(field in params for field in required_remote_fields):
                remote = (
                    params["remote_host"],
                    params["remote_port"],
                    params["remote_request_id"],
                )
                if num_external_tokens > 0:
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        blocks.get_unhashed_block_ids_all_groups(),
                        blocks.get_block_ids(),
                        num_external_tokens,
                    )
                    self._reqs_recv_info[request.request_id] = remote
                else:
                    if self.wan_pd.enabled:
                        self._queue_commit(request.request_id, remote, True)
                    else:
                        if self._recving_thread is None:
                            raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                        self._recving_thread.add_request(*remote)
            else:
                logger.warning("Got invalid KVTransferParams. params=%s.", params)
        else:
            assert num_external_tokens == 0
            if self._commit_sender is not None:
                self._commit_sender.cancel_unsubmitted(request.request_id)
                self._recv_manifests.pop(request.request_id, None)

        # Only trigger one transfer for a request.
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        self._drain_receiver_reservations()
        self._prepare_layerwise_publications(scheduler_output, meta)
        meta.failed_recv, self._reservation_failures = self._reservation_failures, {}
        meta.producer_finished, self._producer_finished = self._producer_finished, set()

        for (
            req_id,
            (req, block_ids, full_block_ids, num_external_tokens),
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
                local_num_prompt_tokens=req.num_prompt_tokens,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        self._reqs_need_recv.clear()
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()
        return meta

    def _source_transfer_params(self, request: "Request", block_ids: BlockIds) -> dict[str, Any]:
        params = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": self._get_transfer_block_ids(block_ids, request.num_prompt_tokens),
            "remote_num_prompt_tokens": request.num_prompt_tokens,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
        }
        if request.output_token_ids:
            params["last_token_id"] = request.output_token_ids[-1]
        return params

    def _prepare_layerwise_publications(self, output: SchedulerOutput, metadata: MooncakeConnectorMetadata) -> None:
        """Publish only the final immutable attention step, with real pool pins.

        Prefix/chunk scheduling stays native. Recursive state, speculation,
        graph replay, CP and asynchronous producer scheduling keep the final
        publication path until their freeze/copy contracts are established.
        """
        if self._publications is None or self._sending_thread is None:
            return
        for request_id in getattr(output, "preempted_req_ids", ()) or ():
            self._abort_layerwise_replay(request_id)
        config = self.vllm_config
        scheduler_config = getattr(config, "scheduler_config", None)
        if (
            not getattr(config.model_config, "enforce_eager", False)
            or self._sending_thread.use_kv_pp
            or getattr(scheduler_config, "async_scheduling", False)
            or config.speculative_config is not None
            or getattr(config.parallel_config, "enable_dbo", False)
            or getattr(config.parallel_config, "ubatch_size", 1) > 1
            or self.pcp_size != 1
            or self.dcp_size != 1
        ):
            return
        worker_layers: dict[str, list[str]] = {}
        for pp_rank, pp in self._sending_thread.metadata_by_pp_rank.items():
            if any(kind != "full_attention" for kind in pp.state_kinds):
                return
            for pcp_rank, pcp in pp.metadata_by_pcp_rank.items():
                for tp_rank, tp in pcp.metadata_by_tp_rank.items():
                    if tp.layer_indices:
                        worker_layers[f"{pp_rank}:{pcp_rank}:{tp_rank}"] = [pp.layer_names[i] for i in tp.layer_indices]
        block_state = getattr(output, "kv_connector_block_state", None)
        if block_state is None:
            return
        for request_id, scheduled in output.num_scheduled_tokens.items():
            request = self._producer_requests.get(request_id)
            if request is None or request_id in self._prepared:
                continue
            params = request.kv_transfer_params or {}
            if (
                "wan_pd_reservation" not in params
                or request.max_tokens != 1
                or request.output_token_ids
                or request.num_computed_tokens + scheduled < request.num_prompt_tokens
            ):
                continue
            blocks = block_state.get_block_ids(request_id)
            if blocks is None or not any(blocks):
                continue
            try:
                ticket = ReservationTicket.from_wire(params["wan_pd_reservation"], self.wan_pd)
                transfer = self._source_transfer_params(request, blocks)
                manifest = replace(self._make_continuation_manifest(request, transfer), publication_mode="layerwise")
                transfer["continuation_manifest"] = manifest.to_wire(self.wan_pd)
                self._publications.prepare(
                    request, transfer, manifest, worker_layers, ticket_fingerprint=ticket.fingerprint
                )
            except ValueError as exc:
                # No plan is visible if preparation fails. Use the same final
                # publication path as unsupported model/execution variants.
                logger.warning("PD early publication unavailable for %s: %s", request_id, exc)
                continue
            self._prepared[request_id] = transfer
            metadata.producer_publications[request_id] = transfer

    def _abort_layerwise_replay(self, request_id: str) -> None:
        if request_id not in self._prepared:
            return
        assert self._publications is not None
        self._publications.abort(request_id)
        request = self._producer_requests.get(request_id)
        if request is not None and request.kv_transfer_params is not None:
            request.kv_transfer_params["do_remote_decode"] = False
            request.kv_transfer_params["_wan_pd_identity_error"] = "Prepared P request was preempted"

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Expose completed producer blocks for a remote READ transfer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished: request_status=%s, kv_transfer_params=%s",
            request.status,
            params,
        )

        self._producer_requests.pop(request.request_id, None)
        allocation = self._awaiting_reservations.pop(request.request_id, None)
        if self._reservations is not None and request.request_id not in self._reqs_recv_info:
            # Admission may have succeeded while native allocate_slots still
            # waits for capacity; such a request has no allocation entry yet.
            self._reservations.cancel(request.request_id)
        if allocation is not None:
            assert self._reservations is not None
            self._reservations.cancel(request.request_id)
            # No READ has been submitted. Report a safe failed receive from
            # every worker so native WAITING_FOR_REMOTE_KVS cancellation frees
            # its actual allocation through the normal completion path.
            self._reservation_failures[request.request_id] = allocation[1]

        prepared = self._prepared.get(request.request_id)
        if prepared is not None:
            assert self._publications is not None
            self._reqs_need_send[request.request_id] = time.time()
            final = self._source_transfer_params(request, block_ids)
            valid = (
                request.status == RequestStatus.FINISHED_LENGTH_CAPPED
                and params is not None
                and params.get("do_remote_decode", False)
                and all(
                    final[key] == prepared[key]
                    for key in ("remote_block_ids", "remote_num_prompt_tokens", "remote_request_id")
                )
            )
            if valid:
                self._publications.seal(request.request_id)
                return True, {**prepared, "last_token_id": request.output_token_ids[-1]}
            self._publications.abort(request.request_id)
            return True, {
                "do_remote_prefill": False,
                "do_remote_decode": False,
                "wan_pd_error": "P publication aborted",
            }

        if self.wan_pd.enabled and params and "_wan_pd_identity_error" in params:
            return False, {
                "do_remote_prefill": False,
                "do_remote_decode": False,
                "wan_pd_error": params["_wan_pd_identity_error"],
            }

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            if self._source_ledger is not None:
                self._source_ledger.cancel_unpublished(request.request_id)
            if self._commit_sender is not None and request.request_id not in self._reqs_recv_info:
                self._commit_sender.cancel_unsubmitted(request.request_id)
                self._recv_manifests.pop(request.request_id, None)
            return False, None

        prompt_len = request.num_prompt_tokens
        computed_block_ids = self._get_transfer_block_ids(block_ids, prompt_len)
        num_computed_blocks = sum(len(group_block_ids) for group_block_ids in computed_block_ids)
        delay_free_blocks = num_computed_blocks > 0
        if delay_free_blocks:
            logger.info(
                "Delaying free of %d blocks for request %s",
                num_computed_blocks,
                request.request_id,
            )
            delay_start_time = time.time()
            self._reqs_need_send[request.request_id] = delay_start_time
            if self._sending_thread is None:
                raise RuntimeError("Mooncake scheduler metadata has not been initialized")
            if not self.wan_pd.enabled:
                self._sending_thread.add_delayed_request(request.request_id, delay_start_time)

        transfer_params = self._source_transfer_params(request, block_ids)
        if self.wan_pd.enabled and delay_free_blocks:
            assert self._source_ledger is not None and self._sending_thread is not None
            try:
                manifest = self._make_continuation_manifest(request, transfer_params)
                transfer_params["continuation_manifest"] = manifest.to_wire(self.wan_pd)
            except ValueError as exc:
                # Nothing was published: declining this individual handoff can
                # safely return its source blocks without disturbing others.
                self._source_ledger.cancel_unpublished(request.request_id)
                self._reqs_need_send.pop(request.request_id, None)
                logger.warning("Declining PD publication for %s: %s", request.request_id, exc)
                return False, {"do_remote_prefill": False, "do_remote_decode": False, "wan_pd_error": str(exc)}
            if "wan_pd_input_fingerprint" in params:
                transfer_params["wan_pd_input_fingerprint"] = params["wan_pd_input_fingerprint"]
            if "wan_pd_input_identity" in params:
                transfer_params["wan_pd_input_identity"] = params["wan_pd_input_identity"]
            self._source_ledger.publish(manifest)
        elif self._source_ledger is not None:
            self._source_ledger.cancel_unpublished(request.request_id)
        return delay_free_blocks, transfer_params

    def _make_continuation_manifest(self, request: "Request", transfer_params: dict[str, Any]) -> ContinuationManifest:
        assert self._source_ledger is not None and self._sending_thread is not None
        params = request.kv_transfer_params
        assert params is not None
        prompt_len = request.num_prompt_tokens
        identity = RequestIdentity.from_wire(params["_wan_pd_original_identity"], self.wan_pd)
        ticket = (
            ReservationTicket.from_wire(params["wan_pd_reservation"], self.wan_pd)
            if "wan_pd_reservation" in params
            else None
        )
        components = []
        for pp_meta in self._sending_thread.metadata_by_pp_rank.values():
            for layer_index, layer_name in enumerate(pp_meta.layer_names):
                kind = pp_meta.state_kinds[layer_index]
                group = pp_meta.group_indices[layer_index]
                start = 0
                if kind == "sliding_attention":
                    windows = [getattr(spec, "sliding_window", prompt_len) for spec in self.group_unique_specs[group]]
                    start = max(0, prompt_len - max(windows))
                for tensor_index, (dtype, shape) in enumerate(
                    zip(pp_meta.state_dtypes[layer_index], pp_meta.block_shapes[layer_index])
                ):
                    components.append(
                        StateComponent(
                            f"{layer_name}:{tensor_index}",
                            layer_name,
                            group,
                            kind,
                            dtype,
                            tuple(shape),
                            start,
                            prompt_len,
                            prompt_len if kind in ("mamba", "circular") else None,
                        )
                    )
        manifest = ContinuationManifest(
            request_id=request.request_id,
            attempt_id=uuid.uuid4().hex,
            engine_epoch=self._source_ledger.engine_epoch,
            namespace=self._namespace,
            layout_fingerprint=self._sending_thread.layout_fingerprint,
            input_fingerprint=identity.input_fingerprint,
            prompt_tokens=identity.prompt_tokens,
            state_position=prompt_len,
            prefix_tokens=0,
            components=tuple(components),
            prefix_hashes=identity.prefix_hashes,
            prefix_reusable=identity.prefix_reusable,
            prefix_hash_block_size=identity.prefix_hash_block_size,
            transfer_plan_fingerprint=fingerprint(
                {
                    key: transfer_params[key]
                    for key in (
                        "remote_block_ids",
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_num_prompt_tokens",
                        "remote_host",
                        "remote_port",
                    )
                }
            ),
            reservation_fingerprint=ticket.fingerprint if ticket else "",
        )
        if ticket is not None:
            manifest.validate_prefix(ticket.identity, prefix_tokens=ticket.prefix_tokens, block_size=self.block_size)
        return manifest

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        # D side: this output has already aggregated completion from all
        # workers. Send one scheduler-to-scheduler ACK for the request.
        for req_id in connector_output.finished_recving or ():
            if self._reservations is not None:
                self._reservations.retire(req_id)
            remote = self._reqs_recv_info.pop(req_id, None)
            if remote is not None and self.wan_pd.enabled:
                self._queue_commit(req_id, remote, req_id not in (connector_output.failed_recving or ()))
            elif remote is not None:
                if self._recving_thread is None:
                    raise RuntimeError("Producer Mooncake scheduler received a receive-completion event")
                self._recving_thread.add_request(*remote)
                if req_id in connector_output.failed_recving:
                    logger.error(
                        "Mooncake D receive failed after worker aggregation: request_id=%s, "
                        "remote_request_id=%s, producer=%s:%s, failure_policy=%s; producer release notification queued",
                        req_id,
                        remote[2],
                        remote[0],
                        remote[1],
                        self.kv_transfer_config.kv_load_failure_policy,
                    )

        # P side: feed scheduler-received ACKs into vLLM's standard delayed
        # free path. Scheduler._update_from_kv_xfer_finished reads this same
        # KVConnectorOutput immediately after this hook returns.
        finished_sending = (
            self._sending_thread.get_and_clear_finished_requests() if self._sending_thread is not None else set()
        )
        if finished_sending:
            for req_id in finished_sending:
                self._reqs_need_send.pop(req_id, None)
                if self._publications is not None and req_id in self._prepared:
                    self._publications.retire(req_id)
                    self._prepared.pop(req_id)
                    self._producer_finished.add(req_id)
            if connector_output.finished_sending is None:
                connector_output.finished_sending = finished_sending
            else:
                connector_output.finished_sending.update(finished_sending)


__all__ = [
    "MooncakePullConnectorScheduler",
    "MooncakeSchedulerRecvingThread",
    "MooncakeSchedulerSendingThread",
]
