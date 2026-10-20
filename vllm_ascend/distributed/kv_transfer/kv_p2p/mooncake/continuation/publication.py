# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Early attention publication with actual scheduler-owned block references.

The caller admits only the final, eager, synchronous, non-speculative prefill
step of a one-token producer request. A reference prevents block reuse, not
mutation: recurrent/circular state is deliberately rejected here. Device events
are polled by the worker publisher; this catalog only accepts their readiness
messages and never touches a device or a BlockPool on its control thread.
"""

import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import Lock, get_ident
from typing import Any

from .config import WanPDConfig
from .manifest import ContinuationManifest, canonical_bytes, fingerprint

GET_PUBLICATION_MSG = b"wan_pd_get_publication"
GET_READY_MSG = b"wan_pd_get_ready"
UPDATE_READY_MSG = b"wan_pd_update_ready"

_PLAN_FIELDS = (
    "remote_block_ids",
    "remote_engine_id",
    "remote_request_id",
    "remote_num_prompt_tokens",
    "remote_host",
    "remote_port",
)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= 4096:
        raise ValueError(f"Invalid publication {label}")
    return value


@dataclass
class _Publication:
    ticket: str
    manifest: ContinuationManifest
    encoded_params: bytes
    workers: dict[str, frozenset[str]]
    pins: list[tuple[Any, Any]]
    ready: dict[str, set[str]] = field(default_factory=dict)
    finalized: set[str] = field(default_factory=set)
    scheduler_sealed: bool = False
    error: str | None = None
    committed: bool = False

    def sealed(self) -> bool:
        return (
            self.scheduler_sealed
            and self.finalized == self.workers.keys()
            and all(self.ready[worker] == layers for worker, layers in self.workers.items())
        )

    def status(self) -> str:
        return "FAILED" if self.error is not None else "SEALED" if self.sealed() else "PREPARED"


class SourcePublicationManager:
    """Bounded, immutable source plans, event readiness, and extra block pins.

    ``prepare``, ``bind_kv_cache_manager``, ``seal`` and ``retire`` run on the
    creating scheduler thread. Lookup/readiness/commit validation are safe on
    the scheduler control thread. A failed publication retains its references;
    a timeout is not permission to recycle an RDMA source.
    """

    def __init__(self, config: WanPDConfig, source_ledger: Any, kv_cache_manager: Any = None) -> None:
        self.config, self.source_ledger = config, source_ledger
        self._manager = kv_cache_manager
        self._owner = get_ident()
        self._lock = Lock()
        self._requests: dict[str, _Publication] = {}
        self._tickets: dict[str, str] = {}
        self._attempts: dict[str, str] = {}
        self._terminal: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def _scheduler_thread(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("Publication block ownership must be changed on the scheduler thread")

    def bind_kv_cache_manager(self, manager: Any) -> None:
        self._scheduler_thread()
        if self._requests and manager is not self._manager:
            raise RuntimeError("Cannot replace a cache manager with active publications")
        self._manager = manager

    def _blocks(self, request_id: str, block_ids: Any, explicit: Any) -> list[tuple[Any, Any]]:
        if explicit is None:
            if self._manager is None:
                raise RuntimeError("Early publication requires a bound KV cache manager")
            groups = self._manager.get_blocks(request_id).blocks
        else:
            groups = getattr(explicit, "blocks", explicit)
        # Explicit blocks may be a flat sequence for a single cache group.
        groups = tuple(groups)
        if groups and hasattr(groups[0], "block_id"):
            groups = (groups,)
        if len(groups) != len(block_ids):
            raise ValueError("Publication block groups do not match the physical transfer plan")
        pins: dict[tuple[int, int], tuple[Any, Any]] = {}
        for group, needed in zip(groups, block_ids):
            by_id = {block.block_id: block for block in group}
            for block_id in needed:
                if type(block_id) is not int or block_id <= 0 or block_id not in by_id:
                    raise ValueError("Publication source block is missing or null")
                block = by_id[block_id]
                if getattr(block, "is_null", False) or block.ref_cnt <= 0:
                    raise ValueError("Publication source block has no live request reference")
                pool = getattr(block, "pool", None)
                if pool is None and self._manager is not None:
                    pool = self._manager.block_pool
                if pool is None:
                    raise ValueError("Publication source block has no owning BlockPool")
                key = (id(pool), block_id)
                previous = pins.setdefault(key, (pool, block))
                if previous[1] is not block:
                    raise ValueError("Publication block ID aliases different block objects")
        if not pins:
            raise ValueError("Early publication requires nonempty source blocks")
        return list(pins.values())

    def prepare(
        self,
        request: Any,
        transfer_params: dict[str, Any],
        manifest: ContinuationManifest,
        worker_keys: Mapping[str, Iterable[str]] | Iterable[str],
        *,
        ticket_fingerprint: str,
        blocks: Any = None,
    ) -> dict[str, Any]:
        """Pin the exact source plan before making any layer discoverable.

        ``worker_keys`` maps ``pp:pcp:tp`` to the registered layer names owned by
        that worker. A simple iterable means every listed worker owns all layers.
        Duplicate preparation is idempotent only for the identical plan/topology.
        """
        self._scheduler_thread()
        request_id = request if isinstance(request, str) else request.request_id
        _text(request_id, "request ID")
        _text(ticket_fingerprint, "ticket fingerprint")
        if manifest.request_id != request_id:
            raise ValueError("Publication request does not match its manifest")
        if not manifest.components or any(component.kind != "full_attention" for component in manifest.components):
            raise ValueError("Early publication supports only immutable full_attention state")
        layers = frozenset(component.layer_name for component in manifest.components)
        if isinstance(worker_keys, Mapping):
            workers = {key: frozenset(owned) for key, owned in worker_keys.items()}
        else:
            workers = {key: layers for key in worker_keys}
        if not workers or len(workers) > self.config.max_manifest_components:
            raise ValueError("Invalid or excessive publication worker count")
        for worker, owned in workers.items():
            _text(worker, "worker key")
            if not owned or not owned <= layers:
                raise ValueError("Publication worker has unknown or empty layer ownership")
        if frozenset().union(*workers.values()) != layers:
            raise ValueError("Publication topology does not own every manifest layer")
        # Bound replicated readiness metadata as well as its manifest.
        if sum(map(len, workers.values())) > self.config.max_manifest_ranges:
            raise ValueError("Publication readiness topology exceeds max_manifest_ranges")
        canonical_bytes(
            {
                "status": "PREPARED",
                "ready_layers": sorted(f"{worker}/{layer}" for worker, owned in workers.items() for layer in owned),
                "finalized_workers": sorted(workers),
                "error": None,
            },
            max_bytes=self.config.max_manifest_bytes,
        )
        params = dict(transfer_params)
        params["continuation_manifest"] = manifest.to_wire(self.config)
        plan = {key: params.get(key) for key in _PLAN_FIELDS}
        if manifest.transfer_plan_fingerprint != fingerprint(plan, max_bytes=self.config.max_manifest_bytes):
            raise ValueError("Publication transfer plan does not match manifest")
        if plan["remote_request_id"] != request_id or plan["remote_num_prompt_tokens"] != manifest.state_position:
            raise ValueError("Publication plan has a different request or token boundary")
        encoded = canonical_bytes(params, max_bytes=self.config.max_manifest_bytes)
        canonical_bytes(
            {"status": "PREPARED", "transfer_params": params, "error": None},
            max_bytes=self.config.max_manifest_bytes,
        )
        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None:
                if (previous.ticket, previous.manifest.digest, previous.encoded_params, previous.workers) != (
                    ticket_fingerprint,
                    manifest.digest,
                    encoded,
                    workers,
                ):
                    raise ValueError("Publication request already has a different immutable plan")
                return self._publication_wire(previous)
            if len(self._requests) >= self.config.max_retained_source_requests:
                raise ValueError("Publication source capacity exhausted")
            if ticket_fingerprint in self._tickets or manifest.attempt_id in self._attempts:
                raise ValueError("Publication ticket or attempt ID is already owned")
            if manifest.attempt_id in self._terminal:
                raise ValueError("Publication attempt ID was already retired")
            pins = self._blocks(request_id, plan["remote_block_ids"], blocks)
            pinned: list[tuple[Any, Any]] = []
            try:
                for pool, block in pins:
                    before = block.ref_cnt
                    try:
                        pool.touch([block])
                    finally:
                        # Native metrics callbacks may fail after incrementing.
                        if block.ref_cnt > before:
                            pinned.append((pool, block))
                if not self.source_ledger.reserve(request_id):
                    raise ValueError("Publication source ledger has no reservation capacity")
                self.source_ledger.publish(manifest)
            except Exception:
                # Nothing was made discoverable; roll back only our references.
                for pool, block in reversed(pinned):
                    pool.free_blocks([block])
                raise
            entry = _Publication(ticket_fingerprint, manifest, encoded, workers, pinned)
            entry.ready = {worker: set() for worker in workers}
            self._requests[request_id] = entry
            self._tickets[ticket_fingerprint] = request_id
            self._attempts[manifest.attempt_id] = request_id
            return self._publication_wire(entry)

    @staticmethod
    def _publication_wire(entry: _Publication) -> dict[str, Any]:
        # Deserialize on return: callers cannot mutate the stored physical plan.
        return {"status": entry.status(), "transfer_params": json.loads(entry.encoded_params), "error": entry.error}

    def get_publication(self, ticket_fingerprint: str) -> dict[str, Any] | None:
        _text(ticket_fingerprint, "ticket fingerprint")
        with self._lock:
            request_id = self._tickets.get(ticket_fingerprint)
            return self._publication_wire(self._requests[request_id]) if request_id is not None else None

    def _entry(self, epoch: str, attempt_id: str, digest: str) -> _Publication:
        _text(epoch, "engine epoch")
        _text(attempt_id, "attempt ID")
        _text(digest, "manifest digest")
        request_id = self._attempts.get(attempt_id)
        entry = self._requests.get(request_id) if request_id is not None else None
        if entry is None or (entry.manifest.engine_epoch, entry.manifest.digest) != (epoch, digest):
            raise ValueError("Unknown or stale publication identity")
        return entry

    def get_ready(self, epoch: str, attempt_id: str, digest: str) -> dict[str, Any]:
        with self._lock:
            entry = self._entry(epoch, attempt_id, digest)
            return {
                "status": entry.status(),
                "ready_layers": sorted(
                    f"{worker}/{layer}" for worker, layers in entry.ready.items() for layer in layers
                ),
                "finalized_workers": sorted(entry.finalized),
                "error": entry.error,
            }

    def update_ready(
        self,
        epoch: str,
        attempt_id: str,
        digest: str,
        worker_key: str,
        ready_layers: Iterable[str],
        finalized: bool,
    ) -> None:
        _text(worker_key, "worker key")
        if type(finalized) is not bool or not isinstance(ready_layers, (list, tuple, set, frozenset)):
            raise ValueError("Invalid publication readiness update")
        if len(ready_layers) > self.config.max_manifest_components:
            raise ValueError("Publication readiness update is too large")
        layers = frozenset(_text(layer, "ready layer") for layer in ready_layers)
        with self._lock:
            entry = self._entry(epoch, attempt_id, digest)
            if entry.error is not None:
                raise ValueError("Publication has failed")
            owned = entry.workers.get(worker_key)
            if owned is None or not layers <= owned:
                raise ValueError("Publication worker reported layers it does not own")
            ready = entry.ready[worker_key] | layers
            if finalized and ready != owned:
                raise ValueError("Publication worker finalized before all owned layers were ready")
            entry.ready[worker_key] = ready
            if finalized:
                entry.finalized.add(worker_key)

    def seal(self, request_id: str) -> None:
        """Confirm P completion without upgrading an earlier D failure.

        A receiver may fail while P is still executing. Recording P completion
        must not turn that request-local failure into a scheduler exception;
        the failed status and quarantined references remain unchanged.
        """
        self._scheduler_thread()
        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None:
                raise ValueError("Cannot seal an unknown publication")
            entry.scheduler_sealed = True

    def abort(self, request_id: str, error: str = "Producer continuation aborted") -> bool:
        """Invalidate further READ admission while retaining every source pin."""
        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None:
                return False
            if entry.committed:
                raise ValueError("Cannot abort an already committed publication")
            entry.error = str(error)[:4096]
            manifest = entry.manifest
            self.source_ledger.finish(manifest.engine_epoch, manifest.attempt_id, manifest.digest, False)
            return True

    def validate_commit(self, epoch: str, attempt_id: str, digest: str, committed: bool) -> bool:
        """Validate early attempts before the existing SourceLedger consumes ACK.

        Return False for non-catalog attempts (the ordinary final-publication
        path), which the SourceLedger must still validate. Successful replay
        remains idempotent after retirement via a bounded terminal history.
        """
        if type(committed) is not bool:
            raise ValueError("Invalid publication commit outcome")
        _text(epoch, "engine epoch")
        _text(attempt_id, "attempt ID")
        _text(digest, "manifest digest")
        with self._lock:
            if attempt_id in self._terminal:
                if self._terminal[attempt_id] != (epoch, digest) or not committed:
                    raise ValueError("Conflicting retired publication outcome")
                return True
            if attempt_id not in self._attempts:
                return False
            entry = self._entry(epoch, attempt_id, digest)
            if committed:
                if entry.error is not None or not entry.sealed():
                    raise ValueError("Publication is not sealed for a successful commit")
                entry.committed = True
            elif not entry.committed:
                entry.error = entry.error or "Consumer continuation failed"
            return True

    def retire(self, request_id: str) -> bool:
        """Release extra references on the scheduler thread after validated ACK."""
        self._scheduler_thread()
        with self._lock:
            entry = self._requests.get(request_id)
            if entry is None:
                return False
            if not entry.committed:
                raise ValueError("Publication cannot retire before a successful commit")
            # If freeing raises, propagate as an engine failure, never silently
            # report a safely retired plan with potentially live source refs.
            while entry.pins:
                pool, block = entry.pins[-1]
                pool.free_blocks([block])
                entry.pins.pop()
            del self._requests[request_id]
            del self._tickets[entry.ticket]
            del self._attempts[entry.manifest.attempt_id]
            self._terminal[entry.manifest.attempt_id] = (entry.manifest.engine_epoch, entry.manifest.digest)
            while len(self._terminal) > self.config.max_terminal_requests:
                self._terminal.popitem(last=False)
            return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "publications": len(self._requests),
                "source_block_pins": sum(len(entry.pins) for entry in self._requests.values()),
                "sealed_publications": sum(entry.status() == "SEALED" for entry in self._requests.values()),
                "failed_publications": sum(entry.error is not None for entry in self._requests.values()),
            }
