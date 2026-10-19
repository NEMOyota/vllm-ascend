# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small real-TCP experiments for cross-process shared egress accounting."""

import multiprocessing as mp
import threading
import time

import pytest

from vllm_ascend.distributed.pp_transport.egress import EgressError, EgressLimits
from vllm_ascend.distributed.pp_transport.egress_service import EgressBudgetServer, EgressClient

_TIMEOUT_S = 10


@pytest.fixture
def budget_server():
    server = EgressBudgetServer(
        ("127.0.0.1", 0), EgressLimits(max_bytes=8, max_fragments=1, max_flows=8, lease_timeout_s=30)
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, ("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(_TIMEOUT_S)


def _contend(endpoint, rank, first_acquired, second_blocked):
    client = EgressClient(endpoint, f"worker-{rank}")
    try:
        if rank == 0:
            lease = client.try_acquire(8)
            assert lease is not None
            first_acquired.set()
            assert second_blocked.wait(_TIMEOUT_S)
        else:
            assert first_acquired.wait(_TIMEOUT_S)
            # Retry exactly the same pending request without duplicate charges.
            assert client.try_acquire(8) is None
            assert client.try_acquire(8) is None
            second_blocked.set()
            lease = None
            deadline = time.monotonic() + _TIMEOUT_S
            while lease is None and time.monotonic() < deadline:
                lease = client.try_acquire(8)
                if lease is None:
                    time.sleep(0.005)
            assert lease is not None
        lease.release()
    finally:
        client.close()


def test_two_processes_share_one_budget(budget_server):
    server, endpoint = budget_server
    context = mp.get_context("spawn")
    first_acquired, second_blocked = context.Event(), context.Event()
    processes = [
        context.Process(target=_contend, args=(endpoint, rank, first_acquired, second_blocked)) for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(_TIMEOUT_S)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(_TIMEOUT_S)
    stats = server.arbiter.stats()
    assert stats["inflight_bytes"] == stats["inflight_fragments"] == 0
    assert stats["granted_bytes"] == 16
    assert stats["granted_fragments"] == 2
    assert stats["peak_inflight_bytes"] == 8
    assert stats["peak_inflight_fragments"] == 1
    assert not stats["failed"]


def test_disconnect_with_unresolved_grant_poison_scope(budget_server):
    server, endpoint = budget_server
    source = EgressClient(endpoint, "uncertain-source")
    other = EgressClient(endpoint, "other-source")
    try:
        assert source.try_acquire(8) is not None
        assert other.try_acquire(8) is None
        # There is no completion proof; losing this control connection must not
        # make possibly active device memory/bandwidth ownership disappear.
        source.fail("control connection lost after submission")
        deadline = time.monotonic() + _TIMEOUT_S
        while not server.arbiter.stats()["failed"] and time.monotonic() < deadline:
            time.sleep(0.005)
        stats = server.arbiter.stats()
        assert stats["failed"]
        assert stats["inflight_bytes"] == 8
        assert stats["inflight_fragments"] == 1
        with pytest.raises(EgressError):
            other.try_acquire(8)
    finally:
        source.close()
        other.close()


def test_cancel_unposted_and_disconnect_pending_are_safe(budget_server):
    server, endpoint = budget_server
    source = EgressClient(endpoint, "source")
    waiter = EgressClient(endpoint, "pending-only")
    try:
        first = source.try_acquire(8)
        assert first is not None
        assert waiter.try_acquire(8) is None
        waiter.cancel_pending()
        assert waiter.try_acquire(8) is None
        first.safe_cancel_before_submit()
        # Completion automatically grants waiter's pending request server-side,
        # but waiter has never received a lease or posted it to a data backend.
        assert server.arbiter.stats()["inflight_bytes"] == 8
        assert server.arbiter.stats()["pending"] == 0
        waiter.close()
        next_lease = source.try_acquire(8, traffic="latency")
        assert next_lease is not None
        next_lease.cancel()
        stats = source.stats()
        assert not stats["failed"]
        assert stats["inflight_bytes"] == stats["inflight_fragments"] == 0
    finally:
        source.close()
        waiter.close()
    assert not server.arbiter.stats()["failed"]


def test_idle_session_survives_message_io_timeout(budget_server):
    server, endpoint = budget_server
    server.socket_timeout_s = 0.1
    client = EgressClient(endpoint, "idle-worker")
    try:
        time.sleep(3 * server.socket_timeout_s)
        lease = client.try_acquire(8)
        assert lease is not None
        lease.release()
        assert client.stats()["inflight_bytes"] == 0
    finally:
        client.close()
