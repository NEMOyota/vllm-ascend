# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-only control-plane checks; hardware completion is the caller's contract."""

import pytest

from vllm_ascend.distributed.pp_transport.egress import EgressArbiter, EgressError, EgressLimits


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_bounded_retry_safe_admission_and_unposted_cancel():
    arbiter = EgressArbiter(EgressLimits(max_bytes=8, max_fragments=2, max_flows=3))
    a, b, c = [arbiter.register(name) for name in ("pp-a", "pp-b", "pd-c")]
    with pytest.raises(EgressError, match="capacity"):
        arbiter.register("too-many")
    first = arbiter.request(a, 0, 3)
    second = arbiter.request(b, 0, 3)
    assert arbiter.request(a, 0, 3) == first
    assert arbiter.stats()["inflight_bytes"] == 6
    assert arbiter.request(c, 0, 4)["status"] == "pending"
    with pytest.raises(EgressError, match="only one pending"):
        arbiter.request(c, 1, 1)
    with pytest.raises(EgressError, match="Retry changed"):
        arbiter.request(c, 0, 2)
    arbiter.complete(a, first["lease_id"])
    assert arbiter.request(c, 0, 4)["status"] == "granted"
    assert arbiter.stats()["inflight_bytes"] == 7
    assert arbiter.request(a, 0, 3)["status"] == "retired"
    assert arbiter.complete(a, first["lease_id"])["status"] == "retired"
    with pytest.raises(EgressError, match="never posted"):
        arbiter.cancel(b, 0)
    arbiter.cancel(b, 0, unposted=True)
    assert arbiter.complete(b, second["lease_id"])["status"] == "retired"
    assert arbiter.stats()["inflight_fragments"] == 1


def test_latency_preference_is_bounded_and_bulk_fairness_charges_bytes():
    arbiter = EgressArbiter(EgressLimits(max_bytes=8, max_fragments=1, latency_burst_bytes=2))
    blocker, bulk = [arbiter.register(name) for name in ("blocker", "bulk")]
    latency = [arbiter.register(f"latency-{index}") for index in range(3)]
    held = arbiter.request(blocker, 0, 8)
    assert arbiter.request(bulk, 0, 4)["status"] == "pending"
    for session in latency:
        assert arbiter.request(session, 0, 1, "latency")["status"] == "pending"
    arbiter.complete(blocker, held["lease_id"])
    for session in latency[:2]:
        grant = arbiter.request(session, 0, 1, "latency")
        assert grant["status"] == "granted"
        arbiter.complete(session, grant["lease_id"])
    assert arbiter.request(bulk, 0, 4)["status"] == "granted"
    assert arbiter.request(latency[2], 0, 1, "latency")["status"] == "pending"

    arbiter = EgressArbiter(EgressLimits(max_bytes=8, max_fragments=1))
    blocker, large, small = [arbiter.register(name) for name in ("blocker", "large", "small")]
    held = arbiter.request(blocker, 0, 8)
    arbiter.request(large, 0, 4)
    arbiter.request(small, 0, 1)
    arbiter.complete(blocker, held["lease_id"])
    # Several small fragments receive comparable bytes, not one equal-sized
    # scheduling turn; the older large request wins the equal finish-tag tie.
    for request_id in range(3):
        grant = arbiter.request(small, request_id, 1)
        assert grant["status"] == "granted"
        arbiter.request(small, request_id + 1, 1)
        arbiter.complete(small, grant["lease_id"])
    assert arbiter.request(large, 0, 4)["status"] == "granted"
    assert arbiter.request(small, 3, 1)["status"] == "pending"


def test_shared_rate_tokens_are_spent_on_grants_and_pending_does_not_expire():
    clock = _Clock()
    arbiter = EgressArbiter(
        EgressLimits(max_bytes=10, max_fragments=1, rate_bytes_per_s=2, burst_bytes=10, lease_timeout_s=1),
        clock=clock,
    )
    a, b = [arbiter.register(name) for name in ("pp", "pd")]
    grant = arbiter.request(a, 0, 10)
    arbiter.complete(a, grant["lease_id"])
    assert arbiter.stats()["rate_tokens"] == 0
    assert arbiter.request(b, 0, 4)["status"] == "pending"
    clock.now = 1.5  # A pending request has outlived lease_timeout_s safely.
    assert arbiter.request(b, 0, 4)["status"] == "pending"
    clock.now = 2
    assert arbiter.request(b, 0, 4)["status"] == "granted"
    arbiter.cancel(b, 0, unposted=True)
    assert arbiter.stats()["rate_tokens"] == 0
    assert arbiter.request(a, 1, 1)["status"] == "pending"


def test_disconnect_and_expired_grants_fail_closed_without_refunding():
    clock = _Clock()
    arbiter = EgressArbiter(EgressLimits(max_bytes=4, max_fragments=1, lease_timeout_s=2), clock=clock)
    a, b = [arbiter.register(name) for name in ("active", "waiting")]
    grant = arbiter.request(a, 0, 4)
    arbiter.request(b, 0, 1)
    arbiter.disconnect(b)  # Pending-only work can be safely removed.
    assert arbiter.stats()["pending"] == 0
    assert not arbiter.stats()["failed"]
    arbiter.disconnect(a)
    assert arbiter.stats()["inflight_bytes"] == 4
    with pytest.raises(EgressError, match="disconnected"):
        arbiter.complete(a, grant["lease_id"])
    with pytest.raises(EgressError, match="disconnected"):
        arbiter.register("new")

    arbiter = EgressArbiter(EgressLimits(max_bytes=4, max_fragments=1, lease_timeout_s=2), clock=clock)
    a = arbiter.register("active")
    arbiter.request(a, 0, 4)
    clock.now = 2
    with pytest.raises(EgressError, match="expired"):
        arbiter.tick()
    assert arbiter.stats()["inflight_bytes"] == 4
    assert arbiter.stats()["inflight_fragments"] == 1
