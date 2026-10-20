# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU receiver-first admission, callback bounds and stale publication checks."""

import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from time import monotonic

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.config import WanPDConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import RequestIdentity, fingerprint
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.reservation import (
    BoundedReservationDispatcher,
    ReservationTicket,
)


def _config(**values):
    return WanPDConfig(
        **{
            "enabled": True,
            "receiver_first": True,
            "max_pending_requests": 2,
            "max_active_requests": 1,
            "reservation_workers": 2,
            "timeout_s": 10,
            **values,
        }
    )


def _submit(dispatcher, request_id="r", **values):
    assert dispatcher.reserve(request_id)
    return dispatcher.submit(
        request_id,
        **{
            "rendezvous_id": "proxy-" + request_id,
            "identity": RequestIdentity("weights-v1", "input-digest", 8),
            "state_position": 7,
            "prefix_tokens": 0,
            "allocation_fingerprint": fingerprint([[1, 3, 4], [5]]),
            **values,
        },
    )


def _wait(predicate):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        result = predicate()
        if result:
            return result
        Event().wait(0.002)
    pytest.fail("Receiver callback did not reach expected state")


def test_ticket_binds_allocation_identity_and_strict_bounded_wire():
    ticket = ReservationTicket(
        "d-epoch", "attempt-a", "r", "proxy-r", RequestIdentity("weights", "input", 8), 7, 0, fingerprint([[1, 2]])
    )
    assert ReservationTicket.from_wire(ticket.to_wire()) == ticket
    for changed in (
        replace(ticket, receiver_epoch="other"),
        replace(ticket, reservation_id="attempt-b"),
        replace(ticket, allocation_fingerprint=fingerprint([[2, 3]])),
        replace(ticket, identity=replace(ticket.identity, input_fingerprint="other")),
        replace(ticket, prefix_tokens=2),
    ):
        assert changed.fingerprint != ticket.fingerprint
    for invalid in (
        {**ticket.to_wire(), "state_position": 9},
        {**ticket.to_wire(), "prefix_tokens": True},
        {**ticket.to_wire(), "protocol_version": True},
        {**ticket.to_wire(), "callback_url": "http://untrusted/"},
        {**ticket.to_wire(), "reservation_id": "x" * 4097},
    ):
        with pytest.raises(ValueError):
            ReservationTicket.from_wire(invalid)
    with pytest.raises(ValueError, match="max_manifest_bytes"):
        ReservationTicket.from_wire(ticket.to_wire(), _config(max_manifest_bytes=32))


def test_submit_once_and_hold_capacity_until_consumer_retires():
    received = []

    def callback(payload, timeout, limit):
        received.append(payload)
        assert 0 < timeout <= 10 and limit == 2 * 1024 * 1024
        return {"transfer_params": {"source": "p", "binding": fingerprint(payload["ticket"])}}

    dispatcher = BoundedReservationDispatcher(_config(max_pending_requests=1), "d", callback)
    dispatcher.start()
    try:
        ticket = _submit(dispatcher)
        assert _submit(dispatcher) == ticket
        with pytest.raises(ValueError, match="changed"):
            _submit(dispatcher, allocation_fingerprint=fingerprint([[9]]))
        result = _wait(dispatcher.poll_completed)[0]
        assert result.error is None and result.ticket == ticket
        assert result.transfer_params["binding"] == ticket.fingerprint
        assert len(received) == 1
        assert dispatcher.poll_completed() == []
        assert not dispatcher.reserve("another")
        dispatcher.retire("r")
        assert dispatcher.reserve("another")
    finally:
        dispatcher.close(timeout=1)


def test_active_and_pending_limits_apply_even_while_first_p_is_slow():
    started, release = Event(), Event()
    calls = []

    def callback(payload, _timeout, _limit):
        calls.append(payload["ticket"]["request_id"])
        started.set()
        assert release.wait(3)
        return {"transfer_params": {}}

    dispatcher = BoundedReservationDispatcher(_config(), "d", callback)
    dispatcher.start()
    try:
        _submit(dispatcher, "first")
        assert started.wait(1)
        _submit(dispatcher, "second")
        assert not dispatcher.reserve("third")
        assert dispatcher.stats() == {"reserved": 2, "active": 1, "queued": 1, "cancelled_active": 0}
        assert calls == ["first"]
        release.set()
        _wait(lambda: dispatcher.stats()["active"] == 0 and dispatcher.stats()["queued"] == 0)
        assert len(dispatcher.poll_completed()) == 2
        assert calls == ["first", "second"]
    finally:
        release.set()
        dispatcher.close(timeout=1)


def test_timeout_cancel_and_request_id_reuse_never_accept_late_publication():
    now = [0.0]
    started, release = Event(), Event()

    def callback(_payload, _timeout, _limit):
        started.set()
        assert release.wait(3)
        return {"transfer_params": {"late": True}}

    dispatcher = BoundedReservationDispatcher(_config(max_pending_requests=1), "d", callback, clock=lambda: now[0])
    dispatcher.start()
    try:
        old = _submit(dispatcher)
        assert started.wait(1)
        now[0] = 11
        result = dispatcher.poll_completed()[0]
        assert result.ticket == old and "timed out" in result.error
        dispatcher.retire("r")
        assert dispatcher.stats()["cancelled_active"] == 1
        assert not dispatcher.reserve("r")
        assert not dispatcher.reserve("other")
        release.set()
        _wait(lambda: dispatcher.stats()["reserved"] == 0)
        assert dispatcher.poll_completed() == []
        new = _submit(dispatcher)
        assert old.reservation_id != new.reservation_id
        assert _wait(dispatcher.poll_completed)[0].ticket == new
    finally:
        release.set()
        dispatcher.close(timeout=1)


def test_close_discards_running_callback_and_rejects_new_admission():
    started, release = Event(), Event()

    def callback(_payload, _timeout, _limit):
        started.set()
        assert release.wait(3)
        return {"transfer_params": {"late": True}}

    dispatcher = BoundedReservationDispatcher(_config(), "d", callback)
    dispatcher.start()
    try:
        _submit(dispatcher)
        assert started.wait(1)
        dispatcher.close()
        assert not dispatcher.reserve("new")
        release.set()
        _wait(lambda: dispatcher.stats()["reserved"] == 0)
        assert dispatcher.poll_completed() == []
    finally:
        release.set()
        dispatcher.close(timeout=1)


def test_queue_deadline_and_malformed_response_fail_without_unbounded_work():
    now = [0.0]
    calls = []

    def callback(payload, _timeout, limit):
        calls.append(payload)
        return {"transfer_params": {"oversized": "x" * limit}}

    dispatcher = BoundedReservationDispatcher(_config(max_manifest_bytes=1024), "d", callback, clock=lambda: now[0])
    try:
        _submit(dispatcher, "expired")
        now[0] = 11
        result = dispatcher.poll_completed()[0]
        assert "timed out" in result.error and not calls
        dispatcher.retire("expired")
        dispatcher.start()
        _submit(dispatcher, "oversized")
        result = _wait(dispatcher.poll_completed)[0]
        assert "max_manifest_bytes" in result.error and len(calls) == 1
        assert dispatcher.stats()["reserved"] == 1
    finally:
        dispatcher.close(timeout=1)


def test_http_uses_configured_endpoint_and_rejects_oversized_response():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            # The receiver must reject the declared size before buffering the body.
            self.send_response(200)
            self.send_header("Content-Length", "1048576")
            self.end_headers()

        def log_message(self, _format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    dispatcher = BoundedReservationDispatcher(
        _config(
            max_manifest_bytes=1024,
            reservation_callback_url=f"http://127.0.0.1:{server.server_port}/admin/callback",
        ),
        "d",
    )
    dispatcher.start()
    try:
        ticket = _submit(dispatcher)
        result = _wait(dispatcher.poll_completed)[0]
        assert "max_manifest_bytes" in result.error
        assert received == [("/admin/callback", {"ticket": ticket.to_wire()})]
    finally:
        dispatcher.close(timeout=1)
        server.shutdown()
        server.server_close()
        serving.join(timeout=1)
