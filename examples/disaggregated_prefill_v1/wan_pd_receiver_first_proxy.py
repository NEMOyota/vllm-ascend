# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Optional receiver-first router for the WAN Mooncake V2/Pull connector.

Start D with ``wan_pd.receiver_first=true`` and ``reservation_callback_url``
pointing to this server's ``/v1/wan-pd/reservation`` endpoint. A D allocation
callback starts P; its final publication is returned to the waiting D connector.
This example supports one prompt/choice per request and one P/D backend pair.
Run one proxy process: its bounded rendezvous table is intentionally local.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import msgspec
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import ContinuationManifest
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.publication import GET_PUBLICATION_MSG
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.reservation import ReservationTicket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyConfig:
    prefill_url: str
    decode_url: str
    max_pending: int = 128
    timeout_s: float = 300.0
    max_body_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 16 * 1024 * 1024
    layerwise_publication: bool = False
    producer_control_address: str | None = None

    def __post_init__(self) -> None:
        for name in ("prefill_url", "decode_url"):
            parsed = urlsplit(getattr(self, name))
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
                raise ValueError(f"{name} must be an HTTP(S) backend base URL")
        for name in ("max_pending", "max_body_bytes", "max_response_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if self.layerwise_publication:
            parsed = urlsplit("tcp://" + (self.producer_control_address or ""))
            if not parsed.hostname or not parsed.port or parsed.path or parsed.query or parsed.fragment:
                raise ValueError("layerwise_publication requires --producer-control-address HOST:PORT")


@dataclass
class _Pending:
    payload: dict[str, Any]
    path: str
    headers: dict[str, str]
    deadline: float
    ticket_fingerprint: str | None = None
    prefill_task: asyncio.Task[dict[str, Any]] | None = None
    publication: asyncio.Task[dict[str, Any]] | None = None
    callback_waiters: int = 0


async def _request_json(request: Request, limit: int) -> dict[str, Any]:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, "Request exceeds the configured byte limit")
        body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Expected a JSON object") from None
    if not isinstance(value, dict):
        raise HTTPException(400, "Expected a JSON object")
    return value


async def _response_json(response: httpx.Response, limit: int) -> dict[str, Any]:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > limit:
            raise HTTPException(502, "P publication exceeds the configured byte limit")
        body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(502, "P returned an invalid JSON publication") from None
    if not isinstance(value, dict):
        raise HTTPException(502, "P returned an invalid JSON publication")
    return value


def _validate_single_request(payload: dict[str, Any], path: str) -> None:
    if payload.get("n", 1) != 1 or payload.get("best_of", 1) not in (None, 1):
        raise HTTPException(400, "Receiver-first example supports one choice per request")
    if path == "/v1/completions":
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) and not (
            isinstance(prompt, list) and prompt and all(type(token) is int for token in prompt)
        ):
            raise HTTPException(
                400, "Receiver-first example accepts one text or token-ID prompt; batches are unsupported"
            )
    params = payload.get("kv_transfer_params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(400, "kv_transfer_params must be an object")


def _identity_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("kv_transfer_params") or {}
    # Forward explicit input identity, but create all routing/control fields
    # here instead of accepting arbitrary backend addresses from the client.
    return {name: params[name] for name in ("wan_pd_input_fingerprint", "wan_pd_input_identity") if name in params}


async def _query_publication(context: zmq.asyncio.Context, address: str, ticket: ReservationTicket, limit: int):
    # Use a fresh REQ socket after any timeout; a half-completed REQ exchange
    # cannot be reused safely. One bounded lookup is live per rendezvous.
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, limit)
    socket.connect("tcp://" + address)
    try:
        async with asyncio.timeout(1):
            await socket.send(msgspec.msgpack.encode((GET_PUBLICATION_MSG, ticket.fingerprint)))
            return msgspec.msgpack.decode(await socket.recv())
    except (TimeoutError, zmq.ZMQError):
        return None
    finally:
        socket.close()


def _observe_prefill(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("Receiver-first P request failed: %s", error)


def create_app(config: ProxyConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    pending: dict[str, _Pending] = {}
    callback_requests = 0

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        context = zmq.asyncio.Context() if config.layerwise_publication else None
        app.state.control_context = context
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_s),
            limits=httpx.Limits(max_connections=2 * config.max_pending, max_keepalive_connections=config.max_pending),
            follow_redirects=False,
            transport=transport,
        ) as client:
            app.state.backend_client = client
            try:
                yield
            finally:
                tasks = {
                    task
                    for item in pending.values()
                    for task in (item.prefill_task, item.publication)
                    if task is not None
                }
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                pending.clear()
                if context is not None:
                    context.term()

    app = FastAPI(lifespan=lifespan)

    async def retire(rendezvous_id: str) -> None:
        item = pending.pop(rendezvous_id, None)
        if item is not None:
            for task in (item.prefill_task, item.publication):
                if task is not None:
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task

    def validate_publication(params: Any, ticket: ReservationTicket) -> dict[str, Any]:
        if not isinstance(params, dict) or not params.get("do_remote_prefill"):
            raise HTTPException(502, "P did not publish a PD continuation")
        try:
            manifest = ContinuationManifest.from_wire(params.get("continuation_manifest"))
        except ValueError:
            raise HTTPException(502, "P returned an invalid continuation manifest") from None
        if manifest.reservation_fingerprint != ticket.fingerprint:
            raise HTTPException(502, "P publication is bound to a different receiver reservation")
        return {"transfer_params": params}

    async def prefill(item: _Pending, ticket: ReservationTicket) -> dict[str, Any]:
        payload = copy.deepcopy(item.payload)
        payload.update(stream=False, max_tokens=1, min_tokens=1, ignore_eos=True, stop=None, stop_token_ids=[])
        if "max_completion_tokens" in payload:
            payload["max_completion_tokens"] = 1
        payload.pop("stream_options", None)
        payload["kv_transfer_params"] = {
            **_identity_params(item.payload),
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "wan_pd_reservation": ticket.to_wire(),
        }
        client = app.state.backend_client
        try:
            async with asyncio.timeout(max(0, item.deadline - monotonic())):
                async with client.stream(
                    "POST", config.prefill_url.rstrip("/") + item.path, json=payload, headers=item.headers
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(502, f"P rejected the request with HTTP {response.status_code}")
                    result = await _response_json(response, config.max_response_bytes)
            return validate_publication(result.get("kv_transfer_params"), ticket)
        except (httpx.HTTPError, TimeoutError):
            raise HTTPException(504, "Timed out awaiting the P publication") from None

    async def await_publication(item: _Pending, ticket: ReservationTicket) -> dict[str, Any]:
        assert item.prefill_task is not None
        if not config.layerwise_publication:
            return await item.prefill_task
        assert config.producer_control_address is not None
        while not item.prefill_task.done():
            if monotonic() >= item.deadline:
                raise HTTPException(504, "Timed out awaiting the P publication")
            publication = await _query_publication(
                app.state.control_context, config.producer_control_address, ticket, config.max_response_bytes
            )
            if publication is not None:
                if not isinstance(publication, dict):
                    raise HTTPException(502, "P returned an invalid prepared publication")
                if publication.get("status") == "FAILED":
                    raise HTTPException(502, "P aborted the prepared publication")
                if publication.get("status") in ("PREPARED", "SEALED"):
                    return validate_publication(publication.get("transfer_params"), ticket)
            await asyncio.sleep(0.01)
        return await item.prefill_task

    async def handle_reservation(request: Request):
        body = await _request_json(request, config.max_body_bytes)
        try:
            ticket = ReservationTicket.from_wire(body.get("ticket"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        item = pending.get(ticket.rendezvous_id)
        if item is None or monotonic() >= item.deadline:
            raise HTTPException(410, "Receiver rendezvous is no longer active")
        if item.ticket_fingerprint not in (None, ticket.fingerprint):
            raise HTTPException(409, "Rendezvous already belongs to a different receiver allocation")
        if item.callback_waiters >= 2:
            raise HTTPException(429, "Too many concurrent callbacks for this reservation")
        if item.publication is None:
            item.ticket_fingerprint = ticket.fingerprint
            item.prefill_task = asyncio.create_task(prefill(item, ticket))
            item.prefill_task.add_done_callback(_observe_prefill)
            item.publication = asyncio.create_task(await_publication(item, ticket))
        item.callback_waiters += 1
        try:
            # A retried/abandoned callback must not start or cancel a second P
            # execution. The original D request owns the publication task.
            return await asyncio.shield(item.publication)
        finally:
            item.callback_waiters -= 1

    @app.post("/v1/wan-pd/reservation")
    async def reservation(request: Request):
        nonlocal callback_requests
        if callback_requests >= 2 * config.max_pending:
            raise HTTPException(429, "Receiver callback capacity exhausted")
        callback_requests += 1
        try:
            async with asyncio.timeout(config.timeout_s):
                return await handle_reservation(request)
        except TimeoutError:
            raise HTTPException(504, "Receiver callback deadline exceeded") from None
        finally:
            callback_requests -= 1

    async def forward(request: Request):
        if len(pending) >= config.max_pending:
            raise HTTPException(429, "Receiver-first request capacity exhausted")
        rendezvous_id = uuid4().hex
        headers = {"X-Request-Id": rendezvous_id}
        if authorization := request.headers.get("authorization"):
            headers["Authorization"] = authorization
        # Claim admission before awaiting the request body, so concurrent slow
        # uploads cannot accumulate an unbounded number of retained payloads.
        item = _Pending({}, request.url.path, headers, monotonic() + config.timeout_s)
        pending[rendezvous_id] = item
        client = app.state.backend_client
        try:
            async with asyncio.timeout(max(0, item.deadline - monotonic())):
                payload = await _request_json(request, config.max_body_bytes)
                _validate_single_request(payload, request.url.path)
                item.payload = payload
                decode_payload = copy.deepcopy(payload)
                decode_payload["kv_transfer_params"] = {
                    **_identity_params(payload),
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                    "wan_pd_receiver_first": True,
                    "wan_pd_rendezvous_id": rendezvous_id,
                }
                response = await client.send(
                    client.build_request(
                        "POST", config.decode_url.rstrip("/") + request.url.path, json=decode_payload, headers=headers
                    ),
                    stream=True,
                )
        except BaseException as exc:
            await retire(rendezvous_id)
            if isinstance(exc, (httpx.HTTPError, TimeoutError)):
                raise HTTPException(504, "Timed out awaiting D") from None
            raise

        async def stream():
            used = 0
            try:
                async with asyncio.timeout(max(0, item.deadline - monotonic())):
                    async for chunk in response.aiter_bytes():
                        used += len(chunk)
                        if used > config.max_response_bytes:
                            raise RuntimeError("D response exceeds the configured byte limit")
                        yield chunk
            finally:
                try:
                    await response.aclose()
                finally:
                    await retire(rendezvous_id)

        return StreamingResponse(
            stream(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    app.post("/v1/completions")(forward)
    app.post("/v1/chat/completions")(forward)

    @app.get("/health")
    async def health():
        return JSONResponse({"pending_requests": len(pending), "max_pending_requests": config.max_pending})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-body-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-response-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--layerwise-publication", action="store_true")
    parser.add_argument("--producer-control-address", help="P scheduler control HOST:PORT (one logical P engine)")
    args = parser.parse_args()
    import uvicorn

    config = ProxyConfig(
        args.prefill_url,
        args.decode_url,
        args.max_pending,
        args.timeout_s,
        args.max_body_bytes,
        args.max_response_bytes,
        args.layerwise_publication,
        args.producer_control_address,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
