# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU HTTP-flow checks of the optional receiver-first proxy example."""

import asyncio
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.manifest import (
    ContinuationManifest,
    RequestIdentity,
    StateComponent,
    fingerprint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.continuation.reservation import ReservationTicket


@pytest.fixture(scope="module")
def proxy():
    path = Path(__file__).parents[4] / "examples/disaggregated_prefill_v1/wan_pd_receiver_first_proxy.py"
    spec = importlib.util.spec_from_file_location("wan_pd_test_proxy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name)


def _ticket(rendezvous_id):
    return ReservationTicket(
        receiver_epoch="decoder-epoch",
        reservation_id="reservation-a",
        request_id="decoder-request",
        rendezvous_id=rendezvous_id,
        identity=RequestIdentity("weights-v1", "input-v1", 4),
        state_position=4,
        prefix_tokens=0,
        allocation_fingerprint=fingerprint({"blocks": [[10, 11]]}),
    )


def _publication(ticket):
    params = {
        "do_remote_prefill": True,
        "remote_block_ids": [[1, 2]],
        "remote_engine_id": "producer",
        "remote_request_id": "producer-request",
        "remote_host": "127.0.0.1",
        "remote_port": 5000,
        "remote_num_prompt_tokens": 4,
    }
    manifest = ContinuationManifest(
        request_id="producer-request",
        attempt_id="producer-attempt",
        engine_epoch="producer-epoch",
        namespace=ticket.identity.namespace,
        layout_fingerprint="layout-a",
        input_fingerprint=ticket.identity.input_fingerprint,
        prompt_tokens=4,
        state_position=4,
        prefix_tokens=0,
        components=(StateComponent("k", "layer.0", 0, "full_attention", "float16", (2, 8), 0, 4),),
        transfer_plan_fingerprint=fingerprint(
            {key: value for key, value in params.items() if key.startswith("remote_")}
        ),
        reservation_fingerprint=ticket.fingerprint,
    )
    params["continuation_manifest"] = manifest.to_wire()
    return params


def test_receiver_allocates_before_p_and_replayed_callback_reuses_publication(proxy):
    async def run():
        calls = []
        last_ticket = None

        async def backend(request):
            nonlocal last_ticket
            payload = json.loads(request.content)
            calls.append(request.url.host)
            if request.url.host == "decode":
                assert calls == ["decode"]
                assert payload["prompt"] == [1, 2, 3, 4]
                assert payload["max_tokens"] == 9 and payload["stop"] == ["end"]
                last_ticket = _ticket(payload["kv_transfer_params"]["wan_pd_rendezvous_id"])
                callback = {"ticket": last_ticket.to_wire()}
                first = await client.post("/v1/wan-pd/reservation", json=callback)
                assert first.status_code == 200
                second = await client.post("/v1/wan-pd/reservation", json=callback)
                assert second.json() == first.json()
                conflicting = replace(last_ticket, reservation_id="reused-allocation")
                assert (
                    await client.post("/v1/wan-pd/reservation", json={"ticket": conflicting.to_wire()})
                ).status_code == 409
                assert (
                    first.json()["transfer_params"]["continuation_manifest"]["manifest"]["reservation_fingerprint"]
                    == last_ticket.fingerprint
                )
                return httpx.Response(200, json={"choices": [{"text": "D answer"}]})
            assert request.url.host == "prefill"
            assert payload["prompt"] == [1, 2, 3, 4]
            assert payload["max_tokens"] == payload["min_tokens"] == 1
            assert payload["stop"] is None and not payload["stream"]
            ticket = ReservationTicket.from_wire(payload["kv_transfer_params"]["wan_pd_reservation"])
            return httpx.Response(200, json={"kv_transfer_params": _publication(ticket)})

        app = proxy.create_app(
            proxy.ProxyConfig("http://prefill", "http://decode"), transport=httpx.MockTransport(backend)
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client,
        ):
            response = await client.post(
                "/v1/completions", json={"prompt": [1, 2, 3, 4], "max_tokens": 9, "stop": ["end"]}
            )
            assert response.status_code == 200 and response.json()["choices"][0]["text"] == "D answer"
            assert calls == ["decode", "prefill"]
            assert (await client.get("/health")).json()["pending_requests"] == 0
            assert (
                await client.post("/v1/wan-pd/reservation", json={"ticket": last_ticket.to_wire()})
            ).status_code == 410
            assert (await client.post("/v1/completions", json={"prompt": ["a", "b"]})).status_code == 400
            assert calls == ["decode", "prefill"]

    asyncio.run(run())


def test_capacity_and_full_prefix_hit_do_not_launch_unneeded_p(proxy):
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def backend(request):
            calls.append(request.url.host)
            assert request.url.host == "decode"
            started.set()
            await finish.wait()
            # A fully cached D can finish without an allocation callback.
            return httpx.Response(200, content=b'data: {"text":"cached"}\n\ndata: [DONE]\n\n')

        app = proxy.create_app(
            proxy.ProxyConfig("http://prefill", "http://decode", max_pending=1), transport=httpx.MockTransport(backend)
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client,
        ):
            first = asyncio.create_task(client.post("/v1/chat/completions", json={"messages": [], "stream": True}))
            await started.wait()
            overflow = await client.post("/v1/completions", json={"prompt": "second"})
            assert overflow.status_code == 429
            finish.set()
            response = await first
            assert response.status_code == 200 and b"[DONE]" in response.content
            assert calls == ["decode"]
            assert (await client.get("/health")).json()["pending_requests"] == 0

    asyncio.run(run())


def test_prepared_layers_can_be_returned_while_p_http_is_running(monkeypatch, proxy):
    async def run():
        p_started, p_finish = asyncio.Event(), asyncio.Event()
        ticket = None
        calls = []

        async def catalog(context, address, queried_ticket, limit):
            assert address == "127.0.0.1:5000"
            await p_started.wait()
            assert not p_finish.is_set()
            return {"status": "PREPARED", "transfer_params": _publication(queried_ticket), "error": None}

        monkeypatch.setattr(proxy, "_query_publication", catalog)

        async def backend(request):
            nonlocal ticket
            calls.append(request.url.host)
            payload = json.loads(request.content)
            if request.url.host == "decode":
                ticket = _ticket(payload["kv_transfer_params"]["wan_pd_rendezvous_id"])
                ready = await client.post("/v1/wan-pd/reservation", json={"ticket": ticket.to_wire()})
                assert ready.status_code == 200
                assert p_started.is_set() and not p_finish.is_set()
                # Actual D now READs ready components, waits for SEALED and
                # copy completion, then commits. This proxy forwards its result.
                p_finish.set()
                return httpx.Response(200, json={"choices": [{"text": "continued"}]})
            p_started.set()
            await p_finish.wait()
            return httpx.Response(200, json={"kv_transfer_params": _publication(ticket)})

        config = proxy.ProxyConfig(
            "http://prefill", "http://decode", layerwise_publication=True, producer_control_address="127.0.0.1:5000"
        )
        app = proxy.create_app(config, transport=httpx.MockTransport(backend))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client,
        ):
            response = await client.post("/v1/completions", json={"prompt": "long input"})
            assert response.status_code == 200
            assert calls == ["decode", "prefill"]
            assert (await client.get("/health")).json()["pending_requests"] == 0

    asyncio.run(run())
