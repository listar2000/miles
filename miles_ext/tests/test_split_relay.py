"""The split trainer's local relay and its heartbeat to the engine container."""

import threading

import pytest


def test_relay_forwards_to_the_engine_port():
    pytest.importorskip("aiohttp")
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from miles_ext.launch.split import relay

    async def run():
        seen = []

        async def upstream(request):
            seen.append((request.path, request.query_string, await request.text()))
            return web.json_response({"ok": True})

        up = web.Application()
        up.router.add_route("*", "/{tail:.*}", upstream)
        async with TestServer(up) as server:
            base = f"http://{server.host}:{server.port}/tok"
            async with TestClient(TestServer(relay.make_app(base, 32001))) as client:
                resp = await client.post("/generate?x=1", data="body")
                assert resp.status == 200 and (await resp.json()) == {"ok": True}
        return seen

    assert asyncio.run(run()) == [("/tok/32001/generate", "x=1", "body")]


def test_no_heartbeat_overwrites_release(monkeypatch):
    from miles_ext.launch import split
    from miles_ext.launch.split import relay

    monkeypatch.setattr(relay, "HEARTBEAT_SECS", 0.001)
    progress = {}
    relay.start_heartbeat("run1", progress)
    relay.release("run1", progress)
    threading.Event().wait(0.05)  # many beat intervals
    assert progress[split.trainer_key("run1")]["done"] is True
