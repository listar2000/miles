"""The session proxy forwards every in-flight agent turn at once.

Each agent turn holds one upstream connection until the model replies, so a pooled
connection cap is a cap on agents generating. aiohttp's default pool is 100.
"""

import asyncio

from aiohttp import ClientSession, TCPConnector, web

from miles_ext.harbor.session_proxy import make_app

N = 150  # above aiohttp's default pool of 100


def test_proxy_forwards_more_than_100_turns_concurrently():
    assert asyncio.run(_peak_concurrent_turns()) == N


async def _peak_concurrent_turns() -> int:
    in_flight, peak, release = 0, 0, asyncio.Event()

    async def slow_turn(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight >= N:
            release.set()
        await release.wait()
        in_flight -= 1
        return web.json_response({"ok": True})

    upstream = web.Application()
    upstream.router.add_post("/sessions/{sid}/v1/chat/completions", slow_turn)
    up_runner = web.AppRunner(upstream)
    await up_runner.setup()
    up_site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await up_site.start()
    up_port = up_site._server.sockets[0].getsockname()[1]

    proxy_runner = web.AppRunner(make_app("127.0.0.1", range(up_port, up_port + 1), "tok"))
    await proxy_runner.setup()
    proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
    await proxy_site.start()
    proxy_port = proxy_site._server.sockets[0].getsockname()[1]

    try:
        async with ClientSession(connector=TCPConnector(limit=0)) as client:

            async def turn(i):
                url = f"http://127.0.0.1:{proxy_port}/tok/{up_port}/sessions/s{i}/v1/chat/completions"
                async with client.post(url, json={}) as r:
                    assert r.status == 200

            await asyncio.wait_for(asyncio.gather(*(turn(i) for i in range(N))), timeout=30)
    finally:
        await proxy_runner.cleanup()
        await up_runner.cleanup()
    return peak
