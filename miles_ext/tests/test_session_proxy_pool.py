"""The session proxy forwards every in-flight agent turn at once (aiohttp's default pool is 100)."""

import asyncio

from aiohttp import ClientSession, TCPConnector, web
from aiohttp.test_utils import TestServer

from miles_ext.harbor.session_proxy import make_app

N = 150


def test_proxy_forwards_more_than_100_turns_concurrently():
    asyncio.run(_send_concurrent_turns())


async def _send_concurrent_turns():
    arrived, all_in = 0, asyncio.Event()

    async def turn(request):
        nonlocal arrived
        arrived += 1
        if arrived == N:
            all_in.set()
        await all_in.wait()
        return web.json_response({})

    upstream = web.Application()
    upstream.router.add_post("/{tail:.*}", turn)
    async with TestServer(upstream) as up:
        proxy_app = make_app("127.0.0.1", range(up.port, up.port + 1), "tok")
        async with TestServer(proxy_app) as proxy, ClientSession(connector=TCPConnector(limit=0)) as client:

            async def call(i):
                async with client.post(proxy.make_url(f"/tok/{up.port}/sessions/s{i}/v1/chat/completions")) as r:
                    assert r.status == 200

            try:
                await asyncio.wait_for(asyncio.gather(*(call(i) for i in range(N))), timeout=10)
            finally:
                all_in.set()  # release parked turns so teardown is quick when this fails
