"""The split trainer's relay: local ports to the rollout nodes' public URLs, following the registry."""

import argparse
import os
import socket
import subprocess
import sys
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from miles_ext.node import contract as C
from miles_ext.node import registry as node_registry
from miles_ext.recipes import recipe as R

POLL_SECS = 3
_DROP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


def make_app(upstream: str, remote_port: int) -> web.Application:
    async def on_startup(app):
        # a generated turn can take an hour and the caller's own timeout governs, as in the session proxy
        app["client"] = ClientSession(
            connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None), auto_decompress=False
        )

    async def on_cleanup(app):
        await app["client"].close()

    async def forward(request: web.Request) -> web.Response:
        url = f"{upstream}/{remote_port}/{request.match_info['tail']}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP}
        async with request.app["client"].request(
            request.method, url, params=request.query, headers=headers, data=await request.read()
        ) as resp:
            body = await resp.read()
            return web.Response(
                status=resp.status,
                body=body,
                headers={k: v for k, v in resp.headers.items() if k.lower() not in _DROP},
            )

    app = web.Application(client_max_size=256 * 1024**2)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", forward)
    return app


def desired(members: list[dict]) -> dict[int, tuple[str, int]]:
    """Each slot and where it leads: ``{slot: (<public url>/<token>, <port on the node>)}``."""
    out = {}
    for m in members:
        upstream = f"{m['url']}/{m['token']}"
        out.update({slot: (upstream, port) for slot, port in zip(m["slots"], m["engine_ports"], strict=True)})
        out[m["agent_slot"]] = (upstream, m["agent_port"])
    return out


async def _follow(registry_url: str, pool_token: str, poll: float = POLL_SECS):
    import asyncio

    open_slots: dict[int, web.AppRunner] = {}
    while True:
        try:
            members = await asyncio.to_thread(node_registry.members, registry_url, pool_token)
        except node_registry.UNREACHABLE as e:  # rank 0 is restarting or busy: keep what is open
            print(f"split relay: reading the registry: {type(e).__name__}: {e}", flush=True)
        else:
            want = desired(members)
            for slot in set(open_slots) - set(want):
                await open_slots.pop(slot).cleanup()
            for slot in set(want) - set(open_slots):
                runner = web.AppRunner(make_app(*want[slot]), access_log=None)
                await runner.setup()
                await web.TCPSite(runner, "127.0.0.1", slot).start()
                open_slots[slot] = runner
        await asyncio.sleep(poll)


def connect(recipe: dict, registry_url: str, pool_token: str, members: list[dict]) -> subprocess.Popen:
    """Start this node's relay, once the first joined nodes' slots are open. Returns the process."""
    if not R.is_elastic(recipe):
        for m in members:
            if len(m["engine_ports"]) != R.num_engines(recipe):
                raise RuntimeError(
                    f"the rollout node serves {len(m['engine_ports'])} engines, the recipe expects "
                    f"{R.num_engines(recipe)}; relaunch both with the same recipe"
                )
    proc = subprocess.Popen(
        [sys.executable, "-m", "miles_ext.launch.split.relay", "--registry", registry_url],
        env={**os.environ, C.POOL_TOKEN_ENV: pool_token},
    )
    for slot in desired(members):
        _wait_port(slot, proc)
    print(f"split: rollout nodes relayed at 127.0.0.1:{sorted(desired(members))}", flush=True)
    return proc


def _wait_port(port: int, proc: subprocess.Popen, timeout: float = 60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"relay port {port} did not open (relay exit code {proc.poll()})") from None
            time.sleep(0.5)


def main():
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True, help="rank 0's registry, as this node reaches it")
    a = ap.parse_args()
    asyncio.run(_follow(a.registry, os.environ[C.POOL_TOKEN_ENV]))


if __name__ == "__main__":
    main()
