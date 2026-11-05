"""The split trainer's relay: local ports to a rollout node's public URL."""

import argparse
import socket
import subprocess
import sys
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from miles_ext.recipes import recipe as R

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


async def _serve(upstream: str, mapping: dict[int, int]):
    import asyncio

    runners = []
    for local, remote in mapping.items():
        runner = web.AppRunner(make_app(upstream, remote), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", local).start()
        runners.append(runner)
    await asyncio.Event().wait()


def local_ports(recipe: dict) -> dict[str, list[int] | int]:
    """The trainer's local ports: one per engine (the argv's external addresses), then the agent."""
    n = R.num_engines(recipe)
    return {"engines": [R.SPLIT_ENGINE_PORT + i for i in range(n)], "agent": R.SPLIT_ENGINE_PORT + n}


def connect(recipe: dict, member: dict) -> tuple[subprocess.Popen, str]:
    """Start this node's relays to a registered rollout node. Returns the relay process and the agent's local URL
    (MILES_SPLIT_AGENT_URL)."""
    ports = local_ports(recipe)
    remote = [*member["engine_ports"], member["agent_port"]]
    local = [*ports["engines"], ports["agent"]]
    if len(remote) != len(local):
        raise RuntimeError(
            f"the rollout node serves {len(remote) - 1} engines, the recipe expects "
            f"{len(local) - 1}; relaunch both with the same recipe"
        )
    mapping = [f"{a}={b}" for a, b in zip(local, remote, strict=True)]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "miles_ext.launch.split.relay",
            "--upstream",
            f"{member['url']}/{member['token']}",
            "--map",
            *mapping,
        ]
    )
    for port in local:
        _wait_port(port, proc)
    print(f"split: engines {member['url']}/<token> relayed at 127.0.0.1:{local}", flush=True)
    return proc, f"http://127.0.0.1:{ports['agent']}"


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
    ap.add_argument("--upstream", required=True, help="<public url>/<token>")
    ap.add_argument("--map", nargs="+", required=True, help="<local port>=<rollout node port> ...")
    a = ap.parse_args()
    asyncio.run(_serve(a.upstream.rstrip("/"), dict(tuple(int(x) for x in m.split("=")) for m in a.map)))


if __name__ == "__main__":
    main()
