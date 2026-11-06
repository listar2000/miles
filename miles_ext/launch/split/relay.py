"""The split trainer's side: local relays from ``127.0.0.1:<port>`` to the engine tunnel, and its heartbeat."""

import argparse
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from miles_ext.launch import split
from miles_ext.recipes import recipe as R

HEARTBEAT_SECS = 30
POLL_SECS = 10
_RELEASED = threading.Event()
_BEAT_LOCK = threading.Lock()  # a beat never lands after release's "done"
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


def agent_url(recipe: dict) -> str:
    """The sync agent's local URL (MILES_SPLIT_AGENT_URL), known before the engines serve."""
    return f"http://127.0.0.1:{local_ports(recipe)['agent']}"


class _Stopped(Exception):
    """The trainer node is shutting down, so nobody needs the relay any more."""


def _wait_engine(
    run_id: str,
    progress,
    *,
    timeout: float,
    reached: Callable[[dict], bool],
    what: str,
    stop: threading.Event | None = None,
) -> dict:
    key, deadline = split.engine_key(run_id), time.monotonic() + timeout
    stop = stop or threading.Event()
    while True:
        state = progress.get(key) or {}
        if state.get("phase") == "ended":
            raise RuntimeError(f"the engine container ended before the trainer connected: {state.get('reason')}")
        if reached(state):
            return state
        if time.monotonic() > deadline:
            raise TimeoutError(f"the engine container was not {what} within {timeout:.0f}s: {state}")
        if stop.wait(POLL_SECS):
            raise _Stopped


def wait_ready(run_id: str, progress, *, timeout: float, stop: threading.Event | None = None) -> dict:
    """Block until the run's engine container serves (its published state), or raise if it ended."""
    return _wait_engine(
        run_id, progress, timeout=timeout, reached=lambda s: s.get("phase") == "ready", what="ready", stop=stop
    )


def wait_container(run_id: str, progress, *, timeout: float) -> None:
    """Block until the engine container runs, so a trainer is not billed while the engines queue for GPUs."""
    _wait_engine(
        run_id,
        progress,
        timeout=timeout,
        reached=lambda s: s.get("phase") == "ready" or "container_at" in s,
        what="running",
    )


def connect(
    recipe: dict, run_id: str, progress, *, timeout: float, stop: threading.Event | None = None
) -> subprocess.Popen:
    """Wait for the run's engine container and start this node's relays; returns the relay process."""
    state = wait_ready(run_id, progress, timeout=timeout, stop=stop)
    ports = local_ports(recipe)
    remote = [*state["engine_ports"], state["agent_port"]]
    local = [*ports["engines"], ports["agent"]]
    if len(remote) != len(local):
        raise RuntimeError(
            f"the engine container serves {len(remote) - 1} engines, the recipe expects "
            f"{len(local) - 1}; relaunch both with the same recipe"
        )
    mapping = [f"{a}={b}" for a, b in zip(local, remote, strict=True)]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "miles_ext.launch.split.relay",
            "--upstream",
            f"{state['url']}/{state['token']}",
            "--map",
            *mapping,
        ]
    )
    try:
        for port in local:
            _wait_port(port, proc, stop=stop)
    except BaseException:
        proc.terminate()
        raise
    print(f"split: engines {state['url']}/<token> relayed at 127.0.0.1:{local}", flush=True)
    return proc


def connect_in_background(
    recipe: dict, run_id: str, progress, *, timeout: float, on_failure: Callable[[str], None]
) -> Callable[[], None]:
    """Run ``connect`` on its own thread, so Ray, the router and the trainer start while the engines load: Miles
    retries the relay's closed ports until they open. Returns the function that stops the relay."""
    stop_event, lock = threading.Event(), threading.Lock()
    started: list[subprocess.Popen] = []

    def run():
        try:
            proc = connect(recipe, run_id, progress, timeout=timeout, stop=stop_event)
        except _Stopped:
            return
        except Exception as e:
            note = f"split: no relay to the engines: {type(e).__name__}: {e}"
            print(note, flush=True)
            if not stop_event.is_set():
                on_failure(note)
            return
        with lock:
            if stop_event.is_set():
                proc.terminate()
            else:
                started.append(proc)

    threading.Thread(target=run, daemon=True).start()

    def stop():
        with lock:
            stop_event.set()
            for proc in started:
                proc.terminate()

    return stop


def _wait_port(port: int, proc: subprocess.Popen, *, stop: threading.Event | None = None, timeout: float = 60):
    deadline = time.monotonic() + timeout
    stop = stop or threading.Event()
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"relay port {port} did not open (relay exit code {proc.poll()})") from None
            if stop.wait(0.5):
                raise _Stopped from None


def publish_segment(run_id: str, progress) -> str:
    """Name this segment for its nodes' failure reports: rank 0 calls this before it opens the head's port."""
    segment = secrets.token_urlsafe(8)
    progress[split.segment_key(run_id)] = segment
    return segment


def current_segment(run_id: str, progress) -> str:
    """The segment a node belongs to: read it once the head's port is open, so after ``publish_segment``."""
    return progress[split.segment_key(run_id)]


def report_node_failure(run_id: str, progress, *, segment: str, note: str) -> None:
    """Tell rank 0 of that segment this node has no relay; a later segment never reads this report."""
    progress[split.relay_failure_key(run_id, segment)] = note


def start_heartbeat(run_id: str, progress, *, segment: str, on_node_failure: Callable[[str], None]):
    def beat():
        while not _RELEASED.is_set():
            try:
                with _BEAT_LOCK:
                    if not _RELEASED.is_set():
                        progress[split.trainer_key(run_id)] = {"heartbeat_at": time.time(), "done": False}
                if note := progress.get(split.relay_failure_key(run_id, segment)):
                    on_node_failure(note)
            except Exception as e:  # a missed beat is fine; the engine's idle cap is minutes
                print(f"split heartbeat: {type(e).__name__}: {e}", flush=True)
            _RELEASED.wait(HEARTBEAT_SECS)

    progress[split.trainer_key(run_id)] = {"heartbeat_at": time.time(), "done": False}
    threading.Thread(target=beat, daemon=True).start()


def release(run_id: str, progress):
    """The run is complete: the engine container exits instead of waiting for another trainer."""
    with _BEAT_LOCK:
        _RELEASED.set()
        progress[split.trainer_key(run_id)] = {"heartbeat_at": time.time(), "done": True}


def main():
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="<tunnel url>/<token>")
    ap.add_argument("--map", nargs="+", required=True, help="<local port>=<engine container port> ...")
    a = ap.parse_args()
    asyncio.run(_serve(a.upstream.rstrip("/"), dict(tuple(int(x) for x in m.split("=")) for m in a.map)))


if __name__ == "__main__":
    main()
