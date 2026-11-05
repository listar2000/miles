"""Rollout membership: a registry on rank 0 that rollout nodes join by beating over HTTP (ext-063)."""

import asyncio
import hmac
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from miles_ext.node.contract import REGISTRY_PORT
from miles_ext.recipes.recipe import SPLIT_ENGINE_PORT

BEAT_SECS = 15
MEMBER_TTL_SECS = 4 * BEAT_SECS
RELEASE_TIMEOUT_SECS = 30
WAIT_POLL_SECS = 5
# what a call across a public tunnel can raise while the other end is away or restarting
UNREACHABLE = (OSError, http.client.HTTPException, ValueError)


class Registry:
    def __init__(self, pool_token: str):
        self.pool_token = pool_token
        self.done = False
        self._members: dict[str, dict] = {}
        self._evicted: set[str] = set()
        self._next_slot = SPLIT_ENGINE_PORT
        self._lock = threading.Lock()

    def beat(self, member: dict) -> dict:
        with self._lock:
            node_id = member["node_id"]
            if node_id in self._evicted:
                return {"done": self.done, "evicted": True}
            known = self._members.get(node_id)
            if known is None:
                n = len(member["engine_ports"])
                slots = list(range(self._next_slot, self._next_slot + n))
                known = {"slots": slots, "agent_slot": self._next_slot + n}
                self._next_slot += n + 1
                print(f"registry: {node_id} joined at {member['url']}, slots {slots}", flush=True)
            self._members[node_id] = {
                **member,
                "slots": known["slots"],
                "agent_slot": known["agent_slot"],
                "seen_at": time.time(),
            }
            return {"done": self.done, "evicted": False}

    def evict(self, node_id: str, why: str) -> None:
        """Out for good: a node that missed a sync serves stale weights, so it must not come back. It exits on its next
        beat, and a caller who wants the capacity back starts another."""
        with self._lock:
            self._evicted.add(node_id)
            self._members.pop(node_id, None)
        print(f"registry: evicted {node_id}: {why}", flush=True)

    def status(self) -> dict:
        """For a caller deciding whether to add or drop nodes: who is here and how loaded each engine is."""
        return {
            "at": time.time(),
            "done": self.done,
            "nodes": [{k: m.get(k) for k in ("node_id", "url", "seen_at", "load")} for m in self.members()],
        }

    def members(self) -> list[dict]:
        cutoff = time.time() - MEMBER_TTL_SECS
        with self._lock:
            return [m for m in self._members.values() if m["seen_at"] >= cutoff]

    def release(self) -> None:
        """The run is complete: tell every live node now; a node that misses this reads ``done`` on its next beat
        if rank 0 is still up, and otherwise leaves after its idle limit."""
        with self._lock:
            self.done = True
        for m in self.members():
            try:
                _post(f"{m['url']}/{m['token']}/{m['agent_port']}/release", {}, None, RELEASE_TIMEOUT_SECS)
            except UNREACHABLE as e:
                print(f"registry: release of {m['node_id']}: {type(e).__name__}: {e}", flush=True)

    def app(self):
        from aiohttp import web

        @web.middleware
        async def auth(request, handler):
            given = request.headers.get("Authorization", "")
            if not hmac.compare_digest(given, f"Bearer {self.pool_token}"):
                raise web.HTTPUnauthorized()
            return await handler(request)

        async def beat(request):
            return web.json_response(self.beat(await request.json()))

        async def members(request):
            return web.json_response({"members": self.members()})

        async def evict(request):
            body = await request.json()
            self.evict(body["node_id"], body.get("why", ""))
            return web.json_response({})

        app = web.Application(middlewares=[auth])
        app.router.add_post("/beat", beat)
        app.router.add_get("/members", members)
        app.router.add_post("/evict", evict)
        return app

    def start(self, port: int = REGISTRY_PORT) -> "Registry":
        ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve(port, ready)), daemon=True)
        self._thread.start()
        if not ready.wait(60):
            raise RuntimeError(f"registry did not open port {port}")
        return self

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._stopped.set)
        self._thread.join(10)

    async def _serve(self, port: int, ready: threading.Event):
        from aiohttp import web

        self._loop, self._stopped = asyncio.get_running_loop(), asyncio.Event()
        runner = web.AppRunner(self.app(), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        ready.set()
        await self._stopped.wait()
        await runner.cleanup()


def _post(url: str, body: dict, pool_token: str | None, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if pool_token:
        headers["Authorization"] = f"Bearer {pool_token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def beat(registry_url: str, pool_token: str, member: dict, timeout: float = 30) -> dict:
    return _post(f"{registry_url.rstrip('/')}/beat", member, pool_token, timeout)


def members(registry_url: str, pool_token: str, timeout: float = 30) -> list[dict]:
    req = urllib.request.Request(
        f"{registry_url.rstrip('/')}/members", headers={"Authorization": f"Bearer {pool_token}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["members"]


def evict(registry_url: str, pool_token: str, node_id: str, why: str) -> None:
    _post(f"{registry_url.rstrip('/')}/evict", {"node_id": node_id, "why": why}, pool_token, 30)


def wait_members(
    registry_url: str, pool_token: str, *, timeout: float, check: Callable[[], None] | None = None
) -> list[dict]:
    """Block until a rollout node has joined. ``check`` may raise to give up early (the caller saw the node die)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if found := members(registry_url, pool_token):
                return found
        except UNREACHABLE:  # rank 0's registry is not up yet
            pass
        if check:
            check()
        if time.monotonic() > deadline:
            raise TimeoutError(f"no rollout node joined {registry_url} within {timeout:.0f}s")
        time.sleep(WAIT_POLL_SECS)
