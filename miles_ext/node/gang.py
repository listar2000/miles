"""Coordination inside a trainer gang: a dict and a queue on rank 0, over the gang's network (ext-064)."""

import asyncio
import collections
import contextlib
import http.client
import json
import queue as queue_lib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from miles_ext.node.contract import GANG_PORT

LONG_POLL_SECS = 30
HEAD_GONE_SECS = 300  # a worker that cannot reach rank 0 for this long fails instead of holding its GPUs
DRAIN_SECS = 60
RELEASE_POLL_SECS = 2.0
_UNREACHABLE = (OSError, http.client.HTTPException, ValueError)


def _result_key(run_id: str) -> str:
    return f"release:{run_id}"


def _seen_key(run_id: str, rank: int) -> str:
    return f"released:{run_id}:{rank}"


class _Queue:
    """Provide partitioned FIFO queues; raise ``queue.Empty`` when ``get`` times out."""

    def __init__(self):
        self._items: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    def put(self, item, *, partition: str) -> None:
        self._items[partition].append(item)

    def take(self, partition: str):
        items = self._items[partition]
        if not items:
            raise queue_lib.Empty
        return items.popleft()

    def get(self, *, partition: str, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.take(partition)
            except queue_lib.Empty:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.1)


class GangServer:
    """Rank 0: ``kv`` and ``queue``, for this process and, over ``GANG_PORT``, for the other nodes."""

    def __init__(self):
        self.kv: dict = {}
        self.queue = _Queue()

    def start(self, port: int = GANG_PORT) -> "GangServer":
        ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve(port, ready)), daemon=True)
        self._thread.start()
        if not ready.wait(60):
            raise RuntimeError(f"gang server did not open port {port}")
        return self

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._stopped.set)
        self._thread.join(10)

    def release(self, run_id: str, succeeded: bool, nodes: int) -> None:
        """Tell every worker how the run ended and stay up until each has confirmed it, or ``DRAIN_SECS`` passed."""
        self.kv[_result_key(run_id)] = succeeded
        deadline = time.monotonic() + DRAIN_SECS
        while time.monotonic() < deadline and any(
            self.kv.get(_seen_key(run_id, rank)) is None for rank in range(1, nodes)
        ):
            time.sleep(0.2)

    async def _serve(self, port: int, ready: threading.Event):
        from aiohttp import web

        async def get_kv(request):
            return web.json_response({"value": self.kv.get(request.match_info["key"])})

        async def put_kv(request):
            self.kv[request.match_info["key"]] = (await request.json())["value"]
            return web.json_response({})

        async def put_q(request):
            self.queue.put((await request.json())["item"], partition=request.match_info["partition"])
            return web.json_response({})

        async def get_q(request):
            deadline = time.monotonic() + float(request.query.get("timeout", 0))
            while True:
                try:
                    return web.json_response({"item": self.queue.take(request.match_info["partition"])})
                except queue_lib.Empty:
                    if time.monotonic() >= deadline:
                        return web.json_response({}, status=404)
                await asyncio.sleep(0.1)

        app = web.Application()
        app.router.add_get("/kv/{key}", get_kv)
        app.router.add_put("/kv/{key}", put_kv)
        app.router.add_put("/q/{partition}", put_q)
        app.router.add_get("/q/{partition}", get_q)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        self._loop, self._stopped = asyncio.get_running_loop(), asyncio.Event()
        ready.set()
        await self._stopped.wait()
        await runner.cleanup()


def _call(url: str, method: str = "GET", body: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class _RemoteKV:
    def __init__(self, base: str):
        self.base = base

    def get(self, key: str):
        return _call(f"{self.base}/kv/{urllib.parse.quote(key, safe='')}")["value"]

    def __setitem__(self, key: str, value) -> None:
        _call(f"{self.base}/kv/{urllib.parse.quote(key, safe='')}", "PUT", {"value": value})


class _RemoteQueue:
    def __init__(self, base: str):
        self.base = base

    def put(self, item, *, partition: str) -> None:
        _call(f"{self.base}/q/{urllib.parse.quote(partition, safe='')}", "PUT", {"item": item})

    def get(self, *, partition: str, timeout: float):
        wait = min(timeout, LONG_POLL_SECS)
        url = f"{self.base}/q/{urllib.parse.quote(partition, safe='')}?timeout={wait}"
        try:
            return _call(url, timeout=wait + 30)["item"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise queue_lib.Empty from None
            raise


class GangClient:
    """A worker node's view of rank 0's ``kv`` and ``queue``. Calls raise ``OSError`` while rank 0 is unreachable."""

    def __init__(self, head_addr: str, port: int = GANG_PORT):
        base = f"http://{head_addr}:{port}"
        self.kv, self.queue = _RemoteKV(base), _RemoteQueue(base)

    def wait_release(
        self,
        run_id: str,
        rank: int,
        *,
        timeout: float,
        head_gone_secs: float = HEAD_GONE_SECS,
        poll: float = RELEASE_POLL_SECS,
    ) -> bool:
        """Block until rank 0 says how the run ended, and confirm it. Raises if rank 0 stays unreachable or silent. The
        result stays readable, so a reply lost on the way here costs one more poll."""
        deadline, last_seen = time.monotonic() + timeout, time.monotonic()
        while True:
            try:
                result = self.kv.get(_result_key(run_id))
                last_seen = time.monotonic()
            except _UNREACHABLE as e:
                if time.monotonic() - last_seen > head_gone_secs:
                    raise RuntimeError(f"rank 0 unreachable for {head_gone_secs:.0f}s: {e}") from e
            else:
                if result is not None:
                    with contextlib.suppress(*_UNREACHABLE):  # rank 0 may already be gone: the result stands
                        self.kv[_seen_key(run_id, rank)] = True
                    return bool(result)
            if time.monotonic() > deadline:
                raise TimeoutError(f"no release from rank 0 within {timeout:.0f}s")
            time.sleep(poll)
