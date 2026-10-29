"""Where a rollout's sandbox time goes: timing spans for each Harbor trial and for the session proxy (ext-040).

Off by default (recipe ``obs.sandbox_profile``). Off, nothing here runs on the trial path: the
agent function hands Harbor the provider unchanged and the proxy builds its app without the
middleware. On, two writers record JSON lines, each duration taken on one clock, and neither
writes on an event loop (a trial's spans are written once, off the loop, when its sandbox stops;
the proxy's lines go through one writer thread):

``<trial_dir>/sandbox_profile.jsonl`` (host side, one file per trial)
    ``ProfiledEnvironment`` wraps the run's sandbox provider (Sail or Harbor's Modal) and
    times ``start``, ``stop``, ``exec``, uploads and downloads. ``depth`` > 0 marks a call made
    inside another timed call (a provider's ``start`` that execs), so sums use depth 0 only.
    ``note_session`` adds the trial's session path, which joins the proxy's lines.
``<run_dir>/sandbox_profile/proxy.jsonl`` (host side, one file per run)
    the proxy's time for every forwarded request, receipt to response, keyed by session path (never
    the token). It includes the upstream's time: in the benchmark (a scripted model) that is near
    zero, in training it is the session server and SGLang, so there it reads "model + proxy".

The in-sandbox side needs nothing new: Harbor's mini-swe-agent step-timing patch already
writes each step's model wait (``harbor_llm_wait_msec``) and command time
(``harbor_tool_exec_msec``) into the agent's trajectory. ``miles_ext.obs.sandbox_bench.report``
joins the three.
"""

import asyncio
import contextvars
import functools
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

ENV = "MILES_EXT_SANDBOX_PROFILE"  # "1": profile this run's trials (set by recipe obs.sandbox_profile)
PROBES_ENV = "MILES_EXT_SANDBOX_PROFILE_PROBES"  # N: time N no-op execs right after each sandbox starts
NO_VERIFIER_ENV = "MILES_EXT_SANDBOX_PROFILE_NO_VERIFIER"  # "1": trials skip the task's verifier (benchmark only)
IMPORT_PATH = "miles_ext.obs.sandbox_profile:ProfiledEnvironment"
TRIAL_FILE = "sandbox_profile.jsonl"
PROXY_FILE = "sandbox_profile/proxy.jsonl"  # under the run dir
_CMD_CHARS = 160  # how much of each exec'd command a span keeps

_depth = contextvars.ContextVar("sandbox_profile_depth", default=0)


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def inner_import_path() -> str:
    """The provider this run would use unprofiled: ``HARBOR_ENV_IMPORT_PATH``, else the class for ``HARBOR_ENV_TYPE``
    (the fork's for ``modal``, Harbor's otherwise)."""
    if path := os.getenv("HARBOR_ENV_IMPORT_PATH", "").strip():
        return path
    from harbor.environments.factory import _ENVIRONMENT_REGISTRY
    from harbor.models.environment_type import EnvironmentType

    from miles_ext.sandboxes import modal_env

    env_type = os.environ["HARBOR_ENV_TYPE"].strip().lower()
    if env_type == modal_env.REPLACES_TYPE:  # ext-041: the agent function runs the fork's Modal class
        return modal_env.IMPORT_PATH
    entry = _ENVIRONMENT_REGISTRY[EnvironmentType(env_type)]
    return f"{entry.module}:{entry.class_name}"


def profiled_config(config):
    """``config`` (a Harbor trial ``EnvironmentConfig``) with its provider wrapped; every other field kept."""
    return config.model_copy(update={"type": None, "import_path": IMPORT_PATH})


def write_lines(path: str | Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with open(path, "a") as f:
        f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))


def write_line(path: str | Path, record: dict[str, Any]) -> None:
    write_lines(path, [record])


def note_session(trial_dir: str | Path, session_url: str) -> None:
    """Record the trial's session path (``/<port>/sessions/<id>``), the key of its lines in the proxy log."""
    from urllib.parse import urlsplit

    url = urlsplit(session_url)  # the head-side URL http://<ip>:<port>/sessions/<id>; the proxy logs /<port>/<tail>
    write_line(Path(trial_dir) / TRIAL_FILE, {"op": "session", "t0": time.time(), "path": f"/{url.port}{url.path}"})


def _timed(op: str, describe=None):
    """Time an async provider method; ``describe(arguments)`` (the call's bound arguments by name) adds fields."""

    def wrap(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        async def timed(self, *args, **kwargs):
            depth = _depth.get()
            t0, mono = time.time(), time.monotonic()
            record: dict[str, Any] = {"op": op, "t0": t0, "depth": depth}
            if describe:
                try:  # a description never fails the call it describes
                    record.update(describe(signature.bind(self, *args, **kwargs).arguments))
                except Exception as e:
                    record["describe_error"] = type(e).__name__
            token = _depth.set(depth + 1)
            try:
                result = await method(self, *args, **kwargs)
                record["ok"] = True
                if (rc := getattr(result, "return_code", None)) is not None:
                    record["rc"] = rc
                return result
            except BaseException as e:
                record["ok"], record["error"] = False, type(e).__name__
                raise
            finally:
                _depth.reset(token)
                record["dt"] = time.monotonic() - mono
                self.__dict__.setdefault("_profile_spans", []).append(record)

        return timed

    return wrap


def _describe_exec(a: dict) -> dict:
    return {"cmd": str(a.get("command"))[:_CMD_CHARS], "timeout_sec": a.get("timeout_sec")}


def _describe_transfer(a: dict) -> dict:
    src = a.get("source_path", a.get("source_dir"))
    dst = a.get("target_path", a.get("target_dir"))
    return {"src": str(src)[:_CMD_CHARS], "dst": str(dst)[:_CMD_CHARS]}


@functools.cache
def profiled(inner: type) -> type:
    """A subclass of the provider class ``inner`` that times its sandbox calls into the trial dir."""

    class Profiled(inner):
        async def start(self, force_build: bool) -> None:
            await self._profiled_start(force_build)
            for _ in range(int(os.getenv(PROBES_ENV) or 0)):
                await self._probe_exec("true")

        @_timed("start", lambda force_build: {"provider": inner.__name__})
        async def _profiled_start(self, force_build: bool) -> None:
            await super().start(force_build)

        @_timed("probe_exec", _describe_exec)
        async def _probe_exec(self, command: str):
            return await super().exec(command)

        async def stop(self, delete: bool):
            try:
                return await self._profiled_stop(delete)
            finally:  # the trial's spans, written once and off the event loop
                spans, self.__dict__["_profile_spans"] = self.__dict__.get("_profile_spans", []), []
                path = Path(self.trial_paths.trial_dir) / TRIAL_FILE
                await asyncio.to_thread(write_lines, path, spans)

        @_timed("stop")
        async def _profiled_stop(self, delete: bool):
            return await super().stop(delete)

        @_timed("exec", _describe_exec)
        async def exec(self, command: str, cwd=None, env=None, timeout_sec=None, user=None):
            # BaseEnvironment.exec's signature, so spans see each argument by name
            return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

        @_timed("upload_file", _describe_transfer)
        async def upload_file(self, source_path, target_path):
            return await super().upload_file(source_path, target_path)

        @_timed("upload_dir", _describe_transfer)
        async def upload_dir(self, source_dir, target_dir):
            return await super().upload_dir(source_dir, target_dir)

        @_timed("download_file", _describe_transfer)
        async def download_file(self, source_path, target_path):
            return await super().download_file(source_path, target_path)

        @_timed("download_dir", _describe_transfer)
        async def download_dir(self, source_dir, target_dir):
            return await super().download_dir(source_dir, target_dir)

    Profiled.__name__ = Profiled.__qualname__ = f"Profiled{inner.__name__}"
    return Profiled


def _load(import_path: str) -> type:
    module, _, name = import_path.partition(":")
    return getattr(importlib.import_module(module), name)


def __getattr__(name: str):
    # Harbor resolves IMPORT_PATH by attribute; the class wraps whatever provider this process's env names
    if name == "ProfiledEnvironment":
        return profiled(_load(inner_import_path()))
    raise AttributeError(name)


def install_proxy_profile(app, log_path: str | Path) -> None:
    """Time every request the session proxy forwards (receipt to response) into ``log_path``.

    The middleware only queues each line; one writer thread appends them, so the proxy's loop never
    waits on the file (the results volume)."""
    import queue
    import threading

    from aiohttp import web

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: queue.SimpleQueue = queue.SimpleQueue()
    done = object()

    def writer() -> None:
        while True:
            batch = [lines.get()]
            while not lines.empty():
                batch.append(lines.get())
            finished = batch[-1] is done
            write_lines(log_path, [r for r in batch if r is not done])
            if finished:
                return

    thread = threading.Thread(target=writer, name="proxy-profile-writer", daemon=True)

    @web.middleware
    async def timing(request, handler):
        t0, mono = time.time(), time.monotonic()
        status = 500
        try:
            response = await handler(request)
            status = response.status
            return response
        except web.HTTPException as e:
            status = e.status
            raise
        finally:
            info = request.match_info
            if "port" in info:  # forwarded requests only, never the token
                lines.put(
                    {
                        "t0": t0,
                        "dt": time.monotonic() - mono,
                        "status": status,
                        "path": f"/{info['port']}/{info['tail']}",
                        "method": request.method,
                    }
                )

    async def start(app):
        thread.start()

    async def stop(app):
        lines.put(done)
        await asyncio.to_thread(thread.join, 30)

    app.middlewares.append(timing)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
