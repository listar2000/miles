"""The rollout role of a split run: engines and a sync agent behind one token, joined to the trainer (ext-063)."""

import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from miles_ext import stack
from miles_ext.node import contract as C
from miles_ext.node import registry, runtime
from miles_ext.node import storage as node_storage
from miles_ext.platform import compile_cache
from miles_ext.recipes import recipe as R

ENGINE_HTTP_PORT = 32000  # engine i serves HTTP on ENGINE_HTTP_PORT + i inside the node
STARTUP_TIMEOUT_SECS = 3 * 3600  # weights from the volume plus CUDA graphs
POLL_SECS = 10

_COMMANDS = r"""
import json, sys
spec = json.loads(sys.stdin.read())
sys.argv = spec["argv"]
from miles.utils.arguments import parse_args
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.specs.inference import compute_inference_engine_env_vars
args = parse_args()
gpus = args.rollout_num_gpus_per_engine
engines = []
for i in range(args.rollout_num_gpus // gpus):
    # as miles.ray.specs.inference does for a one-node engine, with this container's own ports and no
    # launch gate (nothing here holds an engine back until its peers are placed)
    engines.append(compute_engine_launch_cmd(
        args, interpreter_prefix=[sys.executable], node_rank=0, worker_type=WorkerType.REGULAR,
        base_gpu_id=i * gpus, sglang_overrides={}, num_gpus_per_engine=gpus,
        dist_init_addr=f"127.0.0.1:{34000 + 100 * i}", nccl_port=33000 + 10 * i, host="127.0.0.1",
        port=spec["http_port"] + i, disaggregation_bootstrap_port=None, engine_info_bootstrap_port=35000 + i,
        gated_launch_port=None, random_seed=args.seed + i))
print("ENGINES " + json.dumps({"commands": engines, "env": compute_inference_engine_env_vars(args),
                               "gpus_per_engine": gpus}))
"""


def commands(recipe: dict, run_id: str, env: dict) -> dict:
    """Miles' engine launch commands and engine env for the run (the argv a trainer gets, parsed once)."""
    argv = [R.entrypoint(recipe), *R.miles_args(recipe, run_id, resume=False)]
    proc = subprocess.run(
        [sys.executable, "-c", _COMMANDS],
        input=json.dumps({"argv": argv, "http_port": ENGINE_HTTP_PORT}),
        capture_output=True,
        text=True,
        cwd=str(stack.MILES_SRC),
        env=env,
        timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("ENGINES "):
            return json.loads(line[len("ENGINES ") :])
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-20:])
    raise RuntimeError(f"engine commands: Miles parse_args failed (exit {proc.returncode}):\n{tail}")


def _start_engine(cmd: str, env: dict) -> subprocess.Popen:
    """Start an engine and relay its output tagged with an identity the run monitor tells apart."""
    proc = subprocess.Popen(
        shlex.split(cmd),
        env=env,
        cwd=str(stack.MILES_SRC),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    threading.Thread(target=_relay_lines, args=(proc,), daemon=True).start()
    return proc


def _relay_lines(proc: subprocess.Popen):
    tag = f"(SplitEngine pid={proc.pid}) "
    for line in proc.stdout:
        sys.stdout.write(tag + line)
        sys.stdout.flush()


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def _wait_healthy(ports: list[int], procs: list[subprocess.Popen]):
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECS
    pending = set(ports)
    while pending:
        if dead := [p.args for p in procs if p.poll() is not None]:
            raise RuntimeError(f"an engine exited during startup: {dead}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"engines on ports {sorted(pending)} not healthy after {STARTUP_TIMEOUT_SECS}s")
        pending = {p for p in pending if not _healthy(p)}
        time.sleep(POLL_SECS)


def trainer_url(env: dict) -> str | None:
    """The registry's URL from the caller: fixed, or in a file it rewrites when a relaunched trainer moves."""
    if url := env.get(C.TRAINER_URL_ENV):
        return url
    try:
        return Path(env[C.TRAINER_URL_FILE_ENV]).read_text().strip() or None
    except (KeyError, FileNotFoundError):
        return None


def serve(recipe: dict, run_id: str, *, env: dict, storage, on_state: Callable[[dict], None] | None = None) -> dict:
    """Start the engines, join the trainer's registry, serve until the trainer is done or gone. Returns the end state.
    ``on_state`` lets a caller watch the phases (starting, ready, ended) without reading anything from the node."""
    from miles_ext.launch.split.agent import Agent

    idle_secs = int(recipe["cluster"]["engine"]["idle_minutes"]) * 60
    public_url, pool_token = env[C.ROLLOUT_URL_ENV], env[C.POOL_TOKEN_ENV]
    notify = on_state or (lambda state: None)
    now = time.time()
    state = {"run_id": run_id, "phase": "starting", "started_at": now, "container_at": now}
    notify(state)
    spec = commands(recipe, run_id, env)
    engine_env = {**env, **spec["env"]}
    ports = [ENGINE_HTTP_PORT + i for i in range(len(spec["commands"]))]
    agent_port = ENGINE_HTTP_PORT + len(ports)
    procs, agent = [], None
    try:
        for cmd in spec["commands"]:
            print(f"engine: {cmd}", flush=True)
            procs.append(_start_engine(cmd, engine_env))
        _wait_healthy(ports, procs)
        agent = Agent(ports, gpus_per_engine=spec["gpus_per_engine"], storage=storage)
        agent.start(agent_port)
        token = secrets.token_urlsafe(24)
        proxy = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles_ext.harbor.session_proxy",
                "--listen-port",
                str(C.ROLLOUT_PROXY_PORT),
                "--upstream-host",
                "127.0.0.1",
                "--ports",
                f"{ports[0]}-{agent_port}",
                f"--token={token}",
            ]
        )
        procs.append(proxy)
        member = {
            "node_id": f"{socket.gethostname()}-{secrets.token_hex(4)}",
            "url": public_url.rstrip("/"),
            "token": token,
            "engine_ports": ports,
            "agent_port": agent_port,
            "gpus_per_engine": spec["gpus_per_engine"],
        }
        state = {**state, "phase": "ready", "ready_at": time.time(), "url": member["url"]}
        notify(state)
        print(f"engines ready: {len(ports)} at {member['url']}/<token>/<port>, agent port {agent_port}", flush=True)
        state["phase"], state["reason"] = "ended", _serve_until_released(
            member, env, agent, procs, pool_token, idle_secs
        )
    except BaseException as e:
        state = {**state, "phase": "ended", "reason": f"error: {type(e).__name__}: {e}"[:500]}
        raise
    finally:
        state["ended_at"] = time.time()
        notify(state)
        for p in procs:
            p.terminate()
        if agent is not None:
            agent.stop()
        print(f"rollout node: {state.get('reason')}", flush=True)
    return state


def _load(ports: list[int]) -> list[dict]:
    """Each engine's running and waiting requests, as SGLang reports them, for a caller's scaling decision."""
    out = []
    for port in ports:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=5) as r:
                states = json.loads(r.read()).get("internal_states") or [{}]
            out.append(
                {
                    "port": port,
                    "running": states[0].get("num_running_reqs"),
                    "waiting": states[0].get("num_waiting_reqs"),
                }
            )
        except (OSError, ValueError):
            out.append({"port": port, "running": None, "waiting": None})
    return out


def _serve_until_released(
    member: dict, env: dict, agent, procs: list[subprocess.Popen], pool_token: str, idle_secs: int
) -> str:
    """Block while the engines are up and a trainer is (or recently was) there."""
    seen = time.time()
    last_url, next_beat = None, 0.0
    while True:
        if dead := [p.args for p in procs if p.poll() is not None]:
            raise RuntimeError(f"an engine or the proxy exited: {dead}")
        if agent.released.is_set():
            return "trainer done"
        now, url = time.time(), trainer_url(env)
        if url and (url != last_url or now >= next_beat):
            next_beat = now + registry.BEAT_SECS
            try:
                reply = registry.beat(url, pool_token, {**member, "load": _load(member["engine_ports"])})
            except registry.UNREACHABLE as e:  # the trainer is gone or not up yet: the idle limit decides
                print(f"rollout node: beat to {url}: {type(e).__name__}: {e}", flush=True)
            else:
                seen, last_url = now, url
                if reply.get("done"):
                    return "trainer done"
                if reply.get("evicted"):
                    return "evicted by the trainer"
        if now - seen > idle_secs:
            return f"no trainer for {idle_secs // 60} min"
        time.sleep(POLL_SECS)


def run(recipe: dict, run_id: str, *, on_state: Callable[[dict], None] | None = None) -> dict:
    """The role: everything a rollout node does from start to end, read from the contract's env."""
    gpu = recipe["cluster"]["gpu"]
    node_storage.for_mount(R.MODELS).refresh()
    cache_key = runtime.compile_cache_key(gpu)
    compile_cache.restore(cache_key)
    saver = compile_cache.Saver(cache_key, node_storage.for_mount(compile_cache.MOUNT).commit).start()
    env = {**os.environ, **compile_cache.env(), **R.trainer_env(recipe), "PYTHONPATH": runtime.pythonpath()}
    try:
        return serve(recipe, run_id, env=env, storage=node_storage.for_mount(R.RESULTS), on_state=on_state)
    finally:
        saver.stop()
