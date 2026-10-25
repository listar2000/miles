"""Serve a split run's engines, started with Miles' own launch command, and its sync agent behind one tunnel."""

import json
import secrets
import shlex
import subprocess
import sys
import time
import urllib.request

import modal

from miles_ext import stack
from miles_ext.launch import split
from miles_ext.recipes import recipe as R

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
        input=json.dumps({"argv": argv, "http_port": split.ENGINE_HTTP_PORT}),
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


def serve(recipe: dict, run_id: str, *, progress, env: dict, results_volume, call_id: str | None) -> dict:
    """Start the engines, publish the tunnel, serve until the trainer is done or gone. Returns the end state."""
    from miles_ext.launch.split.agent import Agent

    key, idle_secs = split.engine_key(run_id), int(recipe["cluster"]["engine"]["idle_minutes"]) * 60
    state = {"run_id": run_id, "call_id": call_id, "phase": "starting", "started_at": time.time()}
    progress[key] = state
    spec = commands(recipe, run_id, env)
    engine_env = {**env, **spec["env"]}
    ports = [split.ENGINE_HTTP_PORT + i for i in range(len(spec["commands"]))]
    agent_port = split.ENGINE_HTTP_PORT + len(ports)
    procs = []
    try:
        for cmd in spec["commands"]:
            print(f"engine: {cmd}", flush=True)
            procs.append(subprocess.Popen(shlex.split(cmd), env=engine_env, cwd=str(stack.MILES_SRC)))
        _wait_healthy(ports, procs)
        agent = Agent(ports, gpus_per_engine=spec["gpus_per_engine"], results_volume=results_volume)
        agent.start(agent_port)
        token = secrets.token_urlsafe(24)
        proxy = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles_ext.harbor.session_proxy",
                "--listen-port",
                str(split.PROXY_PORT),
                "--upstream-host",
                "127.0.0.1",
                "--ports",
                f"{ports[0]}-{agent_port}",
                f"--token={token}",
            ]
        )
        procs.append(proxy)
        with modal.forward(split.PROXY_PORT) as tunnel:
            state = {
                **state,
                "phase": "ready",
                "ready_at": time.time(),
                "url": tunnel.url,
                "token": token,
                "engine_ports": ports,
                "agent_port": agent_port,
            }
            progress[key] = state
            print(f"engines ready: {len(ports)} at {tunnel.url}/<token>/<port>, agent port {agent_port}", flush=True)
            state["phase"], state["reason"] = "ended", _serve_until_released(run_id, progress, procs, idle_secs)
    except BaseException as e:
        state = {**state, "phase": "ended", "reason": f"error: {type(e).__name__}: {e}"[:500]}
        raise
    finally:
        state["ended_at"] = time.time()
        progress[key] = state
        for p in procs:
            p.terminate()
        print(f"engine container: {state.get('reason')}", flush=True)
    return state


def _serve_until_released(run_id: str, progress, procs: list[subprocess.Popen], idle_secs: int) -> str:
    """Block while the engines are up and a trainer is (or recently was) there."""
    since = time.time()
    while True:
        if dead := [p.args for p in procs if p.poll() is not None]:
            raise RuntimeError(f"an engine or the proxy exited: {dead}")
        trainer = progress.get(split.trainer_key(run_id)) or {}
        if trainer.get("done"):
            return "trainer done"
        seen = max(since, float(trainer.get("heartbeat_at") or 0))
        if time.time() - seen > idle_secs:
            return f"no trainer for {idle_secs // 60} min"
        time.sleep(POLL_SECS)
