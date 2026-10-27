"""Sandbox benchmark on Modal (ext-040): CPU only, no model, the production in-sandbox agent path.

Each arm is one CPU Modal function, the host, standing where the trainer's rollout stands: the scripted model
(``model_server``) listens on the first session-server port, the real ``session_proxy`` forwards to it with
``--profile-log``, and ``modal.forward`` publishes the proxy as production does. Trials go through
``miles_ext.harbor.agent_function.run`` in Miles' agent-function child processes (the recipe's
``agent.function_mode``), with the recipe's Harbor settings and ``obs.sandbox_profile`` on. Everything a run
writes goes to ``<results volume>/sbx-bench/``: never to the repo.

From a fork checkout, with Modal credentials (the image is built from a pushed fork commit):

    export MILES_FORK_COMMIT=<pushed commit> MILES_DOTENV_DIR=<dir with the .env the launcher uses>
    # once per source run: the commands of N of its trials (traces.py)
    modal run -m miles_ext.obs.sandbox_bench.app::extract --source-run overfit-1020-a6 --n 64
    # the arms; the trace workload needs the source run's tasks: MILES_BENCH_TASKS_DIR=<its taskset.tasks_dir>,
    # or MILES_BENCH_TASKS_VOLUME=<the volume> when its recipe sets taskset.tasks_volume
    MILES_BENCH_TASKS_DIR=<task dirs> modal run -m miles_ext.obs.sandbox_bench.app::main --bench-id sbxbench-1027-a1 \\
        --recipe <the source run's recipe yaml> --providers sail,modal --concurrency 1,32 --repeats 2 \\
        --workloads noop,read,write,test,fs-scan,install,trace --traces overfit-1020-a6
    modal run -m miles_ext.obs.sandbox_bench.app::report --bench-id sbxbench-1027-a1

Providers run in parallel (separate infrastructure); one provider's arms run one after another, so arms never
share a sandbox app. Sandboxes live in the app ``<sandbox.app_prefix><bench id>-<provider>``; every arm ends by
terminating what is left in it.
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import modal
from miles_ext.obs import sandbox_profile
from miles_ext.obs.sandbox_bench import report as report_lib
from miles_ext.obs.sandbox_bench import traces as traces_lib
from miles_ext.obs.sandbox_bench import workloads as W
from miles_ext.platform.images import fork_image, image_commit
from miles_ext.platform.target import Target
from miles_ext.recipes import recipe as R

HOURS = 3600
BENCH_DIR = R.RESULTS / "sbx-bench"
TRACES_DIR = BENCH_DIR / "traces"
PROBES = 20  # no-op exec calls after each sandbox start
TASKS_DIR_ENV = "MILES_BENCH_TASKS_DIR"  # replay task dirs from this machine (a recipe's taskset.tasks_dir)
TASKS_VOLUME_ENV = "MILES_BENCH_TASKS_VOLUME"  # or the volume of the recipe's taskset.tasks_volume, mounted read-only
LIFETIME_CAP_S, IDLE_CAP_S = 7200, 900  # project limits for benchmark sandboxes

app = modal.App("miles-sbx-bench")
results_volume = modal.Volume.from_name(os.environ.get("MILES_RESULTS_VOLUME") or Target().results_volume)
if modal.is_local():
    commit, image_ref = os.environ.get("MILES_FORK_COMMIT", ""), os.environ.get("MILES_IMAGE", "")
    image = fork_image(commit or None, image_ref or None) if (commit or image_ref) else modal.Image.debian_slim()
    if tasks_dir := os.environ.get(TASKS_DIR_ENV):
        image = image.add_local_dir(tasks_dir, str(R.TASKS))
    image = image.env({TASKS_VOLUME_ENV: os.environ.get(TASKS_VOLUME_ENV, "")})  # the container mounts the same
else:
    image = modal.Image.debian_slim()  # inside the container its own image and env are used
# the same objects on both sides (the container hydrates them by id, as the launcher's)
secrets = [modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR") or os.getcwd())]
tasks_volume = os.environ.get(TASKS_VOLUME_ENV, "")
arm_volumes = {str(R.RESULTS): results_volume}
if tasks_volume:
    arm_volumes[str(R.DATASETS)] = modal.Volume.from_name(tasks_volume).read_only()


def _versions() -> dict:
    out = {"commit": image_commit(), "workload_image": W.IMAGE}
    for package in ("harbor", "mini-swe-agent", "sail", "modal"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def _wait_for_port(port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"port {port} did not open within {timeout}s") from None
            time.sleep(0.5)


def _trial_specs(arm: dict, tasks_root: Path) -> list[dict]:
    """Every trial of the arm, workload by workload: session key, task, script."""
    specs = []
    cpus, memory_mb = arm["cpus"], arm["memory_mb"]
    for workload in arm["workloads"]:
        if workload == W.TRACE:
            lines = (TRACES_DIR / f"{arm['traces']}.jsonl").read_text().splitlines()
            for t in [json.loads(line) for line in lines][: arm["trace_trials"]]:
                src = Path(arm["tasks_src"]) / t["instance_id"]
                shutil.copytree(src, tasks_root / t["instance_id"], dirs_exist_ok=True)
                specs.append(
                    {
                        "workload": workload,
                        "instance_id": t["instance_id"],
                        "trace": t["trial"],
                        "commands": t["commands"],
                        "think_s": t["think_s"] if arm["think"] == "recorded" else None,
                    }
                )
        else:
            instance_id = W.write_task(tasks_root, workload, cpus, memory_mb)
            specs += [
                {
                    "workload": workload,
                    "instance_id": instance_id,
                    "trace": None,
                    "commands": W.script(workload),
                    "think_s": None,
                }
                for _ in range(arm["trials"])
            ]
    for i, s in enumerate(specs):
        s["key"] = f"{s['workload']}-{i}-{uuid.uuid4().hex[:6]}"
    return specs


async def _run_trials(arm: dict, recipe: dict, specs: list[dict], arm_dir: Path) -> None:
    from aiohttp import web
    from miles_ext.harbor import agent_function
    from miles_ext.obs.sandbox_bench.model_server import Script, make_app

    from miles.rollout.agentic.agent_function import call_agent_function

    scripts: dict = {}
    model_log: list[dict] = []
    runner = web.AppRunner(make_app(scripts, model_log))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", R.SESSION_PORT).start()
    gate = asyncio.Semaphore(arm["concurrency"])
    manifest = {"arm": arm, "versions": _versions(), "trials": []}

    async def one(spec: dict) -> None:
        async with gate:
            scripts[spec["key"]] = Script(spec["commands"], spec["think_s"])
            t0 = time.time()
            entry = {k: spec[k] for k in ("key", "workload", "instance_id", "trace")}
            try:
                out = await call_agent_function(
                    agent_function.run,
                    mode=recipe["agent"]["function_mode"],
                    base_url=f"http://127.0.0.1:{R.SESSION_PORT}/sessions/{spec['key']}",
                    prompt="",
                    request_kwargs={},
                    metadata={"instance_id": spec["instance_id"]},
                )
                entry.update(exit_status=out.get("exit_status"), trial_dir=out.get("trial_dir"))
            except Exception as e:
                entry.update(exit_status="BenchError", error=f"{type(e).__name__}: {e}"[:300])
            entry["wall_s"] = time.time() - t0
            manifest["trials"].append(entry)

    try:
        for workload in arm["workloads"]:  # one workload at a time, so workloads never contend
            batch = [s for s in specs if s["workload"] == workload]
            print(f"[{arm['name']}] {workload}: {len(batch)} trials at concurrency {arm['concurrency']}", flush=True)
            await asyncio.gather(*(one(s) for s in batch))
            await asyncio.to_thread(_save, arm_dir, manifest, model_log)
    finally:
        await runner.cleanup()


def _save(arm_dir: Path, manifest: dict, model_log: list[dict]) -> None:
    (arm_dir / report_lib.MANIFEST).write_text(json.dumps(manifest, indent=1))
    (arm_dir / "model.jsonl").write_text("".join(json.dumps(r) + "\n" for r in model_log))
    results_volume.commit()


# single_use_containers: every arm starts in a fresh container, so no arm inherits another's env or task dirs
@app.function(
    image=image,
    secrets=secrets,
    cpu=8,
    memory=32768,
    timeout=12 * HOURS,
    volumes=arm_volumes,
    single_use_containers=True,
)
def run_arm(arm: dict, recipe: dict) -> dict:
    from miles_ext.platform.teardown import sandbox_adapter

    recipe = R.from_dict(
        recipe,
        {
            "sandbox": {
                "provider": arm["provider"],
                # the sandbox app is always <app_prefix><bench id>-<provider>, never one the source recipe pins:
                # each arm terminates what is left in it
                "app_name": None,
                "lifetime_secs": min(recipe["sandbox"]["lifetime_secs"], LIFETIME_CAP_S),
                "idle_secs": min(recipe["sandbox"]["idle_secs"], IDLE_CAP_S),
            },
            "obs": {"sandbox_profile": True},
        },
    )
    run_id = f"{arm['bench_id']}-{arm['provider']}"
    arm_dir = BENCH_DIR / arm["bench_id"] / arm["name"]
    tasks_root = Path("/tmp/sbx-tasks")
    arm_dir.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    specs = _trial_specs(arm, tasks_root)
    os.environ.update(
        {
            **R.harbor_env(recipe, run_id),
            "HARBOR_TASKS_DIR": str(tasks_root),
            "HARBOR_TRIALS_DIR": str(arm_dir / "trials"),
            "MILES_EXT_HARNESS": R.HARNESS_SANDBOX,
            sandbox_profile.PROBES_ENV: str(PROBES),
        }
    )
    token = uuid.uuid4().hex
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "miles_ext.harbor.session_proxy",
            "--listen-port",
            str(R.PROXY_PORT),
            "--upstream-host",
            "127.0.0.1",
            "--ports",
            f"{R.SESSION_PORT}-{R.SESSION_PORT}",
            f"--token={token}",
            "--profile-log",
            str(arm_dir / report_lib.PROXY_LOG),
        ]
    )
    app_name = R.sandbox_app_name(recipe, run_id)
    try:
        _wait_for_port(R.PROXY_PORT)
        with modal.forward(R.PROXY_PORT) as tunnel:
            os.environ["MILES_SESSION_PUBLIC_URL"] = f"{tunnel.url}/{token}"
            asyncio.run(_run_trials(arm, recipe, specs, arm_dir))
    finally:
        proxy.terminate()
        try:
            n = sandbox_adapter(arm["provider"], None).terminate(app_name)
            print(f"[{arm['name']}] terminated {n} leftover sandboxes in {app_name}", flush=True)
        except Exception as e:
            print(f"[{arm['name']}] sandbox cleanup in {app_name}: {type(e).__name__}: {e}", flush=True)
        results_volume.commit()
    return {"arm": arm["name"], "trials": len(specs)}


@app.function(image=image, cpu=2, memory=8192, timeout=2 * HOURS, volumes={str(R.RESULTS): results_volume})
def extract_traces(source_run: str, n: int, seed: int) -> dict:
    picked = traces_lib.extract(R.RESULTS / source_run / "trials", n, seed)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    out = TRACES_DIR / f"{source_run}.jsonl"
    out.write_text("".join(json.dumps(t) + "\n" for t in picked))
    results_volume.commit()
    return {
        "path": str(out),
        "trials": len(picked),
        "tasks": len({t["instance_id"] for t in picked}),
        "commands": sum(len(t["commands"]) for t in picked),
        "bytes": out.stat().st_size,
    }


@app.function(image=image, cpu=2, memory=8192, timeout=HOURS, volumes={str(R.RESULTS): results_volume})
def build_report(bench_id: str) -> str:
    bench = BENCH_DIR / bench_id
    traces = {}
    for f in TRACES_DIR.glob("*.jsonl"):
        traces.update({t["trial"]: t for t in map(json.loads, f.read_text().splitlines())})
    arms = [report_lib.arm_report(d, traces) for d in sorted(bench.iterdir()) if (d / report_lib.MANIFEST).exists()]
    (bench / "report.json").write_text(json.dumps(arms, indent=1))
    md = report_lib.markdown(arms)
    (bench / "report.md").write_text(md)
    results_volume.commit()
    return md


@app.local_entrypoint()
def extract(source_run: str, n: int = 64, seed: int = 0, run_id: str = ""):
    """``run_id``: accepted for monitor/launch.sh, which appends it; the source run names this job."""
    print(json.dumps(extract_traces.remote(source_run, n, seed), indent=1))


@app.local_entrypoint()
def report(bench_id: str, run_id: str = ""):
    """``run_id``: accepted for monitor/launch.sh, which appends it."""
    print(build_report.remote(bench_id))


def plan_arms(
    bench_id: str,
    providers: list[str],
    concurrency: list[int],
    repeats: int,
    workloads: list[str],
    trials: int,
    trials_at_1: int,
    traces: str,
    think: str,
    cpus: int,
    memory_mb: int,
) -> list[dict]:
    """Every arm, each provider's in the order they run (concurrency order flips each repeat)."""
    arms = []
    for provider in providers:
        for rep in range(repeats):
            for conc in concurrency if rep % 2 == 0 else concurrency[::-1]:
                n = trials if conc > 1 else trials_at_1
                arms.append(
                    {
                        "bench_id": bench_id,
                        "name": f"{provider}-c{conc}-r{rep}",
                        "provider": provider,
                        "concurrency": conc,
                        "repeat": rep,
                        "workloads": workloads,
                        "trials": n,
                        "trace_trials": None if conc > 1 else n,
                        "traces": traces,
                        "think": think,
                        "cpus": cpus,
                        "memory_mb": memory_mb,
                    }
                )
    return arms


@app.local_entrypoint()
def main(
    recipe: str,
    bench_id: str = "",
    providers: str = "sail,modal",
    concurrency: str = "1,32",
    repeats: int = 2,
    workloads: str = "noop,read,write,test,fs-scan,install,trace",
    trials: int = 32,
    trials_at_1: int = 8,
    traces: str = "",
    think: str = "zero",
    cpus: int = 2,
    memory_mb: int = 4096,
    run_id: str = "",
):
    """``run_id`` (monitor/launch.sh appends it) is the bench id when ``--bench-id`` is not given."""
    bench_id = bench_id or run_id
    if not bench_id:
        raise SystemExit("give --bench-id (or --run-id)")
    names = [w for w in workloads.split(",") if w]
    unknown = set(names) - set(W.SYNTHETIC) - {W.TRACE}
    if unknown:
        raise SystemExit(f"unknown workloads {sorted(unknown)}; known: {sorted(W.SYNTHETIC)} and {W.TRACE}")
    if think not in ("zero", "recorded"):
        raise SystemExit("--think is zero or recorded")
    resolved = R.load(recipe)
    tasks_src = ""
    if W.TRACE in names:
        if not traces:
            raise SystemExit("the trace workload needs --traces <source run>")
        if volume := R.tasks_volume(resolved):  # the recipe's tasks live on a volume: read them there
            if tasks_volume != volume[0]:
                raise SystemExit(
                    f"the recipe's tasks are on volume {volume[0]}: export {TASKS_VOLUME_ENV}={volume[0]}"
                )
            tasks_src = str(R.tasks_path(resolved))
        elif os.environ.get(TASKS_DIR_ENV):
            tasks_src = str(R.TASKS)
        else:
            raise SystemExit(
                f"the trace workload needs the source run's tasks: {TASKS_DIR_ENV} or a tasks_volume recipe"
            )
    arms = plan_arms(
        bench_id,
        providers.split(","),
        [int(c) for c in concurrency.split(",")],
        repeats,
        names,
        trials,
        trials_at_1,
        traces,
        think,
        cpus,
        memory_mb,
    )
    for arm in arms:
        arm["tasks_src"] = tasks_src
    by_provider: dict[str, list[dict]] = {}
    for arm in arms:
        by_provider.setdefault(arm["provider"], []).append(arm)

    def run_provider(provider_arms: list[dict]) -> list[dict]:
        return [run_arm.remote(arm, resolved) for arm in provider_arms]

    with ThreadPoolExecutor(len(by_provider)) as pool:
        for done in pool.map(run_provider, by_provider.values()):
            print(json.dumps(done))
    print(build_report.remote(bench_id))
