"""Modal app ``miles-taskfilter``: one oracle + install-probe trial per task on Sail, launched as training does.

    modal run -m miles_ext.taskfilter.sweep --check-only               # imports, trial config, one task fetch
    modal run --detach -m miles_ext.taskfilter.sweep --tasks t.jsonl --run full1

``--tasks`` is a JSONL of ``{"id", "version"}`` task versions of a Mobius
taskset. Each trial runs in its own small CPU container of the trainer image
(``IMAGE_REF``): the Harbor environment comes from the recipe
(``recipe.harbor_env``, Sail provider ``miles_ext.harbor.sail``) and the trial
config from the trainer's own ``agent_function.build_trial_config``; only the
agent (``probe_agent.ProbeOracleAgent``) and the timeouts in ``RECIPE`` differ
from a training rollout. The container fetches its task version through the
Mobius API (``mobius.fetch_task``) onto its ephemeral disk and deletes it after the trial; nothing is
kept elsewhere. Output, on Volume ``miles-fork-results``:

    taskfilter/sweep-<run>/summary.jsonl          one ``summary.summarize`` row per task
    taskfilter/sweep-<run>/trials/<task>/...      summary.json, result.json, install_probe.json, logs

A rerun with the same ``--run`` skips tasks that already have a summary.json.
Run from the launch folder (Sail's key comes from its ``.env``, as Run 4); the
Mobius token from ``MOBIUS_TOKEN`` or the mobius CLI's saved credentials.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

from miles_ext import stack

APP = "miles-taskfilter"
# the image training runs (main's newest build); TASKFILTER_IMAGE pins another, e.g. <repo>:<sha10>
IMAGE_REF = os.environ.get("TASKFILTER_IMAGE") or f"{stack.IMAGE_REPO}:{stack.LATEST_TAG}"
MILES = Path("/root/miles")
RESULTS = Path("/results")
WORK = Path("/tmp/tf")
CONCURRENCY = int(os.environ.get("TASKFILTER_CONCURRENCY", "450"))

# The recipe a training run on these tasks would use; only the caps differ from a rollout's.
RECIPE = {
    "taskset": {"harness": "mini-swe-agent-sandbox"},
    "agent": {"timeout": 3600, "verifier_timeout": 1800, "setup_timeout": 1800, "cwd": "/app",
              "trial_timeout": 7500},
    "sandbox": {"provider": "sail", "lifetime_secs": 7200, "idle_secs": 900},
}
INSTALL_S = 1500  # inside setup_timeout, so a slow install is recorded instead of failing setup
KEEP_MAX_BYTES = 2_000_000  # per copied trial file


def _local_secrets() -> list:
    """The Sail key (from the launch folder's .env or the environment) and the Mobius login, nothing else."""
    if not modal.is_local():  # the container hydrates these by id; the count must match the local list
        return [modal.Secret.from_dict({}), modal.Secret.from_dict({})]
    from miles_ext.taskfilter.mobius import credentials

    dotenv = Path(os.environ.get("TASKFILTER_DOTENV_DIR", Path.cwd())) / ".env"
    sail_key = os.environ.get("SAIL_API_KEY")
    for line in dotenv.read_text().splitlines() if not sail_key and dotenv.is_file() else []:
        key, _, value = line.strip().removeprefix("export ").partition("=")
        if key.strip() == "SAIL_API_KEY":
            sail_key = value.strip().strip("'\"")
    if not sail_key:
        raise SystemExit(f"no SAIL_API_KEY in the environment or {dotenv}")
    token, api = credentials()
    return [modal.Secret.from_dict({"SAIL_API_KEY": sail_key}),
            modal.Secret.from_dict({"MOBIUS_TOKEN": token, "MOBIUS_API_URL": api})]


image = (
    modal.Image.from_registry(IMAGE_REF)
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1", "MSWEA_SILENT_STARTUP": "1", "MSWEA_COST_TRACKING": "ignore_errors"})
    # the only code not in the image: this package
    .add_local_dir(Path(__file__).resolve().parent, str(MILES / "miles_ext" / "taskfilter"),
                   ignore=["__pycache__"])
)
app = modal.App(APP)
VOLUME_NAME = "miles-fork-results"
volume = modal.Volume.from_name(VOLUME_NAME)
secrets = _local_secrets()


def run_id(run: str) -> str:
    return f"taskfilter-{run}"  # Sail app: recipe.sandbox_app_name -> miles-mobius-taskfilter-<run>


def fetch_task(task_id: str, version: str, tasks_dir: Path) -> Path:
    from miles_ext.taskfilter.mobius import fetch_task as fetch

    return fetch(task_id, version, tasks_dir, os.environ["MOBIUS_TOKEN"], os.environ["MOBIUS_API_URL"])


def training_trial_config(task_id: str, run: str, tasks_dir: Path, trials_dir: Path):
    """The trial config a training rollout would build for this task, with the probe agent swapped in."""
    sys.path[:0] = [str(MILES)]
    from miles_ext.recipes import recipe as R

    env = R.harbor_env(R.from_dict(RECIPE), run_id(run))
    env.update({"HARBOR_TASKS_DIR": str(tasks_dir), "HARBOR_TRIALS_DIR": str(trials_dir),
                "MILES_SESSION_PUBLIC_URL": "https://taskfilter.modal.run"})  # a tunnel-like host; no model is called
    os.environ.update(env)
    from miles_ext.harbor.agent_function import build_trial_config

    config = build_trial_config({"instance_id": task_id}, "http://127.0.0.1:30000/sessions/taskfilter", {})
    rollout_agent = config.agent.model_dump(mode="json", exclude={"env"})
    config.agent.name = None
    config.agent.model_name = None
    config.agent.env = {}
    config.agent.import_path = "miles_ext.taskfilter.probe_agent:ProbeOracleAgent"
    config.agent.kwargs = {"task_dir": str(config.task.path), "agent_timeout_sec": RECIPE["agent"]["timeout"],
                           "install_timeout_sec": INSTALL_S}
    return config, rollout_agent, {k: v for k, v in env.items() if k.startswith(("HARBOR_", "AGENT_", "MSWEA_"))}


def _copy_small(trial_dir: Path, out: Path) -> None:
    for f in trial_dir.rglob("*"):
        if f.is_file() and f.stat().st_size <= KEEP_MAX_BYTES:
            dest = out / f.relative_to(trial_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)


@app.function(image=image, secrets=secrets, volumes={RESULTS: volume}, cpu=1, memory=4096,
              timeout=RECIPE["agent"]["trial_timeout"] + 1800, max_containers=CONCURRENCY, retries=0)
def trial(task: dict, run: str) -> dict:
    import asyncio

    sys.path[:0] = [str(MILES)]
    from harbor.trial.trial import Trial

    from miles_ext.taskfilter.probe_agent import PROBE_FILE
    from miles_ext.taskfilter.summary import summarize

    task_id, version = task["id"], task["version"]
    work = WORK / task_id
    out = RESULTS / "taskfilter" / f"sweep-{run}" / "trials" / task_id
    result, probe, error, started = None, None, None, time.time()
    try:
        fetch_task(task_id, version, work / "tasks")
        config, _, _ = training_trial_config(task_id, run, work / "tasks", work / "trials")

        async def go():
            t = await Trial.create(config)
            try:
                await asyncio.wait_for(t.run(), timeout=RECIPE["agent"]["trial_timeout"])
            except asyncio.TimeoutError:
                pass  # the partial result.json still tells which phase it reached
            return t.paths.trial_dir

        trial_dir = asyncio.run(go())
        if (trial_dir / "result.json").is_file():
            result = json.loads((trial_dir / "result.json").read_text())
        if (trial_dir / "agent" / PROBE_FILE).is_file():
            probe = json.loads((trial_dir / "agent" / PROBE_FILE).read_text())
        _copy_small(trial_dir, out)
        if result is None:
            error = f"no result.json after {round(time.time() - started)}s"
    except Exception as e:
        error = f"runner: {type(e).__name__}: {str(e)[:400]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    row = summarize(task_id, version, result, probe, RECIPE["agent"]["verifier_timeout"], error=error)
    row["wall_s"] = round(time.time() - started, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(row))
    volume.commit()
    return row


def _terminate_sail_app(run: str) -> int:
    sys.path[:0] = [str(MILES)]
    from miles_ext.platform.teardown import SailSandboxes
    from miles_ext.recipes import recipe as R

    return SailSandboxes().terminate(R.sandbox_app_name(R.from_dict(RECIPE), run_id(run)))


@app.function(image=image, secrets=secrets, volumes={RESULTS: volume}, cpu=1, memory=4096, timeout=24 * 3600)
def sweep(tasks: list[dict], run: str, concurrency: int, alert_after_same_failures: int = 8) -> dict:
    """Map ``trial`` over the tasks not done yet. When the first trials all fail the same way it prints an
    ALERT line and keeps going: whoever watches the run decides whether to stop it (the watchdog informs)."""
    from collections import Counter

    base = RESULTS / "taskfilter" / f"sweep-{run}"
    base.mkdir(parents=True, exist_ok=True)
    volume.reload()
    todo = [t for t in tasks if not (base / "trials" / t["id"] / "summary.json").is_file()]
    (base / f"launch-{int(time.time())}.json").write_text(json.dumps(
        {"recipe": RECIPE, "install_s": INSTALL_S, "concurrency": concurrency, "image": IMAGE_REF,
         "tasks": len(tasks), "todo": len(todo)}))
    print(f"sweep {run}: {len(todo)} of {len(tasks)} tasks to run, concurrency {concurrency}", flush=True)
    rows, alert = [], None
    try:
        with (base / "summary.jsonl").open("a") as f:
            for row in trial.map(todo, kwargs={"run": run}, order_outputs=False, return_exceptions=True):
                if isinstance(row, BaseException):
                    row = {"task_id": None, "oracle_exception": f"function: {type(row).__name__}: {str(row)[:300]}"}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                if len(rows) % 10 == 0:
                    volume.commit()
                print(json.dumps(row), flush=True)
                if len(rows) == alert_after_same_failures and all(r.get("env_build_s") is None for r in rows):
                    kinds = Counter(str(r.get("oracle_exception"))[:60] for r in rows)
                    if len(kinds) == 1:
                        alert = f"ALERT first {len(rows)} trials failed alike: {next(iter(kinds))}"
                        (base / "ALERT").write_text(alert)
                        print(alert, flush=True)
    finally:
        try:
            print(f"terminated {_terminate_sail_app(run)} Sailboxes", flush=True)
        except Exception as e:
            print(f"Sail cleanup failed: {type(e).__name__}: {e}", flush=True)
        volume.commit()
    return {"done": len(rows), "alert": alert}


@app.function(image=image, secrets=secrets, cpu=1, memory=4096, timeout=1800)
def check(task_id: str = "atrx-vep-crispr-09-datum-grid-reconciliation", version: str = "19dba45e8857") -> dict:
    """Dry check, no sandbox: the trial config training would build, the probe agent loads, Sail key, a task fetch."""
    sys.path[:0] = [str(MILES)]
    from harbor.agents.factory import AgentFactory
    from sail import App

    root = WORK / "check"
    fetch_task(task_id, version, root / "tasks")
    config, rollout_agent, env = training_trial_config(task_id, "check", root / "tasks", root / "trials")
    agent = AgentFactory.create_agent_from_config(config.agent, logs_dir=root / "trial" / "agent")
    out = {"probe_agent": agent.name(), "rollout_agent": rollout_agent, "harbor_env": env,
           "trial_config": config.model_dump(mode="json", exclude={"agent": {"kwargs"}}),
           "sail_apps_visible": sum(1 for _ in App.list()),
           "task_files": sorted(p.name for p in (root / "tasks" / task_id).iterdir())}
    shutil.rmtree(root, ignore_errors=True)
    return out


@app.local_entrypoint()
def main(tasks: str = "", run: str = "", check_only: bool = False):
    if check_only:
        print(json.dumps(check.remote(), indent=1, default=str))
        return
    rows = [json.loads(line) for line in Path(tasks).read_text().splitlines() if line.strip()]
    if not run or not all(c.isalnum() or c in "._-" for c in run):
        raise SystemExit(f"bad --run {run!r}")
    print(f"sweep {run}: {len(rows)} tasks, concurrency {CONCURRENCY}, recipe {RECIPE}")
    # spawn, not remote: killing a client that waits on .remote cancels the sweep even under --detach,
    # and the sweep's cleanup then terminates every in-flight Sailbox. Run with --detach.
    call = sweep.spawn([{"id": r["id"], "version": r["version"]} for r in rows], run, CONCURRENCY)
    print(f"spawned sweep {run}: {call.object_id}; rows go to Volume {VOLUME_NAME}:/taskfilter/sweep-{run}/")
