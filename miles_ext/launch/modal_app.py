"""The trainer function (one segment of a run on a Modal cluster), and a CLI to run it.

``miles_ext.platform.launch`` runs every platform segment through ``train`` here;
the CLI below runs a recipe by hand to validate a fork release. The image already
holds Miles (our branch), Harbor and mini-swe-agent; this module only adds the
run inputs (tasks, prompts) and the cluster wiring. ``train`` splits into rank 0
(``_head``: Ray head, Miles under the supervisor) and the other ranks (``_worker``).

    export MILES_DATA=<dir with the task sets>  MILES_RECIPE=miles_ext/recipes/configs/<name>.yaml
    modal run -m miles_ext.launch.modal_app --run-id <id> --check-only    # dry check in the image
    modal run --detach -m miles_ext.launch.modal_app --run-id <id>        # train (same id again resumes)

Credentials come from the .env in ``MILES_DOTENV_DIR`` (default: cwd) as a
Modal Secret, or from the per-run secret named by ``MILES_SECRET_NAME``
(``miles_ext.platform.secrets``). Names are this fork's own, and the platform's
dev setup overrides them through env (a ``miles_ext.platform.Target``):
app ``miles-fork`` (``MILES_APP_NAME``), volumes ``miles-fork-results``
(``MILES_RESULTS_VOLUME``) and ``miles-swe-models`` (``MILES_WEIGHTS_VOLUME``).
Final checkpoint retention runs after a successful run unless a per-run secret is
used (platform runs retain themselves); ``MILES_FINAL_RETENTION=1/0`` overrides.
Other names:
queue ``miles-fork-completion``, sandbox app (Modal or
Sail, per the recipe's sandbox.provider) ``<sandbox.app_prefix><run_id>``,
default ``miles-mobius-<run_id>`` (``miles-fork-sbx-<run_id>`` before 2026-10-15;
exit cleanup checks both).

``miles_ext.platform.launch`` runs this module in a child process with the whole
configuration in ``MILES_EXT_LAUNCH`` (recipe, run id, target), because Modal binds
the cluster shape when the function is defined; nothing here reads that variable
unless it is set.
"""

import contextlib
import json
import os
import resource
import socket
import subprocess
import threading
import time

import modal

from miles_ext import stack
from miles_ext.platform.checkpoints import apply_retention_local
from miles_ext.platform.images import fork_image
from miles_ext.platform.layout import read_tracker
from miles_ext.platform.target import Target
from miles_ext.platform.teardown import sandbox_adapter
from miles_ext.recipes import recipe as R

_SHAPE_ENV = "MILES_EXT_SHAPE"
_LAUNCH_ENV = "MILES_EXT_LAUNCH"  # set by miles_ext.platform.launch only
_INPUTS_ENV = "MILES_EXT_INPUTS"  # "volume": the taskset is at <results>/<run_id>/inputs, not in the image
DEFAULT_RECIPE = "miles_ext/recipes/configs/qwen35-9b-swegym8-sandbox.yaml"
LAUNCH = json.loads(os.environ[_LAUNCH_ENV]) if modal.is_local() and os.environ.get(_LAUNCH_ENV) else None

if modal.is_local():
    RECIPE = LAUNCH["recipe"] if LAUNCH else R.load(os.environ.get("MILES_RECIPE") or DEFAULT_RECIPE)
    SHAPE = R.shape(RECIPE)
    # MILES_FORK_COMMIT=<sha>: build the same image on Modal from that pushed fork
    # commit instead of pulling FORK_IMAGE (when no registry push is possible)
    FROM_COMMIT = (LAUNCH or {}).get("from_commit") or os.environ.get("MILES_FORK_COMMIT", "")
    if not stack.FORK_IMAGE and not FROM_COMMIT:
        raise RuntimeError("miles_ext/stack.py: FORK_IMAGE is empty; publish it or set MILES_FORK_COMMIT")
else:
    RECIPE = None
    SHAPE = json.loads(os.environ[_SHAPE_ENV])

HOURS = 3600
# Upstream's launchers put the Megatron-LM checkout on PYTHONPATH (command_utils'
# ray backend); its editable install exposes megatron.core but not every package
# Megatron training imports (megatron.post_training), and the image's
# PYTHONPATH=/root/miles would otherwise hide it
MEGATRON_SRC = "/root/Megatron-LM"


def _pythonpath() -> str:
    return ":".join(p for p in (os.environ.get("PYTHONPATH"), MEGATRON_SRC) if p)

# the target's names; inside a container they come back through the image env (_TARGET_ENV)
_TARGET_ENV = "MILES_EXT_TARGET"
if LAUNCH:
    TARGET = Target.of(LAUNCH["target"])
elif modal.is_local():
    TARGET = Target.of({k: v for k, v in {
        "app_name": os.environ.get("MILES_APP_NAME"),
        "results_volume": os.environ.get("MILES_RESULTS_VOLUME"),
        "weights_volume": os.environ.get("MILES_WEIGHTS_VOLUME"),
        "secret_name": os.environ.get("MILES_SECRET_NAME"),
        "modal_environment": os.environ.get("MODAL_ENVIRONMENT"),
    }.items() if v})
else:
    TARGET = Target.of(json.loads(os.environ[_TARGET_ENV]))
# Final checkpoint retention after a successful run, for standalone use. A platform
# caller (one that passes a per-run secret) runs finalize_hf and its own retention,
# with its pinned steps, so the launcher leaves the checkpoints alone.
# MILES_FINAL_RETENTION=1/0 overrides.
_RETENTION_ENV = "MILES_FINAL_RETENTION"
if LAUNCH:
    FINAL_RETENTION = bool(LAUNCH.get("final_retention"))
elif modal.is_local():
    FINAL_RETENTION = os.environ.get(_RETENTION_ENV, "0" if TARGET.secret_name else "1") == "1"
else:
    FINAL_RETENTION = os.environ.get(_RETENTION_ENV) == "1"

app = modal.App(TARGET.app_name)
models_volume = modal.Volume.from_name(TARGET.weights_volume, create_if_missing=True)
results_volume = modal.Volume.from_name(TARGET.results_volume, create_if_missing=True, version=2)
if TARGET.secret_name:
    secrets = [modal.Secret.from_name(TARGET.secret_name)]
else:
    secrets = [modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR") or os.getcwd())]
completion_queue = modal.Queue.from_name("miles-fork-completion", create_if_missing=True)
WORKER_COMMIT_SECS = 120

def _image_ref() -> str | None:
    """The registry image this container runs, or None when Modal built it from a commit."""
    return None if os.environ.get("MILES_FORK_RELEASE", "").startswith("commit-") else (stack.FORK_IMAGE or None)


if RECIPE is None:
    image = modal.Image.debian_slim()  # placeholder: inside the container the function's own image is used
else:
    image = fork_image(FROM_COMMIT or None)
image = image.entrypoint([]).env({"PYTHONUNBUFFERED": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
                                  _SHAPE_ENV: json.dumps(SHAPE, sort_keys=True),
                                  _TARGET_ENV: json.dumps(TARGET.as_dict(), sort_keys=True),
                                  _RETENTION_ENV: "1" if FINAL_RETENTION else "0"})
INPUTS_ON_VOLUME = ((LAUNCH or {}).get("inputs") == "volume") if modal.is_local() else \
    os.environ.get(_INPUTS_ENV) == "volume"
image = image.env({_INPUTS_ENV: "volume" if INPUTS_ON_VOLUME else "image"})
if RECIPE is not None and not INPUTS_ON_VOLUME:
    image = image.add_local_dir(RECIPE["taskset"]["tasks_dir"], str(R.TASKS)).add_local_file(
        RECIPE["taskset"]["prompts"], str(R.PROMPTS)
    )


def _image_commit() -> str:
    # set by fork/docker/build.py; /root/miles/.git still describes the upstream checkout under our layer
    return os.environ.get("MILES_FORK_COMMIT", "unknown")


def _wait_for_port(host: str, port: int, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{host}:{port} did not open within {timeout}s")
            time.sleep(1)


def _terminate_task_sandboxes(recipe: dict, run_id: str):
    """Episodes in flight when the trainer exits never reach Harbor's cleanup; the sandbox app is this run's alone."""
    for name in R.sandbox_app_names_for_cleanup(recipe, run_id):
        _terminate_app_sandboxes(recipe, name)


def _terminate_app_sandboxes(recipe: dict, name: str):
    provider = recipe["sandbox"]["provider"]
    try:
        R.check_own_app_name(name)
        n = sandbox_adapter(provider, TARGET.modal_environment).terminate(name)  # exact name, never created
        print(f"terminated {n} {provider} sandboxes in {name}", flush=True)
    except Exception as e:
        print(f"sandbox cleanup ({provider} {name}): {type(e).__name__}: {e}", flush=True)


def _check_shape(recipe: dict):
    if R.shape(recipe) != SHAPE:
        raise RuntimeError(f"recipe shape {R.shape(recipe)} != the function's {SHAPE}; relaunch with the same recipe")


@app.function(image=image, secrets=secrets, volumes={R.MODELS: models_volume}, gpu="H100", cpu=4, memory=32768,
              timeout=1800)
def check(recipe: dict, run_id: str):
    """Dry run inside the fork image: the platform's deep check (miles_ext.platform.validate) on one GPU."""
    from miles_ext.platform.validate import run_deep_check

    _check_shape(recipe)
    print(f"image commit {_image_commit()}, stack {os.environ.get('MILES_FORK_RELEASE')}")
    result = run_deep_check(recipe, run_id)
    print(json.dumps({k: result[k] for k in ("trial_config", "parsed")}, indent=1, default=str))
    if R.backend(recipe) == "megatron":
        print("megatron layout:", R.megatron_parallel(recipe), "actor nodes x gpus:", R.trainer_layout(recipe))
    if result["errors"]:
        raise RuntimeError("check failed:\n" + "\n".join(f"{e['path']}: {e['message']}" for e in result["errors"]))
    print("check: ok")


@app.function(image=image, secrets=secrets, volumes={R.MODELS: models_volume}, cpu=4, memory=16384, timeout=900)
def check_cpu(recipe: dict, run_id: str):
    """Miles' argument validation without a GPU: FSDP as is, Megatron with the CUDA toolkit's libcuda stub."""
    from miles_ext.platform.validate import _parse_args

    stub = R.backend(recipe) == "megatron"
    try:
        parsed = _parse_args(recipe, run_id, libcuda_stub=stub)
        print(f"parse_args without a GPU (libcuda stub: {stub}): ok", parsed)
        return True
    except RuntimeError as e:
        print(f"parse_args without a GPU (libcuda stub: {stub}) failed: {e}")
        return False


def _image_info() -> dict:
    return {"release": os.environ.get("MILES_FORK_RELEASE"), "image": _image_ref(), "commit": _image_commit()}


@app.function(
    image=image,
    gpu=f"{SHAPE['gpu']}:{R.GPUS_PER_NODE}",
    cpu=SHAPE["cpu"],
    memory=SHAPE["memory_gib"] * 1024,
    secrets=secrets,
    volumes={R.MODELS: models_volume, R.RESULTS: results_volume},
    timeout=int(SHAPE["timeout_hours"] * HOURS),
    single_use_containers=True,
)
@modal.clustered(size=SHAPE["nodes"], rdma=SHAPE["nodes"] > 1)
def train(recipe: dict, run_id: str, opts: dict | None = None):
    """One segment. ``opts``: ``segment_seconds``, counted from this container's start (so time
    spent waiting for GPUs is not lost), after which rank 0 saves and exits; the function
    timeout stays the hard bound (miles_ext.platform.launch sets both). Returns the exit record
    on rank 0."""
    from miles_ext.platform import progress as P
    from miles_ext.platform.validate import _inputs

    opts = opts or {}
    deadline = time.time() + float(opts["segment_seconds"]) if opts.get("segment_seconds") else None
    _check_shape(recipe)
    run_dir = R.RESULTS / run_id
    backend = R.backend(recipe)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    cluster = modal.Cluster.from_context()
    rank = cluster.container_rank()
    ips = cluster.container_ips(family="ipv4")
    head_ip, node_ip = ips[0], ips[rank]
    models_volume.reload()
    results_volume.reload()  # see checkpoints committed by an earlier container
    if INPUTS_ON_VOLUME:
        _inputs(run_id)
    # a relaunch of a finished run (the caller's timer fired after the final save): every
    # rank returns before Ray starts, so it costs a container start, not a cluster start
    if P.is_complete(backend, read_tracker(run_dir), recipe["algorithm"]["num_rollout"]):
        print(f"{run_id} is already complete (checkpoint {read_tracker(run_dir)}); nothing to train", flush=True)
        return _complete_at_launch(recipe, run_id, run_dir, backend) if rank == 0 else None
    env = _ray_env(recipe, run_id, head_ip, node_ip)
    start = ["ray", "start", f"--node-ip-address={node_ip}", f"--num-gpus={R.GPUS_PER_NODE}", "--disable-usage-stats"]
    if rank != 0:
        return _worker(run_id, rank, head_ip, env, start)
    return _head(recipe, run_id, run_dir, backend, deadline, head_ip, env, start)


def _complete_at_launch(recipe: dict, run_id: str, run_dir, backend: str) -> dict:
    """Rank 0 of a segment launched after the run completed: an exit record, no training."""
    from miles_ext.platform import progress as P
    from miles_ext.platform.supervisor import write_exit_record

    tracker = read_tracker(run_dir)
    record = {"run_id": run_id, "call_id": modal.current_function_call_id(), "reason": "success",
              "retryable": False, "returncode": None,
              "segment": None, "started_at": time.time(), "ended_at": time.time(), "step": None,
              "tracker": tracker, "complete": True, "num_rollout": recipe["algorithm"]["num_rollout"],
              "next_rollout_id": P.next_rollout_id(backend, tracker), "metrics": {},
              "image": _image_info(), "note": "already complete at launch", "tail": []}
    write_exit_record(run_dir, record)
    results_volume.commit()
    with contextlib.suppress(Exception):
        d, keys = P.progress_dict(TARGET), P.keys(run_id)
        d[keys["exit"]] = record
        d[keys["run"]] = {"run_id": run_id, "call_id": record["call_id"], "phase": "ended",
                          "heartbeat_at": time.time()}
        d.pop(keys["active"], None)
    return record


def _ray_env(recipe: dict, run_id: str, head_ip: str, node_ip: str) -> dict:
    # Ray workers inherit the raylet's environment, so everything rollout workers
    # read (Harbor, the tunnel URL) is in env before `ray start`
    return {
        **os.environ,
        **R.harbor_env(recipe, run_id),
        **R.trainer_env(recipe),
        "PYTHONPATH": _pythonpath(),
        "MASTER_ADDR": head_ip,
        "MILES_HOST_IP": node_ip,
        "RAY_ADDRESS": f"{head_ip}:6379",
        # no RAY_API_SERVER_ADDRESS: the fork's head-node lookup reads the GCS (fix-003)
        "WANDB_RUN_GROUP": run_id,
        # wandb.init reads tags from here; Miles does not pass any
        "WANDB_TAGS": ",".join(R.wandb_tags(recipe) + [f"fork:{os.environ.get('MILES_FORK_RELEASE')}",
                                                       f"commit:{_image_commit()[:10]}"]),
    }


def _worker(run_id: str, rank: int, head_ip: str, env: dict, start: list[str]) -> None:
    """A worker node: join rank 0's Ray cluster and wait for its result. This node's checkpoint
    shards (Megatron) reach the volume only when it commits: on rank 0's request before it
    commits a new tracker (``shards``), on a timer, and on exit."""
    from miles_ext.platform import progress as P
    from miles_ext.platform import shards

    stop = threading.Event()
    threading.Thread(target=shards.worker_commits, daemon=True, args=(
        run_id, rank, P.progress_dict(TARGET), completion_queue, results_volume.commit, stop),
        kwargs={"key": P.keys(run_id)["commit"], "every": WORKER_COMMIT_SECS}).start()
    try:
        _wait_for_port(head_ip, 6379, timeout=600)
        subprocess.run(start + [f"--address={head_ip}:6379"], env=env, check=True)
        if not completion_queue.get(partition=run_id, timeout=int(SHAPE["timeout_hours"] * HOURS)):
            raise RuntimeError("Miles training failed on the head node; see the App logs")
    finally:
        stop.set()
        results_volume.commit()


def _head(recipe: dict, run_id: str, run_dir, backend: str, deadline: float | None, head_ip: str, env: dict,
          start: list[str]) -> dict:
    """Rank 0: start the Ray head, run Miles under the supervisor, release the workers."""
    import secrets as secrets_lib

    import ray

    from miles_ext.platform import progress as P
    from miles_ext.platform import shards
    from miles_ext.platform.supervisor import SENTINEL, Supervisor

    nodes = SHAPE["nodes"]
    succeeded = False
    record = supervisor = None
    stack_ctx = contextlib.ExitStack()
    try:
        resume = read_tracker(run_dir) is not None
        args = R.miles_args(recipe, run_id, resume) + ["--save-trigger-sentinel", SENTINEL]
        run_dir.mkdir(parents=True, exist_ok=True)
        segment = len(list(run_dir.glob("recipe-*.json")))
        (run_dir / f"recipe-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps({"stack": os.environ.get("MILES_FORK_RELEASE"), "image": _image_ref(), "commit": _image_commit(),
                        "resume": resume, "segment": segment, "deadline": deadline, "recipe": recipe,
                        "argv": args}, indent=1)
        )
        commit = shards.head_commit(run_id, nodes, P.progress_dict(TARGET), completion_queue, results_volume.commit,
                                    key=P.keys(run_id)["commit"])
        supervisor = Supervisor(run_id, recipe, run_dir, progress_dict=P.progress_dict(TARGET), segment=segment,
                                deadline=deadline, commit=commit, image=_image_info(),
                                backend=backend, call_id=modal.current_function_call_id())
        supervisor.heartbeat()
        if R.needs_tunnel(recipe):
            # in-sandbox agents reach the session servers through one public tunnel;
            # the random token keeps it from being an open relay
            token = secrets_lib.token_urlsafe(24)
            proxy = subprocess.Popen([
                "python", "-m", "miles_ext.harbor.session_proxy", "--listen-port", str(R.PROXY_PORT),
                "--upstream-host", head_ip, "--ports", f"{R.SESSION_PORT}-{R.SESSION_PORT + R.SESSION_WORKERS - 1}",
                "--token", token,
            ])
            stack_ctx.callback(proxy.terminate)
            _wait_for_port("127.0.0.1", R.PROXY_PORT, timeout=60)
            tunnel = stack_ctx.enter_context(modal.forward(R.PROXY_PORT))
            env["MILES_SESSION_PUBLIC_URL"] = f"{tunnel.url}/{token}"
            print(f"session tunnel: {tunnel.url}/<token>/<port>/...", flush=True)

        # a segment whose workers died before reading leaves items behind; workers read only after
        # the head's port opens, so clearing here, before the head starts, is safe
        completion_queue.clear(partition=run_id)
        completion_queue.clear(partition=shards.ack_partition(run_id))
        subprocess.run(start + ["--head", "--port=6379", "--dashboard-host=0.0.0.0"], env=env, check=True)
        ray.init(address=f"{head_ip}:6379")
        ray_ready_by = time.monotonic() + 600
        while ray.cluster_resources().get("GPU", 0) < nodes * R.GPUS_PER_NODE:
            if time.monotonic() > ray_ready_by:
                raise TimeoutError(f"Ray cluster incomplete: {ray.cluster_resources()}")
            time.sleep(2)
        print(f"Ray cluster ready: {ray.cluster_resources()}", flush=True)
        ray.shutdown()

        entry = R.entrypoint(recipe)
        print(f"fork {os.environ.get('MILES_FORK_RELEASE')} commit {_image_commit()} resume={resume} "
              f"segment={segment} deadline={deadline}\n{entry} " + " ".join(args), flush=True)
        record = supervisor.run(["python", str(stack.MILES_SRC / entry), *args], cwd=str(stack.MILES_SRC), env=env)
        print(f"exit record: {json.dumps({k: v for k, v in record.items() if k != 'tail'}, default=str)}", flush=True)
        if record["reason"] not in ("success", "segment_end", "lease_expired"):
            raise RuntimeError(f"Miles ended with {record['reason']} (exit code {record['returncode']}); "
                               "see the exit record and the App logs")
        # a segment that saved and stopped cleanly also counts: workers exit without error
        succeeded = True
        if FINAL_RETENTION and record["reason"] == "success":
            ckpt = recipe["checkpoint"]
            dropped = apply_retention_local(run_dir, keep=ckpt["keep"], keep_final_full=ckpt["keep_final_full"],
                                            final=True)
            removed = [f"{d['kind']}:{d['step']}" for d in dropped]
            print(f"retention (final, keep_final_full={ckpt['keep_final_full']}): removed {removed}", flush=True)
        return record
    except BaseException as e:
        if record is None and supervisor is not None:  # failed before or around Miles: still leave a record
            with contextlib.suppress(Exception):
                supervisor.finish("error", None, note=f"launcher: {type(e).__name__}: {e}"[:2000])
        raise
    finally:
        stack_ctx.close()
        results_volume.commit()
        _terminate_task_sandboxes(recipe, run_id)
        for _ in range(nodes - 1):
            completion_queue.put(succeeded, partition=run_id)


def _overrides(num_rollout: int, rollout_batch_size: int, n_samples: int, save_interval: int) -> dict:
    algo = {k: v for k, v in {"num_rollout": num_rollout, "rollout_batch_size": rollout_batch_size,
                              "n_samples": n_samples}.items() if v}
    out = {"algorithm": algo} if algo else {}
    if save_interval:
        out["checkpoint"] = {"save_interval": save_interval}
    return out


def spawn_detached():
    """``miles_ext.platform.launch``'s child process: start the app detached, spawn one segment and
    print its ids on a ``LAUNCHED`` line. The configuration is ``MILES_EXT_LAUNCH``."""
    fn = train.with_options(timeout=int(LAUNCH["timeout"])) if LAUNCH.get("timeout") else train
    with modal.enable_output(), app.run(detach=True, environment_name=TARGET.modal_environment):
        call = fn.spawn(RECIPE, LAUNCH["run_id"], LAUNCH.get("opts") or {})
        out = {"app_id": app.app_id, "call_id": call.object_id}
    print("LAUNCHED " + json.dumps(out), flush=True)


@app.local_entrypoint()
def main(run_id: str = "", num_rollout: int = 0, rollout_batch_size: int = 0, n_samples: int = 0,
         save_interval: int = 0, check_only: bool = False, segment_hours: float = 0):
    recipe = R.load(os.environ.get("MILES_RECIPE") or DEFAULT_RECIPE,
                    _overrides(num_rollout, rollout_batch_size, n_samples, save_interval))
    run_id = run_id or f"{recipe['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    algo = recipe["algorithm"]
    print(f"run_id={run_id} image={'built from ' + FROM_COMMIT[:10] if FROM_COMMIT else stack.FORK_IMAGE} recipe={recipe['name']} harness={recipe['taskset']['harness']} "
          f"num_rollout={algo['num_rollout']} batch={algo['rollout_batch_size']}x{algo['n_samples']} "
          f"save_interval={recipe['checkpoint']['save_interval']} cluster={SHAPE} "
          f"wandb={recipe['tracking']['wandb_entity']}/{R.wandb_project(recipe, run_id)}")
    if check_only:
        check.remote(recipe, run_id)
        print("cpu-only parse_args:", "ok" if check_cpu.remote(recipe, run_id) else "failed (see logs)")
        return
    opts = {"segment_seconds": segment_hours * HOURS} if segment_hours else {}
    print(f"done: {train.spawn(recipe, run_id, opts).get()}")
