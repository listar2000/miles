"""Test launcher: run a recipe on Modal with the published fork image, unchanged.

Not part of the platform API; it exists to validate a fork release. The image
already holds Miles (our branch), Harbor and mini-swe-agent; this module only
adds the run inputs (tasks, prompts) and the cluster wiring.

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
"""

import contextlib
import json
import os
import resource
import socket
import subprocess
import time

import modal

from miles_ext import stack
from miles_ext.platform.checkpoints import apply_retention_local
from miles_ext.platform.images import fork_image
from miles_ext.platform.target import Target
from miles_ext.platform.teardown import sandbox_adapter
from miles_ext.recipes import recipe as R

_SHAPE_ENV = "MILES_EXT_SHAPE"
DEFAULT_RECIPE = "miles_ext/recipes/configs/qwen35-9b-swegym8-sandbox.yaml"

if modal.is_local():
    RECIPE = R.load(os.environ.get("MILES_RECIPE") or DEFAULT_RECIPE)
    SHAPE = R.shape(RECIPE)
    # MILES_FORK_COMMIT=<sha>: build the same image on Modal from that pushed fork
    # commit instead of pulling FORK_IMAGE (when no registry push is possible)
    FROM_COMMIT = os.environ.get("MILES_FORK_COMMIT", "")
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
if modal.is_local():
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
if modal.is_local():
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
if RECIPE is not None:
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
def train(recipe: dict, run_id: str):
    import secrets as secrets_lib

    import ray

    _check_shape(recipe)
    nodes = SHAPE["nodes"]
    run_dir = R.RESULTS / run_id
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    cluster = modal.Cluster.from_context()
    rank = cluster.container_rank()
    ips = cluster.container_ips(family="ipv4")
    head_ip, node_ip = ips[0], ips[rank]
    # Ray workers inherit the raylet's environment, so everything rollout workers
    # read (Harbor, the tunnel URL) is in env before `ray start`
    env = {
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
    start = ["ray", "start", f"--node-ip-address={node_ip}", f"--num-gpus={R.GPUS_PER_NODE}", "--disable-usage-stats"]
    succeeded = False
    stack_ctx = contextlib.ExitStack()
    try:
        models_volume.reload()
        results_volume.reload()  # see checkpoints committed by an earlier container
        if rank != 0:
            _wait_for_port(head_ip, 6379, timeout=600)
            subprocess.run(start + [f"--address={head_ip}:6379"], env=env, check=True)
            if not completion_queue.get(partition=run_id, timeout=int(SHAPE["timeout_hours"] * HOURS)):
                raise RuntimeError("Miles training failed on the head node; see the App logs")
            return

        resume = (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
        args = R.miles_args(recipe, run_id, resume)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"recipe-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps({"stack": os.environ.get("MILES_FORK_RELEASE"), "image": _image_ref(), "commit": _image_commit(),
                        "resume": resume, "recipe": recipe, "argv": args}, indent=1)
        )
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

        subprocess.run(start + ["--head", "--port=6379", "--dashboard-host=0.0.0.0"], env=env, check=True)
        ray.init(address=f"{head_ip}:6379")
        deadline = time.monotonic() + 600
        while ray.cluster_resources().get("GPU", 0) < nodes * R.GPUS_PER_NODE:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Ray cluster incomplete: {ray.cluster_resources()}")
            time.sleep(2)
        print(f"Ray cluster ready: {ray.cluster_resources()}", flush=True)
        ray.shutdown()

        entry = R.entrypoint(recipe)
        print(f"fork {os.environ.get('MILES_FORK_RELEASE')} commit {_image_commit()} resume={resume}\n{entry} " + " ".join(args), flush=True)
        subprocess.run(["python", str(stack.MILES_SRC / entry), *args], cwd=stack.MILES_SRC, env=env, check=True)
        succeeded = True
        if FINAL_RETENTION:
            ckpt = recipe["checkpoint"]
            dropped = apply_retention_local(run_dir, keep=ckpt["keep"], keep_final_full=ckpt["keep_final_full"],
                                            final=True)
            removed = [f"{d['kind']}:{d['step']}" for d in dropped]
            print(f"retention (final, keep_final_full={ckpt['keep_final_full']}): removed {removed}", flush=True)
        return str(run_dir)
    finally:
        stack_ctx.close()
        results_volume.commit()
        if rank == 0:
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


@app.local_entrypoint()
def main(run_id: str = "", num_rollout: int = 0, rollout_batch_size: int = 0, n_samples: int = 0,
         save_interval: int = 0, check_only: bool = False):
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
    print(f"done: {train.spawn(recipe, run_id).get()}")
