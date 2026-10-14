"""Test launcher: run a recipe on Modal with the published fork image, unchanged.

Not part of the platform API; it exists to validate a fork release. The image
already holds Miles (our branch), Harbor and mini-swe-agent; this module only
adds the run inputs (tasks, prompts) and the cluster wiring.

    export MILES_DATA=<dir with the task sets>  MILES_RECIPE=miles_ext/recipes/configs/<name>.yaml
    modal run -m miles_ext.launch.modal_app --run-id <id> --check-only    # dry check in the image
    modal run --detach -m miles_ext.launch.modal_app --run-id <id>        # train (same id again resumes)

Credentials come from the .env in ``MILES_DOTENV_DIR`` (default: cwd) as a
Modal Secret. Names are this fork's own: app ``miles-fork``, volume
``miles-fork-results``, queue ``miles-fork-completion``, sandbox app
``miles-fork-sbx-<run_id>``.
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
from miles_ext.recipes import recipe as R

_SHAPE_ENV = "MILES_EXT_SHAPE"
DEFAULT_RECIPE = "miles_ext/recipes/configs/qwen35-9b-swegym8-sandbox.yaml"

if modal.is_local():
    RECIPE = R.load(os.environ.get("MILES_RECIPE") or DEFAULT_RECIPE)
    SHAPE = R.shape(RECIPE)
    if not stack.FORK_IMAGE:
        raise RuntimeError("miles_ext/stack.py: FORK_IMAGE is empty; build and push the image first")
else:
    RECIPE = None
    SHAPE = json.loads(os.environ[_SHAPE_ENV])

HOURS = 3600

app = modal.App("miles-fork")
models_volume = modal.Volume.from_name("miles-swe-models", create_if_missing=True)
results_volume = modal.Volume.from_name("miles-fork-results", create_if_missing=True, version=2)
secrets = [modal.Secret.from_dotenv(os.environ.get("MILES_DOTENV_DIR") or os.getcwd())]
completion_queue = modal.Queue.from_name("miles-fork-completion", create_if_missing=True)

image = (
    modal.Image.from_registry(stack.FORK_IMAGE or "scratch")
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
          _SHAPE_ENV: json.dumps(SHAPE, sort_keys=True)})
)
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


def _terminate_task_sandboxes(run_id: str):
    """Episodes in flight when the trainer exits never reach Harbor's cleanup; the sandbox app is this run's alone."""
    name = R.sandbox_app_name(run_id)
    try:
        sandbox_app = modal.App.lookup(name)
        sandboxes = list(modal.Sandbox.list(app_id=sandbox_app.app_id))
        for sb in sandboxes:
            sb.terminate()
        print(f"terminated {len(sandboxes)} task sandboxes in {name}", flush=True)
    except Exception as e:
        print(f"task sandbox cleanup ({name}): {type(e).__name__}: {e}", flush=True)


def _check_shape(recipe: dict):
    if R.shape(recipe) != SHAPE:
        raise RuntimeError(f"recipe shape {R.shape(recipe)} != the function's {SHAPE}; relaunch with the same recipe")


@app.function(image=image, secrets=secrets, volumes={R.MODELS: models_volume}, gpu="H100", cpu=4, memory=32768,
              timeout=1800)
def check(recipe: dict, run_id: str):
    """Dry run inside the fork image: commit, fork fixes, package imports, trial config, Miles argv."""
    import sys

    _check_shape(recipe)
    print(f"image commit {_image_commit()}, stack {os.environ.get('MILES_FORK_RELEASE')}")
    os.environ.update(R.harbor_env(recipe, run_id))
    os.environ["MILES_SESSION_PUBLIC_URL"] = "https://tunnel.invalid/token"
    subprocess.run([sys.executable, str(stack.MILES_SRC / "fork/tools/ledger.py"), "markers"],
                   cwd=stack.MILES_SRC, check=True)

    # fix-003: the head-node lookup must work with no Ray dashboard at all
    subprocess.run(["ray", "start", "--head", "--include-dashboard=false", "--disable-usage-stats"], check=True,
                   capture_output=True)
    import ray

    from miles.utils.ray_utils import _get_head_node_id

    ray.init(address="auto")
    print("head node (GCS lookup, no dashboard):", _get_head_node_id())
    ray.shutdown()
    subprocess.run(["ray", "stop", "--force"], capture_output=True)

    from miles_ext.harbor import agent_function, host_agent, reward, sandbox_agent, session_proxy  # noqa: F401

    print("upstream agent function:", agent_function.hf.__file__)
    config = agent_function.build_trial_config(
        {"instance_id": sorted(os.listdir(R.TASKS))[0]}, "http://10.0.0.1:30003/sessions/abc/v1", {"temperature": 1.0}
    )
    print("trial config:", config.agent.import_path, config.agent.env.get("OPENAI_API_BASE"), config.task.path)
    print("prompt rows:", len(R.PROMPTS.read_text().splitlines()), "tasks:", len(os.listdir(R.TASKS)))
    args = R.miles_args(recipe, run_id, resume=False)
    os.chdir(stack.MILES_SRC)
    sys.argv = [R.entrypoint(recipe), *args]
    from miles.utils.arguments import parse_args

    parsed = parse_args()
    keys = ["train_backend", "fully_async", "rollout_num_gpus", "actor_num_gpus_per_node", "global_batch_size",
            "num_rollout", "save_interval", "tito_model", "sglang_tool_call_parser", "wandb_team"]
    print({k: getattr(parsed, k, None) for k in keys})
    print("check: ok")


@app.function(image=image, secrets=secrets, volumes={R.MODELS: models_volume}, cpu=4, memory=16384, timeout=900)
def check_cpu(recipe: dict, run_id: str):
    """Can Miles' argument validation run without a GPU? (the platform wants to validate specs cheaply)"""
    import sys
    import traceback

    os.chdir(stack.MILES_SRC)
    sys.argv = [R.entrypoint(recipe), *R.miles_args(recipe, run_id, resume=False)]
    try:
        from miles.utils.arguments import parse_args

        parsed = parse_args()
        print("parse_args without a GPU: ok", {k: getattr(parsed, k, None) for k in ("train_backend", "fully_async")})
        return True
    except BaseException:
        print("parse_args without a GPU failed:")
        traceback.print_exc()
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
        "MASTER_ADDR": head_ip,
        "MILES_HOST_IP": node_ip,
        "RAY_ADDRESS": f"{head_ip}:6379",
        # no RAY_API_SERVER_ADDRESS: the fork's head-node lookup reads the GCS (fix-003)
        "WANDB_RUN_GROUP": run_id,
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
            json.dumps({"stack": os.environ.get("MILES_FORK_RELEASE"), "image": stack.FORK_IMAGE, "commit": _image_commit(),
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
        return str(run_dir)
    finally:
        stack_ctx.close()
        results_volume.commit()
        if rank == 0:
            _terminate_task_sandboxes(run_id)
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
    print(f"run_id={run_id} image={stack.FORK_IMAGE} recipe={recipe['name']} harness={recipe['taskset']['harness']} "
          f"num_rollout={algo['num_rollout']} batch={algo['rollout_batch_size']}x{algo['n_samples']} "
          f"save_interval={recipe['checkpoint']['save_interval']} cluster={SHAPE}")
    if check_only:
        check.remote(recipe, run_id)
        print("cpu-only parse_args:", "ok" if check_cpu.remote(recipe, run_id) else "failed (see logs)")
        return
    print(f"done: {train.spawn(recipe, run_id).get()}")
