"""The trainer role: rank 0 runs Miles under the supervisor, the others join its Ray cluster (ext-064)."""

import contextlib
import json
import os
import resource
import secrets as secrets_lib
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from miles_ext import stack
from miles_ext.checkpoint import save_policy
from miles_ext.node import contract as C
from miles_ext.node import runtime
from miles_ext.node import state as node_state
from miles_ext.node import storage as node_storage
from miles_ext.node.gang import GangClient, GangServer
from miles_ext.obs import sandbox_profile
from miles_ext.platform import compile_cache
from miles_ext.platform import progress as P
from miles_ext.platform import shards
from miles_ext.platform.checkpoints import apply_retention_local, remove_other_node_shards
from miles_ext.platform.images import IMAGE_ENV, TRAINER_ENV, image_commit, image_version
from miles_ext.platform.layout import iter_dir, read_tracker
from miles_ext.platform.supervisor import SENTINEL, Supervisor, write_exit_record
from miles_ext.platform.validate import _inputs
from miles_ext.recipes import recipe as R
from miles_ext.sandboxes.cleanup import sandbox_adapter

WORKER_COMMIT_SECS = 120
HEAD_PORT = 6379


@dataclass(frozen=True)
class Gang:
    rank: int
    nnodes: int
    head_addr: str
    node_ip: str

    @classmethod
    def from_env(cls) -> "Gang":
        return cls(
            int(os.environ[C.NODE_RANK_ENV]),
            int(os.environ[C.NNODES_ENV]),
            os.environ[C.HEAD_ADDR_ENV],
            os.environ[C.NODE_IP_ENV],
        )


def _image_info() -> dict:
    # "release" repeats "version" for callers of releases before 2026-10-15
    return {
        "trainer": os.environ.get(TRAINER_ENV),
        "version": image_version(),
        "release": image_version(),
        "image": os.environ.get(IMAGE_ENV) or None,
        "commit": image_commit(),
    }


def _wait_for_port(host: str, port: int, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as e:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{host}:{port} did not open within {timeout}s") from e
            time.sleep(1)


def _terminate_task_sandboxes(recipe: dict, run_id: str):
    """Episodes in flight when the trainer exits never reach Harbor's cleanup; the sandbox app is this run's alone."""
    for name in R.sandbox_app_names_for_cleanup(recipe, run_id):
        _terminate_app_sandboxes(recipe, name)


def _terminate_app_sandboxes(recipe: dict, name: str):
    provider = recipe["sandbox"]["provider"]
    try:
        R.check_own_app_name(name)
        n = sandbox_adapter(provider, os.environ.get(C.SANDBOX_ENVIRONMENT_ENV)).terminate(
            name
        )  # exact name, never created
        print(f"terminated {n} {provider} sandboxes in {name}", flush=True)
    except Exception as e:
        print(f"sandbox cleanup ({provider} {name}): {type(e).__name__}: {e}", flush=True)


def _trainer_tag() -> str:
    """The W&B tag value: the run's trainer (latest, stable-...) or, launched by hand, the image's version."""
    return os.environ.get(TRAINER_ENV) or image_version() or "unknown"


def _ray_env(recipe: dict, run_id: str, gang: Gang) -> dict:
    # Ray workers inherit the raylet's environment, so everything rollout workers
    # read (Harbor, the tunnel URL) is in env before `ray start`
    return {
        **os.environ,
        **compile_cache.env(),
        **R.harbor_env(recipe, run_id),
        **R.trainer_env(recipe),
        "PYTHONPATH": runtime.pythonpath(),
        "MASTER_ADDR": gang.head_addr,
        "MILES_HOST_IP": gang.node_ip,
        "RAY_ADDRESS": f"{gang.head_addr}:{HEAD_PORT}",
        # no RAY_API_SERVER_ADDRESS: Miles' head-node lookup reads the GCS (upstream #3640)
        "WANDB_RUN_GROUP": run_id,
        # wandb.init reads tags from here; Miles does not pass any
        "WANDB_TAGS": ",".join(
            R.wandb_tags(recipe)
            + [f"miles-mobius:{_trainer_tag()}", f"upstream:{stack.UPSTREAM_LABEL}", f"commit:{image_commit()[:10]}"]
        ),
        # like WANDB_RUN_GROUP, the env wins over the name Miles passes (<group>_<random>-RANK_0)
        **({"WANDB_NAME": recipe["tracking"]["wandb_name"]} if recipe["tracking"].get("wandb_name") else {}),
    }


def _retention_while_training(recipe: dict, run_dir, nodes: int, every: float = 60.0) -> threading.Event:
    """Apply checkpoint retention (final=False) each time the tracker moves; set the returned event to stop."""
    ckpt, stop, storage = recipe["checkpoint"], threading.Event(), node_storage.for_mount(R.RESULTS)

    def loop() -> None:
        seen = read_tracker(run_dir)
        while not stop.wait(every):
            if (step := read_tracker(run_dir)) == seen:
                continue
            seen = step
            try:
                dropped = apply_retention_local(
                    run_dir,
                    keep=ckpt["keep"],
                    keep_final_full=ckpt["keep_final_full"],
                    keep_every=ckpt["keep_every"],
                    final=False,
                )
                if dropped:
                    storage.commit()
                    remove_other_node_shards(storage, R.RESULTS, dropped, nodes)
                    print(f"retention (tracker {step}): removed {[d['step'] for d in dropped]}", flush=True)
            except Exception as e:  # retention never stops training
                print(f"retention (tracker {step}) failed: {type(e).__name__}: {e}", flush=True)

    threading.Thread(target=loop, daemon=True, name="retention").start()
    return stop


def _complete_at_launch(recipe: dict, run_id: str, run_dir, backend: str) -> dict:
    """Rank 0 of a segment launched after the run completed: an exit record, no training."""
    tracker = read_tracker(run_dir)
    call_id = os.environ.get(C.CALL_ID_ENV)
    record = {
        "run_id": run_id,
        "call_id": call_id,
        "reason": "success",
        "retryable": False,
        "returncode": None,
        "segment": None,
        "started_at": time.time(),
        "ended_at": time.time(),
        "step": None,
        "tracker": tracker,
        "complete": True,
        "num_rollout": recipe["algorithm"]["num_rollout"],
        "next_rollout_id": P.next_rollout_id(backend, tracker),
        "metrics": {},
        "image": _image_info(),
        "note": "already complete at launch",
        "tail": [],
    }
    write_exit_record(run_dir, record)
    node_storage.for_mount(R.RESULTS).commit()
    with contextlib.suppress(Exception):
        d, keys = node_state.for_run(run_id), P.keys(run_id)
        d[keys["exit"]] = record
        d[keys["run"]] = {"run_id": run_id, "call_id": call_id, "phase": "ended", "heartbeat_at": time.time()}
        d.pop(keys["active"], None)
    return record


def _split_links(
    recipe: dict,
    run_dir,
    env: dict,
    stack_ctx: contextlib.ExitStack,
    registry_url: str,
    wait_check: Callable[[], None] | None,
    on_failure: Callable[[str], None],
) -> None:
    """Split, on each trainer node before `ray start` (Ray workers inherit env): start the relay and the commit watcher,
    and stop the run through ``on_failure`` if no rollout node comes (ext-069)."""
    from miles_ext.launch.split import relay, volume_sync
    from miles_ext.node import registry as node_registry

    stack_ctx.callback(volume_sync.commit_watcher(node_storage.for_mount(R.RESULTS).commit).set)
    pool_token = os.environ[C.POOL_TOKEN_ENV]
    proc = relay.start(registry_url, pool_token)
    stack_ctx.callback(proc.terminate)
    env[volume_sync.REGISTRY_ENV], env[volume_sync.RUN_DIR_ENV] = registry_url, str(run_dir)
    timeout = C.first_node_timeout_secs()
    stopped = threading.Event()
    stack_ctx.callback(stopped.set)

    def check() -> None:
        if stopped.is_set():
            raise _Stopped
        if wait_check:
            wait_check()

    def watch() -> None:
        try:
            members = node_registry.wait_members(registry_url, pool_token, timeout=timeout, check=check)
            relay.check_members(recipe, members)
            relay.wait_open(members, proc)
        except _Stopped:
            pass
        except Exception as e:
            note = f"split: no relay to a rollout node: {type(e).__name__}: {e}"
            print(note, flush=True)
            if not stopped.is_set():
                on_failure(note)

    threading.Thread(target=watch, daemon=True, name="split-links").start()


class _Stopped(Exception):
    """This node is shutting down, so nobody waits for a rollout node any more."""


def _relay_failure_key(run_id: str) -> str:
    return f"relay-failure:{run_id}"


def _watch_worker_relays(
    server: GangServer, run_id: str, on_failure: Callable[[str], None], stop: threading.Event
) -> None:
    """Stop the run when a worker node reports it has no relay."""
    while not stop.wait(5):
        if note := server.kv.get(_relay_failure_key(run_id)):
            on_failure(note)
            return


def _publish_pool_status(registry, state, key: str, stop: threading.Event, every: float = 30.0) -> None:
    """The rollout pool's size and load where a caller reads run state; it decides whether to add or drop nodes."""
    while not stop.wait(every):
        try:
            state[key] = registry.status()
        except Exception as e:
            print(f"pool status: {type(e).__name__}: {e}", flush=True)


def _worker(
    recipe: dict,
    run_id: str,
    run_dir,
    gang: Gang,
    env: dict,
    start: list[str],
    cache_key: str,
    wait_check: Callable[[], None] | None,
) -> None:
    """A worker node: join rank 0's Ray cluster and wait for its result. This node's checkpoint
    shards (Megatron) reach the storage only when it commits: on rank 0's request before it
    commits a new tracker (``shards``), on a timer, and on exit."""
    stop = threading.Event()
    stack_ctx = contextlib.ExitStack()
    storage = node_storage.for_mount(R.RESULTS)
    saver = compile_cache.Saver(cache_key, node_storage.for_mount(compile_cache.MOUNT).commit, node=gang.rank).start()
    client = GangClient(gang.head_addr)
    threading.Thread(
        target=shards.worker_commits,
        daemon=True,
        args=(run_id, gang.rank, client.kv, client.queue, storage.commit, stop),
        kwargs={"key": P.keys(run_id)["commit"], "every": WORKER_COMMIT_SECS},
    ).start()
    try:
        _wait_for_port(gang.head_addr, HEAD_PORT, timeout=600)
        if R.is_split(recipe):

            def report_relay_failure(note: str) -> None:
                with contextlib.suppress(OSError):
                    client.kv[_relay_failure_key(run_id)] = note

            registry_url = f"http://{gang.head_addr}:{C.REGISTRY_PORT}"
            _split_links(recipe, run_dir, env, stack_ctx, registry_url, wait_check, report_relay_failure)
        subprocess.run(start + [f"--address={gang.head_addr}:{HEAD_PORT}"], env=env, check=True)
        if not client.wait_release(run_id, gang.rank, timeout=int(R.shape(recipe)["timeout_hours"] * 3600)):
            raise RuntimeError("Miles training failed on the head node; see the App logs")
    finally:
        stack_ctx.close()
        stop.set()
        storage.commit()
        saver.stop()


def _head(
    recipe: dict,
    run_id: str,
    run_dir,
    backend: str,
    deadline: float | None,
    gang: Gang,
    env: dict,
    start: list[str],
    cache_key: str,
    final_retention: bool,
    wait_check: Callable[[], None] | None,
) -> dict:
    """Rank 0: start the Ray head, run Miles under the supervisor, release the workers."""
    import ray

    nodes = gang.nnodes
    succeeded = False
    record = supervisor = registry = None
    server = GangServer().start() if nodes > 1 else GangServer()
    stack_ctx = contextlib.ExitStack()
    storage = node_storage.for_mount(R.RESULTS)
    # kernels compiled and tuned by this segment reach the storage while it runs and at exit
    stack_ctx.callback(compile_cache.Saver(cache_key, node_storage.for_mount(compile_cache.MOUNT).commit).start().stop)
    try:
        step = read_tracker(run_dir)
        resume = step is not None
        # a run saving Adam's state on every k-th save only may resume from a save without it (ext-056)
        resume_optimizer = (
            not resume
            or recipe["checkpoint"]["optimizer_every"] == 1
            or save_policy.has_optimizer(run_dir / "checkpoints" / iter_dir(step))
        )
        args = R.miles_args(recipe, run_id, resume, resume_optimizer) + ["--save-trigger-sentinel", SENTINEL]
        run_dir.mkdir(parents=True, exist_ok=True)
        segment = len(list(run_dir.glob("recipe-*.json")))
        (run_dir / f"recipe-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps(
                {
                    "trainer": os.environ.get(TRAINER_ENV),
                    "version": image_version(),
                    "stack": image_version(),
                    "image": os.environ.get(IMAGE_ENV) or None,
                    "commit": image_commit(),
                    "resume": resume,
                    "segment": segment,
                    "deadline": deadline,
                    "recipe": recipe,
                    "argv": args,
                },
                indent=1,
            )
        )
        commit = shards.head_commit(
            run_id, nodes, server.kv, server.queue, storage.commit, key=P.keys(run_id)["commit"]
        )
        supervisor = Supervisor(
            run_id,
            recipe,
            run_dir,
            progress_dict=node_state.for_run(run_id),
            segment=segment,
            deadline=deadline,
            commit=commit,
            image=_image_info(),
            backend=backend,
            call_id=os.environ.get(C.CALL_ID_ENV),
        )
        supervisor.heartbeat()
        if R.needs_tunnel(recipe):
            # in-sandbox agents reach the session servers through one public URL;
            # the random token keeps it from being an open relay
            token = secrets_lib.token_urlsafe(24)
            proxy = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "miles_ext.harbor.session_proxy",
                    "--listen-port",
                    str(R.PROXY_PORT),
                    "--upstream-host",
                    gang.head_addr,
                    "--ports",
                    f"{R.SESSION_PORT}-{R.SESSION_PORT + R.SESSION_WORKERS - 1}",
                    f"--token={token}",  # token_urlsafe can start with "-", which argparse reads as a flag
                    *(
                        ["--profile-log", str(run_dir / sandbox_profile.PROXY_FILE)]
                        if recipe["obs"]["sandbox_profile"]
                        else []
                    ),
                ]
            )
            stack_ctx.callback(proxy.terminate)
            _wait_for_port("127.0.0.1", R.PROXY_PORT, timeout=60)
            public = os.environ[C.SESSION_URL_ENV].rstrip("/")
            env[C.SESSION_URL_ENV] = f"{public}/{token}"
            print(f"session proxy: {public}/<token>/<port>/...", flush=True)

        if R.is_split(recipe):  # before `ray start`: Ray workers inherit the relays' addresses through env
            from miles_ext.node.registry import Registry

            registry = Registry(os.environ[C.POOL_TOKEN_ENV]).start()
            status_stop = threading.Event()
            stack_ctx.callback(status_stop.set)
            threading.Thread(
                target=_publish_pool_status,
                daemon=True,
                args=(registry, node_state.for_run(run_id), P.keys(run_id)["pool"], status_stop),
            ).start()
            registry_url = f"http://127.0.0.1:{C.REGISTRY_PORT}"
            _split_links(recipe, run_dir, env, stack_ctx, registry_url, wait_check, supervisor.fail)
            if nodes > 1:
                threading.Thread(
                    target=_watch_worker_relays,
                    args=(server, run_id, supervisor.fail, status_stop),
                    daemon=True,
                    name="worker-relays",
                ).start()

        if final_retention:  # ext-056: prune old saves while training, so a long run does not fill the storage
            stack_ctx.callback(_retention_while_training(recipe, run_dir, nodes).set)
        subprocess.run(start + ["--head", f"--port={HEAD_PORT}", "--dashboard-host=0.0.0.0"], env=env, check=True)
        ray.init(address=f"{gang.head_addr}:{HEAD_PORT}")
        ray_ready_by = time.monotonic() + 600
        while ray.cluster_resources().get("GPU", 0) < nodes * R.gpus_per_node(recipe):
            if time.monotonic() > ray_ready_by:
                raise TimeoutError(f"Ray cluster incomplete: {ray.cluster_resources()}")
            time.sleep(2)
        print(f"Ray cluster ready: {ray.cluster_resources()}", flush=True)
        ray.shutdown()

        entry = R.entrypoint(recipe)
        print(
            f"miles-mobius {image_version()} commit {image_commit()} resume={resume} "
            f"segment={segment} deadline={deadline}\n{entry} " + " ".join(args),
            flush=True,
        )
        record = supervisor.run(["python", str(stack.MILES_SRC / entry), *args], cwd=str(stack.MILES_SRC), env=env)
        print(f"exit record: {json.dumps({k: v for k, v in record.items() if k != 'tail'}, default=str)}", flush=True)
        if record["reason"] not in ("success", "segment_end", "lease_expired"):
            raise RuntimeError(
                f"Miles ended with {record['reason']} (exit code {record['returncode']}); "
                "see the exit record and the App logs"
            )
        # a segment that saved and stopped cleanly also counts: workers exit without error
        succeeded = True
        if registry is not None and record["reason"] == "success":  # the run is complete: the rollout node exits
            registry.release()
        if final_retention and record["reason"] == "success":
            ckpt = recipe["checkpoint"]
            dropped = apply_retention_local(
                run_dir,
                keep=ckpt["keep"],
                keep_final_full=ckpt["keep_final_full"],
                keep_every=ckpt["keep_every"],
                final=True,
            )
            storage.commit()
            remove_other_node_shards(storage, R.RESULTS, dropped, nodes)
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
        storage.commit()
        _terminate_task_sandboxes(recipe, run_id)
        server.release(run_id, succeeded, nodes)


def run(recipe: dict, run_id: str, *, wait_check: Callable[[], None] | None = None) -> dict | None:
    """One segment on this node. Returns the exit record on rank 0. ``wait_check`` may raise while a split trainer
    waits for a rollout node, for a caller that can see the node has died."""
    deadline = float(os.environ[C.DEADLINE_ENV]) if os.environ.get(C.DEADLINE_ENV) else None
    final_retention = os.environ.get(C.FINAL_RETENTION_ENV, "1") == "1"
    gang = Gang.from_env()
    run_dir = R.RESULTS / run_id
    backend = R.backend(recipe)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    node_storage.for_mount(R.MODELS).refresh()
    node_storage.for_mount(R.RESULTS).refresh()  # see checkpoints committed by an earlier container
    _inputs(run_id, tasks=R.reads_task_dirs(recipe))
    # a relaunch of a finished run (the caller's timer fired after the final save): every
    # rank returns before Ray starts, so it costs a container start, not a cluster start
    if P.is_complete(backend, read_tracker(run_dir), recipe["algorithm"]["num_rollout"]):
        print(f"{run_id} is already complete (checkpoint {read_tracker(run_dir)}); nothing to train", flush=True)
        return _complete_at_launch(recipe, run_id, run_dir, backend) if gang.rank == 0 else None
    cache_key = runtime.compile_cache_key(R.shape(recipe)["gpu"])
    compile_cache.restore(cache_key)
    env = _ray_env(recipe, run_id, gang)
    start = [
        "ray",
        "start",
        f"--node-ip-address={gang.node_ip}",
        f"--num-gpus={R.gpus_per_node(recipe)}",
        "--disable-usage-stats",
    ]
    if gang.rank != 0:
        return _worker(recipe, run_id, run_dir, gang, env, start, cache_key, wait_check)
    return _head(recipe, run_id, run_dir, backend, deadline, gang, env, start, cache_key, final_retention, wait_check)
