"""R3: ``launch`` one training segment of a run, from Python, and return its ids.

Modal binds a clustered function's shape (GPU, nodes, CPU, memory) when the
function is defined, so the trainer function cannot be rebuilt per run inside one
process. ``launch`` therefore runs ``miles_ext.launch.modal_app`` in a child
interpreter with the whole configuration in one variable (``MILES_EXT_LAUNCH``);
the caller's process reads no environment, and any number of runs can be launched
from it. The child starts the app detached and spawns one call; GPUs bill from
placement until the call ends.

Before spawning, ``launch`` returns without starting anything when the run is
already complete (its checkpoints reach ``num_rollout``): a relaunch after the
final save costs nothing. It refuses while another segment of the run is live
(``active:<run_id>`` in the progress Dict holds its call id), and marks the new
segment ``pending`` with its call id until the trainer's first heartbeat.
"""

import json
import os
import subprocess
import sys
import time

from miles_ext.platform import progress as P
from miles_ext.platform.secrets import run_secret_exists
from miles_ext.platform.target import Target, run_secret_name
from miles_ext.platform.teardown import resource_names
from miles_ext.platform.weights import recipe_weights_ready
from miles_ext.recipes import recipe as R

GRACE_SECS = 20 * 60  # function timeout beyond the segment: the save and shutdown after the deadline
MAX_TIMEOUT_SECS = 24 * 3600  # Modal's limit
SPAWN_TIMEOUT_SECS = 30 * 60  # the child: image pull or build, then the spawn


def segment_timeout(segment_seconds: float | None, shape_timeout_hours: float) -> int:
    if not segment_seconds:
        return int(min(shape_timeout_hours * 3600, MAX_TIMEOUT_SECS))
    return int(min(segment_seconds + GRACE_SECS, MAX_TIMEOUT_SECS))


def launch(recipe: dict, run_id: str, target: Target | dict | None, *, segment_seconds: float | None = None,
           lease_seconds: float | None = None, inputs: str = "volume", from_commit: str | None = None,
           final_retention: bool = False) -> dict:
    """Start one segment; returns ``{launched, complete, app_id, call_id, names, timeout, segment_seconds}``.

    - ``recipe``: the resolved recipe (``validate`` passed); ``target``: where it runs.
      The per-run secret (``create_run_secret``) and, with ``model.revision``, the
      weights (``prepare_weights``) must exist first; launch checks, it does not create.
    - ``segment_seconds``: rank 0 saves and exits this long after its container starts
      (exit reason ``segment_end``); the function timeout is that plus ``GRACE_SECS``.
      None: no deadline, the recipe's ``cluster.timeout_hours`` is the timeout.
    - ``lease_seconds``: write the first lease (``renew_lease``); the caller renews it.
    - ``inputs``: ``volume`` (taskset at ``<results>/<run_id>/inputs/``) or ``image``
      (the recipe's local taskset paths, added to the image).
    - ``from_commit``: build the image on Modal from a pushed fork commit instead of
      pulling the release image (``stack.FORK_IMAGE``).
    - ``final_retention``: let the trainer apply final retention itself (standalone use;
      the platform runs its own, with its pinned steps).
    """
    from miles_ext.platform.validate import instant_errors

    target = Target.of(target)
    target = target.with_(secret_name=run_secret_name(run_id, target))
    recipe = target.apply(recipe)
    if errors := instant_errors(recipe, run_id, target):
        raise ValueError("launch: " + "; ".join(f"{e['path']}: {e['message']}" for e in errors))
    if inputs not in ("volume", "image"):
        raise ValueError(f"inputs {inputs!r} must be volume or image")
    names = resource_names(recipe, run_id, target)
    timeout = segment_timeout(segment_seconds, R.shape(recipe)["timeout_hours"])
    out = {"launched": False, "complete": False, "app_id": None, "call_id": None, "names": names,
           "timeout": timeout, "segment_seconds": segment_seconds}

    done = P.completion(run_id, target, recipe)
    out.update(complete=done["complete"], next_rollout_id=done["next_rollout_id"], num_rollout=done["num_rollout"])
    if done["complete"]:
        return out
    if not run_secret_exists(run_id, target):
        raise RuntimeError(f"secret {target.secret_name} does not exist (create_run_secret first)")
    if recipe["model"].get("revision") and not recipe_weights_ready(recipe, target):
        raise RuntimeError(f"weights {R.model_dir(recipe)} are not ready (prepare_weights first)")
    _claim(run_id, target)
    try:
        if lease_seconds:
            P.renew_lease(run_id, target, lease_seconds)
        ids = _spawn(recipe, run_id, target, inputs=inputs, from_commit=from_commit,
                     final_retention=final_retention, timeout=timeout, segment_seconds=segment_seconds)
    except BaseException:
        P.progress_dict(target).pop(P.keys(run_id)["active"], None)
        raise
    d, now = P.progress_dict(target), time.time()
    d[P.keys(run_id)["active"]] = {"call_id": ids["call_id"], "app_id": ids["app_id"], "ts": now}
    d[P.keys(run_id)["run"]] = {"run_id": run_id, "phase": "pending", "call_id": ids["call_id"],
                                "app_id": ids["app_id"], "launched_at": now, "heartbeat_at": now}
    out.update(ids, launched=True)
    return out


def _claim(run_id: str, target: Target) -> bool:
    """Take ``active:<run_id>``; raise if a segment of this run is live or being launched."""
    d, key = P.progress_dict(target), P.keys(run_id)["active"]
    for _ in range(2):
        if d.put(key, {"call_id": None, "ts": time.time()}, skip_if_exists=True):
            return True
        held = d.get(key) or {}
        if held.get("call_id") and P.call_running(held["call_id"]):
            raise RuntimeError(f"{run_id}: segment {held['call_id']} is still running (teardown it first)")
        if not held.get("call_id") and time.time() - held.get("ts", 0) < SPAWN_TIMEOUT_SECS:
            raise RuntimeError(f"{run_id}: another launch of this run is in progress")
        d.pop(key, None)  # the segment it names has ended, or a launch died: take over
    raise RuntimeError(f"{run_id}: could not claim the run for launch")


def _spawn(recipe: dict, run_id: str, target: Target, *, inputs: str, from_commit: str | None,
           final_retention: bool, timeout: int, segment_seconds: float | None) -> dict:
    config = {"recipe": recipe, "run_id": run_id, "target": target.as_dict(), "inputs": inputs,
              "from_commit": from_commit, "final_retention": final_retention, "timeout": timeout,
              "opts": {"segment_seconds": segment_seconds} if segment_seconds else {}}
    # the child's configuration is MILES_EXT_LAUNCH alone: drop the test launcher's MILES_* switches
    # (a stray MILES_FORK_COMMIT in the caller's env would otherwise change the image)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MILES_")}
    env["MILES_EXT_LAUNCH"] = json.dumps(config)
    if target.modal_environment:
        env["MODAL_ENVIRONMENT"] = target.modal_environment
    code = "from miles_ext.launch import modal_app; modal_app.spawn_detached()"
    started = time.time()
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=SPAWN_TIMEOUT_SECS, cwd=_package_root())
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("LAUNCHED ")), None)
    if proc.returncode != 0 or line is None:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        raise RuntimeError(f"launch of {run_id} failed (exit {proc.returncode}):\n{tail}")
    return {**json.loads(line[len("LAUNCHED "):]), "spawn_seconds": round(time.time() - started, 1)}


def _package_root() -> str:
    """The directory holding ``miles_ext`` (the child imports it from there, as Modal mounts it)."""
    import miles_ext

    return os.path.dirname(os.path.dirname(os.path.abspath(miles_ext.__file__)))
