"""R5/R6: what a run reports while it trains and when it ends, and ``observe``.

Two places hold it, so a reader never depends on the trainer being alive:

- the progress Dict ``<namespace of app_name>progress`` (``mdev-progress`` in dev),
  keys ``run:<run_id>`` (rank 0's heartbeat, every ``HEARTBEAT_SECS``),
  ``lease:<run_id>`` (written by the caller, see ``renew_lease``) and
  ``exit:<run_id>`` (the exit record); teardown deletes all three;
- the results volume: ``<run_id>/exit.json`` (the newest exit record) and
  ``<run_id>/exits.jsonl`` (one line per segment), next to the checkpoints.

Whether a run is complete never needs either: it follows from the checkpoint
tracker and ``num_rollout`` (``next_rollout_id``), so a segment killed right
after its final save still reads as complete, and a relaunch of it returns
without starting a cluster.
"""

import json
import re
import time

import modal

from miles_ext.platform.target import Target, check_own_name, namespace_of
from miles_ext.recipes import recipe as R

HEARTBEAT_SECS = 30
STALE_HEARTBEAT_SECS = 5 * 60
TRACKER = "latest_checkpointed_iteration.txt"
EXIT_FILE = "exit.json"
EXITS_LOG = "exits.jsonl"

# exit reasons; the platform retries a run from its newest save on the RETRYABLE ones
REASONS = ("success", "segment_end", "lease_expired", "stopped", "oom", "nccl_timeout", "error")
RETRYABLE = ("segment_end", "lease_expired", "nccl_timeout")


def next_rollout_id(backend: str, tracker: int | None) -> int:
    """The rollout a resume starts at, from ``latest_checkpointed_iteration.txt``.

    The two backends write different numbers there: FSDP names a save after the
    next rollout (``iter_{rollout_id + 1}``, fsdp_utils/checkpoint.py), Megatron
    after the rollout it completed (``iter_{rollout_id}``, resumes at +1)."""
    if tracker is None:
        return 0
    if backend == "fsdp":
        return int(tracker)
    if backend == "megatron":
        return int(tracker) + 1
    raise ValueError(f"backend {backend!r}")


def is_complete(backend: str, tracker: int | None, num_rollout: int) -> bool:
    return next_rollout_id(backend, tracker) >= int(num_rollout)


def progress_dict_name(target: Target) -> str:
    return check_own_name(f"{namespace_of(target.app_name)}progress", "progress dict")


def keys(run_id: str) -> dict:
    if not re.fullmatch(R.RUN_ID_RE, run_id or ""):
        raise ValueError(f"run id {run_id!r}")
    return {"run": f"run:{run_id}", "lease": f"lease:{run_id}", "exit": f"exit:{run_id}"}


def progress_dict(target: Target, create: bool = True) -> modal.Dict:
    return modal.Dict.from_name(progress_dict_name(target), environment_name=target.modal_environment,
                                create_if_missing=create)


# ---- the caller's side ----------------------------------------------------------------------------

def renew_lease(run_id: str, target: Target | dict | None, ttl_seconds: float) -> dict:
    """The controller's lease: while it holds, the trainer keeps going; once ``expires_at``
    passes, rank 0 saves at the next step and exits (reason ``lease_expired``).
    A run whose lease key was never written has no lease (hand-launched runs)."""
    target = Target.of(target)
    lease = {"expires_at": time.time() + float(ttl_seconds), "renewed_at": time.time()}
    progress_dict(target)[keys(run_id)["lease"]] = lease
    return lease


def delete_keys(run_id: str, target: Target) -> list[str]:
    """Teardown: drop this run's keys from the progress Dict; returns the ones that existed."""
    d, gone = progress_dict(target), []
    for key in keys(run_id).values():
        if d.pop(key, None) is not None:
            gone.append(key)
    return gone


def _volume(target: Target) -> modal.Volume:
    return modal.Volume.from_name(target.results_volume, environment_name=target.modal_environment)


def _read_json(volume: modal.Volume, path: str):
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except (FileNotFoundError, modal.exception.NotFoundError, ValueError):
        return None


def _tracker(volume: modal.Volume, run_id: str) -> int | None:
    try:
        return int(b"".join(volume.read_file(f"{run_id}/checkpoints/{TRACKER}")).decode().strip())
    except (FileNotFoundError, modal.exception.NotFoundError, ValueError):
        return None


def _launch_record(volume: modal.Volume, run_id: str) -> dict | None:
    """The newest ``recipe-*.json`` the trainer wrote when a segment started."""
    try:
        names = sorted(e.path.rsplit("/", 1)[-1] for e in volume.listdir(run_id)
                       if re.fullmatch(r"recipe-\d{8}-\d{6}\.json", e.path.rsplit("/", 1)[-1]))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return _read_json(volume, f"{run_id}/{names[-1]}") if names else None


def completion(run_id: str, target: Target | dict | None, recipe: dict | None = None) -> dict:
    """``{complete, next_rollout_id, num_rollout, tracker}`` from the volume alone (no trainer needed).
    ``recipe`` gives ``num_rollout`` and the backend; without it, the newest launch record does."""
    target = Target.of(target)
    volume = _volume(target)
    if recipe is None:
        recipe = (_launch_record(volume, run_id) or {}).get("recipe")
    tracker = _tracker(volume, run_id)
    if recipe is None:
        return {"complete": None, "next_rollout_id": None, "num_rollout": None, "tracker": tracker}
    backend, num_rollout = R.backend(recipe), recipe["algorithm"]["num_rollout"]
    nxt = next_rollout_id(backend, tracker)
    return {"complete": nxt >= num_rollout, "next_rollout_id": nxt, "num_rollout": num_rollout, "tracker": tracker}


def observe(run_id: str, target: Target | dict | None, recipe: dict | None = None) -> dict:
    """One poll: ``{phase, step, next_rollout_id, num_rollout, complete, last_checkpoint, checkpoints,
    heartbeat_age, deadline, lease, metrics, exit}``.

    ``phase`` is ``training``, ``checkpointing`` or ``stopping`` from a live heartbeat;
    ``ended`` once an exit record exists for the current segment; ``no_heartbeat`` when
    the trainer has not reported yet (placement, startup) or stopped reporting
    (``heartbeat_age`` says for how long). ``complete`` comes from the checkpoints,
    so it holds even when the segment was killed before writing its exit record."""
    from miles_ext.platform.checkpoints import list_checkpoints

    target = Target.of(target)
    k, d = keys(run_id), progress_dict(target)
    beat = d.get(k["run"])
    lease = d.get(k["lease"])
    exit_record = d.get(k["exit"]) or _read_json(_volume(target), f"{run_id}/{EXIT_FILE}")
    done = completion(run_id, target, recipe)
    saves = list_checkpoints(run_id, target, sizes=False)
    full = [s["step"] for s in saves if s["kind"] == "full"]
    age = time.time() - beat["heartbeat_at"] if beat else None
    if beat and exit_record and exit_record.get("segment") == beat.get("segment"):
        phase = "ended"
    elif beat and beat.get("phase") == "ended":
        phase = "ended"
    elif beat and age is not None and age < STALE_HEARTBEAT_SECS:
        phase = beat.get("phase", "training")
    else:
        phase = "no_heartbeat"
    return {
        "phase": phase,
        "step": (beat or {}).get("step"),
        "next_rollout_id": done["next_rollout_id"],
        "num_rollout": done["num_rollout"],
        "complete": done["complete"],
        "last_checkpoint": max(full) if full else None,
        "checkpoints": saves,
        "heartbeat_age": age,
        "segment": (beat or {}).get("segment"),
        "deadline": (beat or {}).get("deadline"),
        "lease": lease,
        "metrics": (beat or {}).get("metrics", {}),
        "exit": exit_record,
    }
