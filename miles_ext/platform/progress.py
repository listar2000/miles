"""R5/R6: what a run reports while it trains and when it ends, and ``observe``.

Two places hold it, so a reader never depends on the trainer being alive:

- the progress Dict ``<scope>progress`` (``rl-dev-progress``; ``mdev-progress`` before option A),
  keys ``run:<run_id>`` (``launch`` writes it as ``pending`` with the segment's call
  id, then rank 0's heartbeat replaces it every ``HEARTBEAT_SECS``),
  ``active:<run_id>`` (the live segment's call id: at most one segment per run),
  ``lease:<run_id>`` (written by the caller, see ``renew_lease``) and
  ``exit:<run_id>`` (the exit record) and ``commit:<run_id>`` (multi-node commit
  requests, see ``shards``); teardown deletes all of them;
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

from miles_ext.platform.layout import read_tracker_volume
from miles_ext.platform.layout import results_volume as _volume
from miles_ext.platform.target import Target, check_own_name, scope_of
from miles_ext.recipes import recipe as R

HEARTBEAT_SECS = 30
STALE_HEARTBEAT_SECS = 5 * 60
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
    return check_own_name(f"{scope_of(target)}progress", "progress dict")


def keys(run_id: str) -> dict:
    if not re.fullmatch(R.RUN_ID_RE, run_id or ""):
        raise ValueError(f"run id {run_id!r}")
    return {"run": f"run:{run_id}", "active": f"active:{run_id}", "lease": f"lease:{run_id}",
            "exit": f"exit:{run_id}", "commit": f"commit:{run_id}"}


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


def call_running(call_id: str | None) -> bool:
    """Whether a Modal function call (a segment) is still running."""
    if not call_id:
        return False
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return True
    except Exception:  # finished with an error, cancelled, or unknown
        return False
    return False


def delete_keys(run_id: str, target: Target) -> list[str]:
    """Teardown: drop this run's keys from the progress Dict; returns the ones that existed."""
    d, gone = progress_dict(target), []
    for key in keys(run_id).values():
        if d.pop(key, None) is not None:
            gone.append(key)
    return gone


def _read_json(volume: modal.Volume, path: str):
    try:
        return json.loads(b"".join(volume.read_file(path)))
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


def rollouts_done(backend: str, iteration: int) -> int:
    """How many rollouts a checkpoint holds (what a person means by its step): FSDP's
    ``iter_N`` holds N, Megatron's holds N + 1. The iter number stays the path key."""
    return next_rollout_id(backend, iteration)


def completion(run_id: str, target: Target | dict | None, recipe: dict | None = None) -> dict:
    """``{complete, next_rollout_id, num_rollout, tracker}`` from the volume alone (no trainer needed).
    ``recipe`` gives ``num_rollout`` and the backend; without it, the newest launch record does."""
    target = Target.of(target)
    volume = _volume(target)
    if recipe is None:
        recipe = (_launch_record(volume, run_id) or {}).get("recipe")
    tracker = read_tracker_volume(volume, run_id)
    if recipe is None:
        return {"complete": None, "next_rollout_id": None, "num_rollout": None, "tracker": tracker}
    backend, num_rollout = R.backend(recipe), recipe["algorithm"]["num_rollout"]
    nxt = next_rollout_id(backend, tracker)
    return {"complete": nxt >= num_rollout, "next_rollout_id": nxt, "num_rollout": num_rollout, "tracker": tracker}


def phase_of(beat: dict | None, exit_record: dict | None, now: float) -> str:
    """``pending`` (launched, no heartbeat yet: placement, image pull), ``starting``,
    ``training``, ``checkpointing``, ``stopping`` (live heartbeat), ``ended`` (the exit record
    belongs to the segment the heartbeat is from, or only an exit record is left after
    teardown), ``no_heartbeat`` (the trainer stopped reporting; ``heartbeat_age`` says how long)."""
    if beat is None:
        return "ended" if exit_record else "no_heartbeat"
    if beat.get("phase") == "pending":
        return "pending"
    if beat.get("phase") == "ended":
        return "ended"
    if exit_record and beat.get("call_id") and exit_record.get("call_id") == beat.get("call_id"):
        return "ended"
    if now - beat.get("heartbeat_at", 0) < STALE_HEARTBEAT_SECS:
        return beat.get("phase", "training")
    return "no_heartbeat"


def observe(run_id: str, target: Target | dict | None, recipe: dict | None = None) -> dict:
    """One poll: ``{phase, call_id, step, next_rollout_id, num_rollout, complete, last_checkpoint,
    rollouts_done, checkpoints, heartbeat_age, segment, deadline, lease, metrics, exit}``.

    ``phase``: see ``phase_of``. ``call_id`` is the segment the heartbeat (or pending marker)
    is from; ``exit`` is the newest exit record, which belongs to that segment only when
    its ``call_id`` matches: relaunch on an exit only for the call you launched.
    ``complete`` comes from the checkpoints, so it holds even when the segment was killed
    before writing its exit record. Checkpoint entries carry ``rollouts_done``."""
    from miles_ext.platform.checkpoints import list_checkpoints

    target = Target.of(target)
    if recipe is None:
        recipe = (_launch_record(_volume(target), run_id) or {}).get("recipe")
    k, d = keys(run_id), progress_dict(target)
    beat = d.get(k["run"])
    lease = d.get(k["lease"])
    exit_record = d.get(k["exit"]) or _read_json(_volume(target), f"{run_id}/{EXIT_FILE}")
    done = completion(run_id, target, recipe)
    saves = list_checkpoints(run_id, target, sizes=False)
    backend = R.backend(recipe) if recipe else None
    for s in saves:  # an HF export is named like the full save it came from, on both backends
        s["rollouts_done"] = rollouts_done(backend, s["step"]) if backend else None
    full = [s["step"] for s in saves if s["kind"] == "full"]
    now = time.time()
    age = now - beat["heartbeat_at"] if beat and beat.get("heartbeat_at") else None
    return {
        "phase": phase_of(beat, exit_record, now),
        "call_id": (beat or {}).get("call_id"),
        "step": (beat or {}).get("step"),
        "next_rollout_id": done["next_rollout_id"],
        "num_rollout": done["num_rollout"],
        "complete": done["complete"],
        "last_checkpoint": max(full) if full else None,
        "rollouts_done": rollouts_done(backend, max(full)) if full and backend else None,
        "checkpoints": saves,
        "heartbeat_age": age,
        "segment": (beat or {}).get("segment"),
        "deadline": (beat or {}).get("deadline"),
        "lease": lease,
        "metrics": (beat or {}).get("metrics", {}),
        "exit": exit_record,
    }
