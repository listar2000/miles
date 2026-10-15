"""R9: a run's checkpoints on the target's results volume: list, retain, delete, export to HF.

Layout (relative to the volume root; the trainer mounts it at ``recipe.RESULTS``)::

    <run_id>/checkpoints/iter_NNNNNNN/   full saves (optimizer state), for resuming
    <run_id>/checkpoints/latest_checkpointed_iteration.txt
    <run_id>/hf/iter_NNNNNNN/            HF weights: Megatron writes one per save
                                         (``checkpoint.save_hf``); FSDP gets one from
                                         ``finalize_hf`` (upstream tools/convert_fsdp_to_hf.py)

Every path is built here from a validated run id, an integer step and a kind,
and must resolve to one of the two layouts above: nothing outside
``<run_id>/checkpoints/iter_N`` or ``<run_id>/hf/iter_N`` can be deleted.
These calls use the Modal Volume API, so they run anywhere (no container).
"""

import re

import modal

from miles_ext.platform.layout import read_tracker, read_tracker_volume
from miles_ext.platform.layout import results_volume as _volume
from miles_ext.platform.target import Target
from miles_ext.recipes import recipe as R

KINDS = {"full": "checkpoints", "hf": "hf"}
ITER_RE = r"iter_(\d{7})"
# Miles' rollout executor snapshots its event log into <save>/iter_{rollout_id}/debug_events at
# each save; FSDP names its save iter_{rollout_id + 1}, so an FSDP run has dirs holding only that
NOT_A_SAVE = {"debug_events"}


def iter_dir(step: int) -> str:
    return f"iter_{int(step):07d}"


def checkpoint_path(run_id: str, step: int, kind: str) -> str:
    """``<run_id>/checkpoints/iter_N`` or ``<run_id>/hf/iter_N``; raises on anything else."""
    if not re.fullmatch(R.RUN_ID_RE, run_id or ""):
        raise ValueError(f"run id {run_id!r}")
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} must be one of {sorted(KINDS)}")
    if isinstance(step, bool) or int(step) != step or not 0 <= int(step) < 10**7:
        raise ValueError(f"step {step!r} must be an integer 0..9999999")
    path = f"{run_id}/{KINDS[kind]}/{iter_dir(step)}"
    if not re.fullmatch(rf"{re.escape(run_id)}/(checkpoints|hf)/{ITER_RE}", path) or ".." in path.split("/"):
        raise ValueError(f"refusing checkpoint path {path!r}")
    return path


def _bytes(volume: modal.Volume, path: str) -> int:
    return sum(e.size for e in volume.listdir(path, recursive=True) if e.type == modal.volume.FileEntryType.FILE)


def list_checkpoints(run_id: str, target: Target | dict | None, *, sizes: bool = True) -> list[dict]:
    """``[{step, kind, path, bytes}]``, newest first within each kind."""
    target = Target.of(target)
    volume, out = _volume(target), []
    for kind, sub in KINDS.items():
        checkpoint_path(run_id, 0, kind)  # validates the run id
        try:
            entries = volume.listdir(f"{run_id}/{sub}")
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
        for e in entries:
            m = re.fullmatch(ITER_RE, e.path.rstrip("/").split("/")[-1])
            if m:
                path = checkpoint_path(run_id, int(m.group(1)), kind)
                if not {c.path.rstrip("/").rsplit("/", 1)[-1] for c in volume.listdir(path)} - NOT_A_SAVE:
                    continue
                out.append({"step": int(m.group(1)), "kind": kind, "path": path,
                            "bytes": _bytes(volume, path) if sizes else None})
    return sorted(out, key=lambda c: (c["kind"], -c["step"]))


def delete_checkpoint(run_id: str, step: int, kind: str, target: Target | dict | None) -> dict:
    """Delete one save. Refuses the full save the run would resume from
    (``latest_checkpointed_iteration.txt``) unless ``apply_retention`` is finishing the run."""
    target = Target.of(target)
    path = checkpoint_path(run_id, step, kind)
    volume = _volume(target)
    if kind == "full" and read_tracker_volume(volume, run_id) == int(step):
        raise RuntimeError(f"{path} is the save this run resumes from; not deleting")
    return _remove(volume, path)


def _remove(volume: modal.Volume, path: str) -> dict:
    try:
        volume.remove_file(path, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return {"path": path, "deleted": False}
    return {"path": path, "deleted": True}


def retention_plan(saves: list[dict], *, keep: int, keep_final_full: bool, final: bool,
                   latest: int | None, pinned: tuple[int, ...] = ()) -> list[dict]:
    """Which saves to delete (pure; tested without Modal).

    While training (``final=False``): keep the newest ``keep`` full saves and the
    one the run resumes from; HF exports are kept. At the end (``final=True``):
    keep the final HF export, and the final full save if ``keep_final_full`` or if there
    is no HF export yet.
    Pinned steps are never deleted."""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    full = sorted((s for s in saves if s["kind"] == "full"), key=lambda s: -s["step"])
    hf = sorted((s for s in saves if s["kind"] == "hf"), key=lambda s: -s["step"])
    drop = []
    if not final:
        keep_steps = {s["step"] for s in full[:keep]} | ({latest} if latest is not None else set())
        drop = [s for s in full if s["step"] not in keep_steps]
    else:
        # the final full save stays when asked, and always while no HF export exists
        # (FSDP before finalize_hf): it is then the run's only copy of its weights
        keep_full = {full[0]["step"]} if (full and (keep_final_full or not hf)) else set()
        drop = [s for s in full if s["step"] not in keep_full] + hf[1:]
    return [s for s in drop if s["step"] not in set(pinned)]


def apply_retention(run_id: str, target: Target | dict | None, *, keep: int = 1, keep_final_full: bool = True,
                    final: bool = False, pinned: tuple[int, ...] = (), dry_run: bool = False) -> list[dict]:
    """Delete what ``retention_plan`` says; returns the deleted (or, dry run, the planned) saves."""
    target = Target.of(target)
    volume = _volume(target)
    saves = list_checkpoints(run_id, target, sizes=False)
    plan = retention_plan(saves, keep=keep, keep_final_full=keep_final_full, final=final,
                          latest=read_tracker_volume(volume, run_id), pinned=pinned)
    if not dry_run:
        for s in plan:
            _remove(volume, checkpoint_path(run_id, s["step"], s["kind"]))
    return plan


def apply_retention_local(run_dir, *, keep: int, keep_final_full: bool, final: bool,
                          pinned: tuple[int, ...] = ()) -> list[dict]:
    """``apply_retention`` inside a container that has the results volume mounted (the trainer)."""
    import shutil
    from pathlib import Path

    run_dir = Path(run_dir)
    saves = []
    for kind, sub in KINDS.items():
        for d in sorted((run_dir / sub).glob("iter_*")) if (run_dir / sub).is_dir() else []:
            if d.is_dir() and (m := re.fullmatch(ITER_RE, d.name)) and {c.name for c in d.iterdir()} - NOT_A_SAVE:
                saves.append({"step": int(m.group(1)), "kind": kind, "path": str(d)})
    latest = read_tracker(run_dir)
    plan = retention_plan(saves, keep=keep, keep_final_full=keep_final_full, final=final, latest=latest,
                          pinned=pinned)
    for s in plan:
        path = Path(s["path"]).resolve()
        if path.parent.parent != run_dir.resolve() or not re.fullmatch(ITER_RE, path.name):
            raise RuntimeError(f"refusing to delete {path}")
        shutil.rmtree(path)
    return plan


def hf_export_template(run_id: str) -> str:
    """Miles' ``--save-hf`` (Megatron): one HF export per save, formatted with ``rollout_id``."""
    checkpoint_path(run_id, 0, "hf")
    return str(R.RESULTS / run_id / "hf" / "iter_{rollout_id:07d}")


def finalize_hf_command(run_id: str, step: int, origin_hf_dir: str) -> list[str]:
    """The upstream converter, run in the fork image with the results volume at ``recipe.RESULTS``."""
    from miles_ext import stack

    src = R.RESULTS / checkpoint_path(run_id, step, "full")  # the converter finds <iter>/model itself
    return ["python", str(stack.MILES_SRC / "tools/convert_fsdp_to_hf.py"), "--input-dir", str(src),
            "--output-dir", str(R.RESULTS / checkpoint_path(run_id, step, "hf")), "--origin-hf-dir", origin_hf_dir,
            "--force"]


def finalize_hf(run_id: str, recipe: dict, target: Target | dict | None, *, step: int | None = None) -> dict:
    """FSDP: convert the final (or given) full save to HF weights at ``<run_id>/hf/iter_N``,
    in a CPU container of the fork image. Megatron runs already have HF exports (``checkpoint.save_hf``)."""
    from miles_ext.platform import functions

    target = Target.of(target)
    if R.backend(recipe) != "fsdp":
        raise ValueError("finalize_hf converts FSDP saves; Megatron writes HF weights itself (checkpoint.save_hf)")
    volume = _volume(target)
    step = step if step is not None else read_tracker_volume(volume, run_id)
    if step is None:
        raise RuntimeError(f"{run_id} has no full save to convert")
    cmd = finalize_hf_command(run_id, step, str(R.model_dir(recipe)))
    weights = modal.Volume.from_name(target.weights_volume, environment_name=target.modal_environment)
    fn = functions.convert_fsdp.with_options(volumes={R.RESULTS: volume, R.MODELS: weights})
    with functions.run(target):
        fn.remote(cmd)
    return {"step": step, "path": checkpoint_path(run_id, step, "hf")}
