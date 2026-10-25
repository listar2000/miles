"""Where a run's files are, and the handles to read them: one place for the paths.

On the results volume (mounted at ``recipe.RESULTS`` in the trainer)::

    <run_id>/checkpoints/latest_checkpointed_iteration.txt   the tracker (``TRACKER``)
    <run_id>/checkpoints/iter_NNNNNNN/                       full saves
    <run_id>/checkpoints/iter_NNNNNNN/adapter/               LoRA saves (no tracker written)
    <run_id>/checkpoints/rollout/<rollout_id>/               rollout-side state per save
    <run_id>/hf/iter_NNNNNNN/                                HF exports
    <run_id>/recipe-YYYYmmdd-HHMMSS.json                     one launch record per segment
    <run_id>/exit.json, exits.jsonl                          exit records (``progress``)

The tracker is read here, from the trainer's local mount (``read_tracker``) or
through the Volume API (``read_tracker_volume``); nothing else parses it. A LoRA run
writes no tracker, so for one these return the step Miles resumes it from instead.
"""

from collections.abc import Callable
from pathlib import Path

import modal

from miles_ext.platform.target import Target

TRACKER = "latest_checkpointed_iteration.txt"


def tracker_path(run_id: str) -> str:
    return f"{run_id}/checkpoints/{TRACKER}"


def read_tracker(run_dir: str | Path) -> int | None:
    """The step a run under a mounted run directory resumes from, or None before the first save."""
    ckpt = Path(run_dir) / "checkpoints"
    try:
        return int((ckpt / TRACKER).read_text().strip())
    except (OSError, ValueError):
        pass
    try:
        rollout_ids = [p.name for p in (ckpt / "rollout").iterdir()]
    except OSError:
        return None
    return lora_resume_step(rollout_ids, lambda step: (ckpt / iter_dir(step) / "adapter").is_dir())


def read_tracker_volume(volume: modal.Volume, run_id: str) -> int | None:
    try:
        return int(b"".join(volume.read_file(tracker_path(run_id))).decode().strip())
    except (FileNotFoundError, modal.exception.NotFoundError, ValueError):
        pass

    def names(path: str) -> list[str]:
        try:
            return [e.path.rstrip("/").rsplit("/", 1)[-1] for e in volume.listdir(path)]
        except (FileNotFoundError, modal.exception.NotFoundError):
            return []

    return lora_resume_step(
        names(f"{run_id}/checkpoints/rollout"),
        lambda step: "adapter" in names(f"{run_id}/checkpoints/{iter_dir(step)}"),
    )


def lora_resume_step(rollout_ids: list[str], has_adapter: Callable[[int], bool]) -> int | None:
    """The newest step with both ``rollout/<N>`` and ``iter_N/adapter`` saved, the rule fork hook-006 resumes
    a LoRA run by (``_lora_resume_iteration``, miles/backends/megatron_utils/megatron_config.py)."""
    steps = sorted((int(r) for r in rollout_ids if r.isdigit()), reverse=True)
    return next((step for step in steps if has_adapter(step)), None)


def iter_dir(step: int) -> str:
    return f"iter_{int(step):07d}"


def results_volume(target: Target) -> modal.Volume:
    return modal.Volume.from_name(target.results_volume, environment_name=target.modal_environment)


def weights_volume(target: Target, create: bool = False) -> modal.Volume:
    return modal.Volume.from_name(target.weights_volume, environment_name=target.modal_environment,
                                  create_if_missing=create)
