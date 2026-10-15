"""Where a run's files are, and the handles to read them: one place for the paths.

On the results volume (mounted at ``recipe.RESULTS`` in the trainer)::

    <run_id>/checkpoints/latest_checkpointed_iteration.txt   the tracker (``TRACKER``)
    <run_id>/checkpoints/iter_NNNNNNN/                       full saves
    <run_id>/hf/iter_NNNNNNN/                                HF exports
    <run_id>/recipe-YYYYmmdd-HHMMSS.json                     one launch record per segment
    <run_id>/exit.json, exits.jsonl                          exit records (``progress``)

The tracker is read here, from the trainer's local mount (``read_tracker``) or
through the Volume API (``read_tracker_volume``); nothing else parses it.
"""

from pathlib import Path

import modal

from miles_ext.platform.target import Target

TRACKER = "latest_checkpointed_iteration.txt"


def tracker_path(run_id: str) -> str:
    return f"{run_id}/checkpoints/{TRACKER}"


def read_tracker(run_dir: str | Path) -> int | None:
    """The tracker under a mounted run directory, or None before the first save."""
    try:
        return int((Path(run_dir) / "checkpoints" / TRACKER).read_text().strip())
    except (OSError, ValueError):
        return None


def read_tracker_volume(volume: modal.Volume, run_id: str) -> int | None:
    try:
        return int(b"".join(volume.read_file(tracker_path(run_id))).decode().strip())
    except (FileNotFoundError, modal.exception.NotFoundError, ValueError):
        return None


def results_volume(target: Target) -> modal.Volume:
    return modal.Volume.from_name(target.results_volume, environment_name=target.modal_environment)


def weights_volume(target: Target, create: bool = False) -> modal.Volume:
    return modal.Volume.from_name(target.weights_volume, environment_name=target.modal_environment,
                                  create_if_missing=create)
