"""Names and images shared by the SFT data, training and export functions."""

from pathlib import Path

import modal

from miles_ext import stack

BASE_REPO = "Qwen/Qwen3-30B-A3B-Thinking-2507"
BASE_REVISION = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
BASE_DIR = "Qwen3-30B-A3B-Thinking-2507"
DATASET_REPO = "nvidia/Nemotron-SFT-SWE-v3"
DATASET_REVISION = "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a"
DATASET_DIR = "sft/nemotron-swe-v3-qwen3-strict64k"

WEIGHTS_VOLUME = "miles-rl-weights"
DATA_VOLUME = "miles-rl-datasets"
RESULTS_VOLUME = "miles-rl-results"
WEIGHTS_MOUNT = "/models"
DATA_MOUNT = "/datasets"
RESULTS_MOUNT = "/results"

# main@d011f17c96, the commit this branch starts from; the branch's own files are overlaid on it
FORK_IMAGE = "docker.io/starlimobius/miles-mobius@sha256:9e6d2256be24f9ebb2a86a7e0cc3eb4aeea46925b768079bddc351fba65e9af0"
REPO = Path(__file__).resolve().parents[2]
# files this branch adds or changes, copied over the image's checkout at container start
OVERLAY = ["miles_ext"]


def overlay(image: modal.Image) -> modal.Image:
    for rel in OVERLAY:
        src = REPO / rel
        dst = f"{stack.MILES_SRC}/{rel}"
        image = image.add_local_dir(src, dst, ignore=["**/__pycache__/**"]) if src.is_dir() else \
            image.add_local_file(src, dst)
    return image


def base_image() -> modal.Image:
    return modal.Image.from_registry(FORK_IMAGE).entrypoint([]).env({"PYTHONUNBUFFERED": "1"})


def cpu_image() -> modal.Image:
    return overlay(base_image())
