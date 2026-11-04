"""What the SFT Modal apps share: the run config named by ``MILES_SFT_CONFIG``, the image (the config's, with this
checkout's ``miles_ext/`` and ``experiments/`` laid over it) and the volumes."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from miles_ext import stack
from miles_ext.sft import config as sft_config

REPO = Path(__file__).resolve().parents[2]
OVERLAY = ("miles_ext", "experiments")

CONFIG_PATH = os.environ.get("MILES_SFT_CONFIG")
# inside the container the function's own image is used; the config is only read where the app is built
CONFIG = sft_config.load(CONFIG_PATH) if CONFIG_PATH else None


def image() -> modal.Image:
    if CONFIG is None:
        return modal.Image.debian_slim()
    img = (
        modal.Image.from_registry(CONFIG.image)
        .entrypoint([])
        .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{stack.MILES_SRC}:/root/Megatron-LM"})
    )
    for rel in OVERLAY:
        if (REPO / rel).is_dir():
            img = img.add_local_dir(REPO / rel, f"{stack.MILES_SRC}/{rel}", ignore=["**/__pycache__/**"])
    return img


def volume(name_env: str) -> modal.Volume:
    name = os.environ.get(name_env)
    if CONFIG is not None and not name:
        raise RuntimeError(f"{name_env} is not set")
    return modal.Volume.from_name(name or "unset", version=2)


def data_volume() -> modal.Volume:
    return modal.Volume.from_name(CONFIG.data.volume if CONFIG else "unset", version=2)
