"""Decide per Megatron save whether it carries Adam's state, and mark the saves that do (patch page ext-056)."""

import contextlib
import os
from pathlib import Path

OPTIM_EVERY_ENV = "MILES_EXT_SAVE_OPTIM_EVERY"  # checkpoint.optimizer_every
OPTIM_MARKER = ".optimizer"


def with_optimizer(args, iteration: int) -> bool:
    """Whether the save after rollout ``iteration`` (0-based) carries Adam's state: every k-th periodic save;
    a save off the interval (the final one, one the supervisor requested) holds weights only."""
    if args.no_save_optim:
        return False
    every = int(os.environ.get(OPTIM_EVERY_ENV, "1"))
    return every == 1 or (iteration + 1) % (every * (args.save_interval or 1)) == 0


def has_optimizer(iter_dir: str | Path) -> bool:
    return (Path(iter_dir) / OPTIM_MARKER).exists()


@contextlib.contextmanager
def context(args, iteration: int):
    """``--custom-megatron-save-context-path``: this save carries Adam's state only when ``with_optimizer`` says so."""
    optim = with_optimizer(args, iteration)
    no_optim, args.no_save_optim = args.no_save_optim, not optim
    try:
        yield
    finally:
        args.no_save_optim = no_optim
    import torch.distributed as dist

    if optim and dist.get_rank() == 0:
        (Path(args.save) / f"iter_{iteration:07d}" / OPTIM_MARKER).touch()
