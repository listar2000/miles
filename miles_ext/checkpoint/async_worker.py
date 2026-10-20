"""Run Megatron's persistent async-save worker with sglang's torch patch applied (patch page ext-032)."""

import contextlib

from megatron.core.dist_checkpointing.strategies.async_utils import PersistentAsyncCaller

_MEGATRON_LOOP = PersistentAsyncCaller.async_loop


def _loop(*args) -> None:
    """The spawned worker's entry: the trainer pickles CUDA tensors with sglang's reducer, whose rebuild
    needs ``monkey_patch_torch_reductions`` in the receiving process too."""
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()
    _MEGATRON_LOOP(*args)


def install() -> None:
    """Make the worker Megatron spawns at the first async save start in ``_loop``."""
    PersistentAsyncCaller.async_loop = staticmethod(_loop)


@contextlib.contextmanager
def context(args, iteration: int):
    """``--custom-megatron-save-context-path``: ``install`` before each save (the first one spawns the worker)."""
    install()
    yield
