"""What every role of a node sets up the same way."""

import os
from importlib.metadata import version

from miles_ext.platform import compile_cache

# as upstream's launchers do: the image's PYTHONPATH would hide packages like megatron.post_training
MEGATRON_SRC = "/root/Megatron-LM"


def pythonpath() -> str:
    return ":".join(p for p in (os.environ.get("PYTHONPATH"), MEGATRON_SRC) if p)


def compile_cache_key(gpu: str) -> str:
    return compile_cache.key(gpu, version("sglang"))
