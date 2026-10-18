"""The recomputed layer's saved activations in pinned host memory during its backward [ext-020].

Install with ``--custom-megatron-init-path miles_ext.memory.recompute_offload.install`` (also installs ckpt_offload).
Saved non-parameter tensors of at least ``MIN_BYTES`` go to host as they are saved, except the newest
``MILES_RECOMPUTE_OFFLOAD_RESIDENT_GIB``, and come back in reverse save order with
``MILES_RECOMPUTE_OFFLOAD_WINDOW_GIB`` (default 8 for both) prefetched; results are bit-identical.
"""

import logging
import os
import weakref

import torch
from megatron.core.tensor_parallel import random as tp_random

from miles_ext.memory import ckpt_offload

logger = logging.getLogger(__name__)

MIN_BYTES = 32 << 20
WINDOW_BYTES = int(float(os.environ.get("MILES_RECOMPUTE_OFFLOAD_WINDOW_GIB", "8")) * 2**30)
# the most recently saved bytes stay on the GPU (backward needs them first) and count against the window
RESIDENT_BYTES = min(WINDOW_BYTES, int(float(os.environ.get("MILES_RECOMPUTE_OFFLOAD_RESIDENT_GIB", "8")) * 2**30))


class _Item:
    def __init__(self, tensor: torch.Tensor):
        self.device, self.shape, self.dtype = tensor.device, tensor.shape, tensor.dtype
        self.nbytes = tensor.numel() * tensor.element_size()
        self.gpu = tensor  # until offloaded
        self.buffer = self.host = self.ready = None
        self.refs = 1

    def offload(self, d2h: torch.cuda.Stream) -> None:
        self.buffer, last_use = ckpt_offload._pool.acquire(self.nbytes)
        self.host = self.buffer[: self.nbytes].view(self.dtype).view(self.shape)
        self.ready = torch.cuda.Event()
        d2h.wait_stream(torch.cuda.current_stream())
        if last_use is not None:
            d2h.wait_event(last_use)
        with torch.cuda.stream(d2h):
            self.host.copy_(self.gpu, non_blocking=True)
            self.ready.record()
        self.gpu.record_stream(d2h)
        self.gpu = None

    def fetch(self, h2d: torch.cuda.Stream) -> None:
        self.gpu = ckpt_offload._fetch(self.host, self.device, self.ready, h2d)
        self.ready = torch.cuda.Event()
        self.ready.record(h2d)

    def release(self) -> None:
        if self.buffer is not None:
            ckpt_offload._pool.release(self.buffer, self.ready)
            self.buffer = self.host = None


class _Window:
    """The offloaded activations of one layer recompute, in save order; backward prefetches from the end."""

    def __init__(self, streams):
        self.d2h, self.h2d = streams
        self.items: list[_Item] = []
        self.alive: dict[tuple, tuple[weakref.ref, int]] = {}
        self.next = -1  # highest index not yet fetched
        self.oldest = 0  # lowest index not yet offloaded
        self.ahead = 0  # bytes on the GPU (resident or fetched) and not yet consumed

    def pack(self, tensor: torch.Tensor):
        if (
            not tensor.is_cuda
            or tensor.numel() * tensor.element_size() < MIN_BYTES
            or isinstance(tensor, torch.nn.Parameter)
            or isinstance(tensor._base, torch.nn.Parameter)
            or not tensor.is_contiguous()
        ):
            return None, tensor
        key = (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tensor.dtype)
        seen = self.alive.get(key)
        if seen is not None and seen[0]() is tensor:  # saved twice by the same tensor object: one copy
            self.items[seen[1]].refs += 1
            return self, seen[1]
        self.items.append(_Item(tensor))
        self.alive[key] = (weakref.ref(tensor), len(self.items) - 1)
        self.next = len(self.items) - 1
        self.ahead += self.items[-1].nbytes
        while self.ahead > RESIDENT_BYTES and self.oldest < len(self.items):
            self.items[self.oldest].offload(self.d2h)
            self.ahead -= self.items[self.oldest].nbytes
            self.oldest += 1
        return self, len(self.items) - 1

    def unpack(self, index: int) -> torch.Tensor:
        item = self.items[index]
        if item.gpu is None:
            item.fetch(self.h2d)
            self.ahead += item.nbytes
        gpu = item.gpu
        if item.buffer is not None:  # copied back, not resident
            current = torch.cuda.current_stream()
            current.wait_event(item.ready)
            gpu.record_stream(current)
        item.refs -= 1
        if item.refs == 0:
            item.gpu = None
            self.ahead -= item.nbytes
            item.release()
        self._prefetch()
        return gpu

    def _prefetch(self) -> None:
        while self.next >= 0 and self.ahead < WINDOW_BYTES:
            item = self.items[self.next]
            if item.gpu is None and item.buffer is not None:
                item.fetch(self.h2d)
                self.ahead += item.nbytes
            self.next -= 1

    def close(self) -> None:
        for item in self.items:  # saved but never needed by backward
            item.gpu = None
            item.release()


def _hook(packed):
    owner, value = packed
    return value if owner is None else owner.unpack(value)


def _offloading_backward(original):
    def backward(ctx, *args):
        window = _Window(ckpt_offload._streams(torch.cuda.current_device()))
        try:
            # backward runs on autograd's thread; hooks entered elsewhere would not apply to the recompute
            with torch.autograd.graph.saved_tensors_hooks(lambda t: window.pack(t), _hook):
                return original(ctx, *args)
        finally:
            window.close()

    return staticmethod(backward)


def install(args=None) -> None:
    """``--custom-megatron-init-path`` entry point; idempotent."""
    ckpt_offload.install(args)
    function = tp_random.CheckpointFunction
    if getattr(function, "miles_recompute_offload", False):
        return
    function.backward = _offloading_backward(function.backward)
    function.miles_recompute_offload = True
    logger.info(
        "recomputed activations are kept in pinned host memory, %.1f GiB prefetch window (ext-020)",
        WINDOW_BYTES / 2**30,
    )
