"""Full-recompute layer inputs in pinned host memory from forward to backward [ext-020].

Install with ``--custom-megatron-init-path miles_ext.memory.ckpt_offload.install``. Each checkpointed hidden state is
copied to host on a side stream after it is saved and back one layer ahead in backward; results are bit-identical.
"""

import logging
import threading

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)

_state = threading.local()


class _HostPool:
    """Exact-size page-locked buffers, reused across steps; the pinned caching allocator rounds up to a power of two."""

    def __init__(self):
        self._free: list[tuple[torch.Tensor, torch.cuda.Event | None]] = []
        self._lock = threading.Lock()
        self.total_bytes = 0

    def acquire(self, nbytes: int) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        with self._lock:
            fits = [i for i, (buffer, _) in enumerate(self._free) if buffer.numel() >= nbytes]
            if fits:
                return self._free.pop(min(fits, key=lambda i: self._free[i][0].numel()))
            # Nothing free is big enough: drop the free buffers that are too small before growing, so micro-batches of
            # varying length keep the pool at the in-flight working set instead of one set of buffers per new length.
            self._drop_free()
            self.total_bytes += nbytes
        buffer = torch.empty(nbytes, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), nbytes, 0)
        assert int(err) == 0, f"cudaHostRegister of {nbytes} bytes failed: {err}"
        return buffer, None

    def release(self, buffer: torch.Tensor, done: torch.cuda.Event) -> None:
        with self._lock:
            self._free.append((buffer, done))

    def close(self) -> None:
        """Unregister the free buffers; buffers still handed out stay registered."""
        with self._lock:
            self._drop_free()

    def _drop_free(self) -> None:
        dropped, self._free = self._free, []
        for buffer, done in dropped:
            if done is not None:
                done.synchronize()
            torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
            self.total_bytes -= buffer.numel()


_pool = _HostPool()


def _fetch(host: torch.Tensor, device, ready: torch.cuda.Event, h2d: torch.cuda.Stream) -> torch.Tensor:
    """Copies host back on h2d into a block of the compute stream's pool, so backward can reuse it once freed."""
    gpu = torch.empty_like(host, device=device)
    h2d.wait_stream(torch.cuda.current_stream())  # the block may be one the compute stream just freed
    h2d.wait_event(ready)
    with torch.cuda.stream(h2d):
        gpu.copy_(host, non_blocking=True)
    gpu.record_stream(h2d)
    return gpu


class _Saved:
    """One offloaded checkpoint: the pinned host copy, and the GPU copy once prefetched."""

    def __init__(self, gpu: torch.Tensor, streams):
        self.device = gpu.device
        nbytes = gpu.numel() * gpu.element_size()
        self.buffer, last_use = _pool.acquire(nbytes)
        self.host = self.buffer[:nbytes].view(gpu.dtype).view(gpu.shape)
        self.gpu = None
        self.ready = torch.cuda.Event()
        d2h = streams[0]
        d2h.wait_stream(torch.cuda.current_stream())
        if last_use is not None:
            d2h.wait_event(last_use)
        with torch.cuda.stream(d2h):
            self.host.copy_(gpu, non_blocking=True)
            self.ready.record()
        gpu.record_stream(d2h)

    def prefetch(self, h2d: torch.cuda.Stream):
        if self.gpu is not None or self.host is None:
            return
        self.gpu = _fetch(self.host, self.device, self.ready, h2d)
        self.ready = torch.cuda.Event()
        self.ready.record(h2d)

    def take(self, h2d: torch.cuda.Stream) -> torch.Tensor:
        self.prefetch(h2d)
        current = torch.cuda.current_stream()
        current.wait_event(self.ready)
        _pool.release(self.buffer, self.ready)
        gpu, self.gpu, self.host, self.buffer = self.gpu, None, None, None
        gpu.record_stream(current)
        return gpu


class _Microbatch:
    """The checkpoints of one _checkpointed_forward call, in save order; backward takes them in reverse."""

    def __init__(self, streams):
        self.streams = streams
        self.saved: list[_Saved] = []

    def pack(self, tensor: torch.Tensor):
        saved = _Saved(tensor, self.streams)
        self.saved.append(saved)
        return len(self.saved) - 1

    def unpack(self, index: int) -> torch.Tensor:
        h2d = self.streams[1]
        tensor = self.saved[index].take(h2d)
        if index > 0:
            self.saved[index - 1].prefetch(h2d)
        return tensor


def _log_once_per_shape(microbatch: _Microbatch) -> None:
    """Pinned bytes per microbatch: times the in-flight microbatches, the host RAM a stage needs."""
    if not microbatch.saved:
        return
    key = tuple(tuple(s.host.shape) for s in microbatch.saved if s.host is not None)
    logged = _state.__dict__.setdefault("logged", set())
    if key not in logged:
        logged.add(key)
        gib = sum(s.host.numel() * s.host.element_size() for s in microbatch.saved if s.host is not None) / 2**30
        logger.info(
            "ckpt offload: %d checkpoints, %.2f GiB per microbatch; %.2f GiB pinned in total",
            len(key),
            gib,
            _pool.total_bytes / 2**30,
        )


def _streams(device: torch.device):
    cache = _state.__dict__.setdefault("streams", {})
    if device not in cache:
        cache[device] = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
    return cache[device]


def _offloading_checkpoint(original):
    def checkpoint(function, distribute_saved_activations, *args):
        microbatch = getattr(_state, "microbatch", None)
        hidden = args[0] if args else None
        if microbatch is None or not torch.is_tensor(hidden) or not hidden.is_cuda or not torch.is_grad_enabled():
            return original(function, distribute_saved_activations, *args)

        # autograd keeps the hooks alive with the graph, so they hold only the identity of the input, not the input
        target = (hidden.data_ptr(), hidden.shape)

        def pack(tensor):
            if (tensor.data_ptr(), tensor.shape) == target and tensor.is_contiguous():
                return microbatch, microbatch.pack(tensor)
            return None, tensor

        def unpack(packed):
            owner, value = packed
            return value if owner is None else owner.unpack(value)

        _state.microbatch = None  # nested checkpoints inside the layer keep their tensors on the GPU
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                return original(function, distribute_saved_activations, *args)
        finally:
            _state.microbatch = microbatch

    return checkpoint


def _offloading_forward(original):
    def _checkpointed_forward(self, hidden_states, *args, **kwargs):
        if not hidden_states.is_cuda:
            return original(self, hidden_states, *args, **kwargs)
        microbatch = _Microbatch(_streams(hidden_states.device))
        _state.microbatch = microbatch
        try:
            return original(self, hidden_states, *args, **kwargs)
        finally:
            _state.microbatch = None
            _log_once_per_shape(microbatch)

    return _checkpointed_forward


def install(args=None) -> None:
    """``--custom-megatron-init-path`` entry point; idempotent. bf16 path only, not TE's ``te_checkpoint``."""
    if getattr(TransformerBlock._checkpointed_forward, "miles_ckpt_offload", False):
        return
    TransformerBlock._checkpointed_forward = _offloading_forward(TransformerBlock._checkpointed_forward)
    TransformerBlock._checkpointed_forward.miles_ckpt_offload = True
    tensor_parallel.checkpoint = _offloading_checkpoint(tensor_parallel.checkpoint)
    logger.info("full-recompute layer-input checkpoints are kept in pinned host memory (ext-020)")
