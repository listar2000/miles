import pytest
import torch

pytest.importorskip("megatron.core")
from megatron.core import tensor_parallel

from miles_ext.memory import ckpt_offload, recompute_offload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

LAYERS, ROWS, COLS = 6, 4096, 1024  # 16 MiB fp32 per checkpoint


def _block(self, hidden, weights):
    for w in weights:
        hidden = tensor_parallel.checkpoint(lambda x, w=w: torch.tanh(x @ w) + x, False, hidden)
    return hidden


def _step(forward):
    torch.manual_seed(0)
    x = torch.randn(ROWS, COLS, device="cuda", requires_grad=True)
    weights = [(torch.randn(COLS, COLS, device="cuda") / COLS**0.5).requires_grad_() for _ in range(LAYERS)]
    before = torch.cuda.memory_allocated()
    out = forward(None, x * 1.0, weights)
    torch.cuda.synchronize()
    torch.empty(1, device="cuda")  # frees the blocks the side-stream copies were holding
    held = torch.cuda.memory_allocated() - before
    out.square().sum().backward()
    return out.detach(), [x.grad] + [w.grad for w in weights], held


@pytest.fixture
def offloading(monkeypatch):
    monkeypatch.setattr(tensor_parallel, "checkpoint", ckpt_offload._offloading_checkpoint(tensor_parallel.checkpoint))
    return ckpt_offload._offloading_forward(_block)


def test_offload_is_bit_identical_and_frees_the_checkpoints(offloading):
    out_ref, grads_ref, held_ref = _step(_block)
    out, grads, held = _step(offloading)

    assert torch.equal(out, out_ref)
    assert all(torch.equal(a, b) for a, b in zip(grads, grads_ref, strict=True))
    # every layer's input (16 MiB each) left the GPU
    assert held_ref - held >= LAYERS * ROWS * COLS * 4


def test_host_pool_hands_out_the_smallest_free_buffer_that_fits():
    pool = ckpt_offload._HostPool()
    small, _ = pool.acquire(1 << 20)
    large, _ = pool.acquire(4 << 20)
    pool.release(small, torch.cuda.Event())  # the fitting buffer is not first in the free list
    pool.release(large, torch.cuda.Event())
    assert pool.acquire(2 << 20)[0] is large
    assert pool.acquire(1 << 20)[0] is small
    assert pool.total_bytes == 5 << 20


def test_host_buffers_are_reused_across_steps(offloading):
    _step(offloading)
    pinned = ckpt_offload._pool.total_bytes
    _step(offloading)
    assert ckpt_offload._pool.total_bytes == pinned


def test_no_grad_forward_saves_nothing_and_is_untouched(offloading):
    x = torch.randn(ROWS, COLS, device="cuda")
    weights = [torch.randn(COLS, COLS, device="cuda") / COLS**0.5 for _ in range(LAYERS)]
    with torch.no_grad():
        assert torch.equal(offloading(None, x, weights), _block(None, x, weights))


def _deep_block(self, hidden, weights):
    def layer(x):
        for w in weights:
            x = torch.tanh(x @ w)
        return x

    return tensor_parallel.checkpoint(layer, False, hidden)


def _recompute_step():
    torch.manual_seed(0)
    x = torch.randn(2 * ROWS, COLS, device="cuda", requires_grad=True)  # 32 MiB activations
    weights = [(torch.randn(COLS, COLS, device="cuda") / COLS**0.5).requires_grad_() for _ in range(LAYERS)]
    out = _deep_block(None, x * 1.0, weights)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out.square().sum().backward()
    peak = torch.cuda.max_memory_allocated() - base
    return [x.grad] + [w.grad for w in weights], peak


@pytest.mark.parametrize("resident", [0, 2])
def test_recompute_offload_is_bit_identical_and_bounds_what_backward_holds(monkeypatch, resident):
    grads_ref, peak_ref = _recompute_step()
    function = tensor_parallel.random.CheckpointFunction
    act = 2 * ROWS * COLS * 4
    monkeypatch.setattr(recompute_offload, "WINDOW_BYTES", 2 * act)
    monkeypatch.setattr(recompute_offload, "RESIDENT_BYTES", resident * act)
    monkeypatch.setattr(function, "backward", recompute_offload._offloading_backward(function.backward))
    seen = []
    unpack = recompute_offload._Window.unpack

    def watched(self, index):
        seen.append((len(self.items), self.oldest, self.ahead))
        return unpack(self, index)

    monkeypatch.setattr(recompute_offload._Window, "unpack", watched)
    grads, peak = _recompute_step()

    assert all(torch.equal(a, b) for a, b in zip(grads, grads_ref, strict=True))
    # x @ w saves x and tanh saves its output, the next layer's x: one host copy per activation, not two
    assert seen[0][0] == LAYERS + 1
    assert seen[0][1] == LAYERS + 1 - resident  # the last `resident` activations never left the GPU
    assert max(ahead for _, _, ahead in seen) <= 3 * act
    assert peak < peak_ref


def test_copies_back_land_in_the_compute_streams_pool(monkeypatch, offloading):
    function = tensor_parallel.random.CheckpointFunction
    monkeypatch.setattr(function, "backward", recompute_offload._offloading_backward(function.backward))
    torch.cuda.empty_cache()
    _step(offloading)
    _recompute_step()
    compute = torch.cuda.current_stream().cuda_stream
    assert {seg["stream"] for seg in torch.cuda.memory_snapshot()} == {compute}
