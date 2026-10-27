import pytest
import torch

pytest.importorskip("megatron.core")
from megatron.core import tensor_parallel

from miles_ext.memory import ckpt_offload

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


@pytest.fixture
def pool():
    pool = ckpt_offload._HostPool()
    yield pool
    pool.close()  # a freed but still registered range would fail the next registration that reuses it


def test_host_pool_hands_out_the_smallest_free_buffer_that_fits(pool):
    small, _ = pool.acquire(1 << 20)
    large, _ = pool.acquire(4 << 20)
    pool.release(small, torch.cuda.Event())  # the fitting buffer is not first in the free list
    pool.release(large, torch.cuda.Event())
    assert pool.acquire(2 << 20)[0] is large
    assert pool.acquire(1 << 20)[0] is small
    assert pool.total_bytes == 5 << 20
    pool.release(small, torch.cuda.Event())
    pool.release(large, torch.cuda.Event())


def test_host_pool_drops_buffers_too_small_for_a_longer_micro_batch(pool):
    for nbytes in (1 << 20, 2 << 20, 3 << 20, 4 << 20, 2 << 20):  # micro-batches of varying length
        buffers = [pool.acquire(nbytes)[0] for _ in range(2)]  # two checkpoints each
        for buffer in buffers:
            pool.release(buffer, torch.cuda.Event())
    assert pool.total_bytes == 2 * (4 << 20)


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


def test_copies_back_land_in_the_compute_streams_pool(monkeypatch, offloading):
    torch.cuda.empty_cache()
    _step(offloading)
    compute = torch.cuda.current_stream().cuda_stream
    assert {seg["stream"] for seg in torch.cuda.memory_snapshot()} == {compute}


def test_offload_composes_with_the_recompute_release(monkeypatch, offloading):
    """fix-011 (released recompute inputs) wraps the recompute's backward; the offload stays bit-identical."""
    from miles.backends.megatron_utils import initialize

    out_ref, grads_ref, _ = _step(_block)
    function = tensor_parallel.random.CheckpointFunction
    monkeypatch.setattr(function, "backward", function.backward)  # restored after the test
    initialize._release_recompute_inputs()  # as in init(); the offload installs later from the init path
    assert getattr(function.backward, "miles_release", False)

    for _ in range(2):
        out, grads, _ = _step(offloading)
        assert torch.equal(out, out_ref)
        assert all(torch.equal(a, b) for a, b in zip(grads, grads_ref, strict=True))
