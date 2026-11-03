"""GPU check for fix-070/071/072 (GLM-5.3 kpool indexer): bit-identical top-k and peak memory."""

import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/root/overlay")

import torch  # noqa: E402

import miles_plugins.models.glm5_next.ops.kpool_indexer as ki  # noqa: E402
from miles.kernels.attention.dsa.kpool import build_pooled_keys, pool_boundaries  # noqa: E402

assert ki.__file__.startswith("/root/overlay"), ki.__file__
H, D, TOPK, KPOOL, TP = 32, 128, 2048, 4, 8
GiB = 1 << 30
RESULTS = {"device": torch.cuda.get_device_name()}


def make_inputs(seq_lens, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cu = torch.tensor([0, *seq_lens], device="cuda").cumsum(0).to(torch.int32)
    T = int(cu[-1])
    kw = dict(device="cuda", generator=g)
    index_k = torch.randn(T, D, dtype=torch.bfloat16, **kw)
    gate = torch.randn(T, D, dtype=torch.bfloat16, **kw)
    ape = 0.1 * torch.randn(KPOOL, D, dtype=torch.float32, **kw)
    pooled = build_pooled_keys(index_k, gate, ape, cu, KPOOL)
    del index_k, gate
    return dict(
        index_q=torch.randn(T, H, D, dtype=torch.bfloat16, **kw),
        pooled_k=pooled,
        head_weights=torch.randn(T, H, dtype=torch.float32, **kw) * (H**-0.5 * D**-0.5),
        cu_seqlens=cu,
        pool_cu_seqlens=pool_boundaries(cu, KPOOL),
        index_topk=TOPK,
        kpool=KPOOL,
    )


def timed_peak(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() - base
    return out, round(peak / GiB, 3), round(dt, 3)


def select(inputs, chunk):
    ki._QUERY_CHUNK = chunk
    return ki.kpool_select_topk(**inputs)


class FakeDist:
    """One process plays the TP ranks: rank r's rows, the all-reduce done by summing the ranks' buffers."""

    def __init__(self, world):
        self.world, self.rank = world, 0

    def get_world_size(self, group):
        return self.world

    def get_rank(self, group):
        return self.rank

    def all_reduce(self, tensor, group):
        pass


def select_split(inputs, world, ranks=None):
    real = ki.dist
    fake = FakeDist(world)
    ki.dist = fake
    ki._QUERY_CHUNK = 8192
    try:
        total, own_rows = None, []
        for r in ranks if ranks is not None else range(world):
            fake.rank = r
            rows = ki.kpool_select_rows(**inputs, split_group=object())
            own_rows.append(rows.tokens.shape[0])
            placed = ki.gather_query_rows(rows)
            total = placed if total is None else total + placed
        return total, own_rows
    finally:
        ki.dist = real


def parity(seq_lens, seed):
    inputs = make_inputs(seq_lens, seed)
    whole = select(inputs, 1 << 30)
    chunked = select(inputs, 8192)
    split, own = select_split(inputs, TP)
    res = dict(
        seq_lens=seq_lens,
        tokens=whole.shape[0],
        width=whole.shape[-1],
        chunked_identical=bool(torch.equal(chunked, whole)),
        split_identical=bool(torch.equal(split, whole)),
        chunked_rows_differing=int((chunked != whole).any(-1).sum()),
        split_rows_differing=int((split != whole).any(-1).sum()),
        rows_per_rank=own,
        selected_ids=int((whole >= 0).sum()),
    )
    print(json.dumps(res), flush=True)
    return res


def memory(T, try_whole):
    inputs = make_inputs([T], seed=1)
    res = dict(tokens=T, pools=int(inputs["pooled_k"].shape[0]), dense_scores_gib=round(T * (T // KPOOL) * 4 / GiB, 2))
    chunked, res["chunked_peak_gib"], res["chunked_s"] = timed_peak(lambda: select(inputs, 8192))
    out_gib = chunked.numel() * 4 / GiB
    res["topk_out_gib"] = round(out_gib, 3)
    (_, own), res["split_rank0_peak_gib"], res["split_rank0_s"] = timed_peak(
        lambda: select_split(inputs, TP, ranks=[0])
    )
    res["split_rank0_rows"] = own[0]
    if try_whole:
        try:
            whole, res["whole_peak_gib"], res["whole_s"] = timed_peak(lambda: select(inputs, 1 << 30))
            res["chunked_identical_to_whole"] = bool(torch.equal(chunked, whole))
            del whole
        except torch.OutOfMemoryError as e:
            res["whole"] = f"OOM: {str(e).splitlines()[0]}"
    split, _ = select_split(inputs, TP)
    res["split_identical_to_chunked"] = bool(torch.equal(split, chunked))
    print(json.dumps(res), flush=True)
    del chunked, split
    torch.cuda.empty_cache()
    return res


def recompute_flags():
    """Megatron's checkpoint: is_checkpointing() in the no_grad forward and in the grad recompute (fix-072)."""
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.tensor_parallel.random import is_checkpointing

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    seen = []

    def fn(x):
        seen.append((is_checkpointing(), torch.is_grad_enabled()))
        return x * 2

    x = torch.randn(4, device="cuda", requires_grad=True)
    tensor_parallel.checkpoint(fn, False, x).sum().backward()
    import miles_plugins.models.glm5_next.dsa as dsa  # real Megatron imports

    res = dict(checkpoint_calls=seen, dsa_has_index_rows=hasattr(dsa.Glm5NextDSAAttention, "_index_rows"))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    RESULTS["parity"] = [
        parity([16384], 0),
        parity([20000, 8000, 4768], 1),
        parity([32768], 2),
        parity([3000, 61000, 1536], 3),
    ]
    RESULTS["memory"] = [memory(65536, True), memory(262144, True)]
    RESULTS["recompute"] = recompute_flags()
    print("RESULTS " + json.dumps(RESULTS), flush=True)
