"""Run one distributed Megatron checkpoint probe under torchrun (driven by ckpt_async_probe.py)."""

import argparse
import inspect
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

TENSOR_ELEMS = 128 << 20  # 256 MiB of bf16


def _state_dict(gib: int, rank: int, world: int) -> tuple[dict, float]:
    torch.manual_seed(rank)
    sd, total = {}, 0.0
    for i in range((gib << 30) // (TENSOR_ELEMS * 2)):
        t = torch.randn(TENSOR_ELEMS, device="cuda", dtype=torch.bfloat16)
        total += t.float().sum().item()
        sd[f"w{i}"] = ShardedTensor.from_rank_offsets(f"w{i}", t, (0, rank, world))
    return sd, total


def _finalize(queue: AsyncCallsQueue, poll: bool) -> None:
    while poll and queue.get_num_unfinalized_calls():
        queue.maybe_finalize_async_calls(blocking=False)
        if queue.get_num_unfinalized_calls():
            time.sleep(5)
    queue.maybe_finalize_async_calls(blocking=True)


def _save(
    sd: dict, out: str, threads: int, use_async: bool, queue: AsyncCallsQueue | None, poll: bool
) -> tuple[float, float]:
    strategy = TorchDistSaveShardedStrategy("torch_dist", 1)
    strategy.thread_count = threads
    kwargs = {"async_sharded_save": use_async}
    if "async_strategy" in inspect.signature(dist_checkpointing.save).parameters:
        kwargs["async_strategy"] = os.environ.get("ASYNC_STRATEGY", "nvrx")
    dist.barrier()
    t0 = time.monotonic()
    req = dist_checkpointing.save(sd, out, strategy, **kwargs)
    if use_async:
        queue.schedule_async_request(req)
    if dist.get_rank() == 0:
        print(f"rank 0: save to {out} returned after {time.monotonic() - t0:.1f}s", flush=True)
    dist.barrier()
    blocking = time.monotonic() - t0
    if use_async:
        _finalize(queue, poll)
        if dist.get_rank() == 0:
            print(f"rank 0: async save finalized after {time.monotonic() - t0:.1f}s", flush=True)
    dist.barrier()
    return blocking, time.monotonic() - t0 - blocking


def _load_checksum(gib: int, out: str, rank: int, world: int) -> float:
    sd = {}
    for i in range((gib << 30) // (TENSOR_ELEMS * 2)):
        t = torch.empty(TENSOR_ELEMS, device="cuda", dtype=torch.bfloat16)
        sd[f"w{i}"] = ShardedTensor.from_rank_offsets(f"w{i}", t, (0, rank, world))
    loaded = dist_checkpointing.load(sd, out)
    return sum(v.float().sum().item() for v in loaded.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gib", type=int, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--async-save", action="store_true")
    p.add_argument("--persistent-worker", action="store_true", help="Megatron's persistent async worker process")
    p.add_argument("--repeat", type=int, default=1, help="saves in a row (the 2nd reuses cached host buffers)")
    p.add_argument("--poll-finalize", action="store_true", help="finalize in short rounds (fix-043)")
    p.add_argument("--nccl-timeout-min", type=int, default=10, help="process group timeout")
    p.add_argument(
        "--sglang-patch",
        action="store_true",
        help="apply sglang's torch patch here, as Miles' weight sync does, and start the worker through ext-032",
    )
    p.add_argument("--verify", action="store_true", help="load the last save back and compare checksums")
    a = p.parse_args()

    if a.sglang_patch:
        from miles_ext.checkpoint import async_worker
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        async_worker.install()
    dist.init_process_group("nccl", timeout=timedelta(minutes=a.nccl_timeout_min))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    sd, checksum = _state_dict(a.gib, rank, world)
    queue = AsyncCallsQueue(persistent=a.persistent_worker) if a.async_save else None
    saves = []
    for r in range(a.repeat):
        out = f"{a.out}/save{r}"
        if rank == 0:
            os.makedirs(out, exist_ok=True)
        dist.barrier()
        blocking, background = _save(sd, out, a.threads, a.async_save, queue, a.poll_finalize)
        saves.append({"blocking_s": round(blocking, 1), "background_s": round(background, 1)})
    res = {
        "case": a.case,
        "ranks": world,
        "gib_per_rank": a.gib,
        "threads": a.threads,
        "async": a.async_save,
        "poll_finalize": a.poll_finalize,
        "sglang_patch": a.sglang_patch,
        "nccl_timeout_min": a.nccl_timeout_min,
        "saves": saves,
        "write_gbs_total": round(world * (a.gib << 30) / 1e9 / sum(saves[-1].values()), 3),
    }
    if a.verify:
        del sd  # the load needs the GPU memory back
        torch.cuda.empty_cache()
        res["checksum_ok"] = abs(_load_checksum(a.gib, out, rank, world) - checksum) <= 1e-3 * max(1.0, abs(checksum))
        ok = torch.tensor([int(res["checksum_ok"])], device="cuda")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        res["checksum_ok"] = bool(ok.item())
    if rank == 0:
        print("CASE " + json.dumps(res), flush=True)
    if queue is not None:
        queue.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
