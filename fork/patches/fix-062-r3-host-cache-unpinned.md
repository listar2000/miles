# fix-062: R3's host routing cache is plain host memory, not pinned

## Why
With routing replay (R3) on, every TP rank of an engine allocates a host buffer for the routed experts of every token
the KV pool can hold: 30.4-33.9 GB per rank on DeepSeek-V4.1-Flash (shape 34M x 40 layers x 6 experts, int32).
SGLang allocates it with mmap and pins it with `cudaHostRegister`. Pinning one buffer took 12.0 s on one B300 under
Modal's gVisor (a one-GPU probe: mmap 0.0 s, register 12.0 s, zero 1.6-2.2 s, and a side thread does not hide
it, since the main thread made no GPU progress while it ran). With eight TP ranks on a node the "KV cache
allocation" step of engine start took 68-78 s on every R3 launch from 10-20 to 10-27, almost all of it between
"Memory pool end" and "HostCache[routed_experts] allocated".

## Decision
Nothing copies to or from this buffer over DMA: `finalize()` scatters CPU tensors into it after the device-to-host
copy into separate staging tensors, and `get_topk()` gathers rows out of it on the CPU. Pinning buys nothing, so
the buffer is `torch.zeros(...)` on the CPU (it fills the buffer like the old `zero_()`, about 1.6-2.2 s, without
the register), and `destroy()` unregisters only a pinned
buffer. The indexer top-k capturer shares the class and gets the same change.
SGLang already keeps host caches unpinned when nothing needs DMA: `HostKVCache` (`mem_cache/memory_pool_host.py`,
`pin_memory` argument, lines 222-296 at `3a938213c1`) allocates its host buffers with `pin_memory=False` on request;
this patch makes the same choice for the capturer's buffer, unconditionally, since it is never a DMA target.
Lost: pinning on a background thread (it blocked GPU work in the probe); hugepages (needs host configuration Modal
does not give us).

## How it is applied
`fork/deps/sglang/fix-062-r3-host-cache-unpinned.patch`, applied by `fork/deps/sglang/apply.py` to SGLang
`3a938213c1`: `state_capturer/base.py`, marked `[fork fix-062]`. Verify: an R3 engine start's
`kv_cache_allocation` in "Engine startup timings" drops from ~70 s to a few seconds, and R3 rows stay non-zero.
Measured 10-28 on two split DeepSeek-V4.1 LoRA smokes (two engines each): `kv_cache_allocation` 2.9-3.1 s on all
four engine starts, against 68-78 s on the R3 launches of 10-20 to 10-27; every token row of the four saved samples
holds routed experts.

## Remove when
SGLang stops pinning the state-capturer host cache. Upstream status: not filed; still pinned on SGLang main
`8136cbce3d`, where this patch applies unchanged.
