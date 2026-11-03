# fix-079: GPU-delta stages its decompression input through device memory where the host mapping is refused

## Why
port-001 maps each receiver rank's encoded payload into host memory the hardware decompression engine reads
directly (`cuMemCreate` + `cuMemSetAccess`; location HOST under fix-074). In Modal's gVisor sandbox that mapping fails
with `CUDA_ERROR_OUT_OF_MEMORY` once it grows past a small size: in the first full FP8 GLM-5.3-Flash split run with
GPU-delta, every engine rank refused its first real delta (~490 MB per owner) at `cuMemSetAccess`, though the
container had 1.5 TiB of RAM. Device memory from `cuMemAlloc` is decompression-capable on the same hosts (fix-074's
capability probe).

## Decision
When `HostAllocation` fails with CUDA result 2 (`CUDA_ERROR_OUT_OF_MEMORY`; `memory._check` now keeps the code on the
error), or with `GPU_DELTA_DEVICE_ARENA=1`, the rank arena becomes `memory.DeviceStagedArena`: an ordinary host buffer
the CPU decode workers fill, plus a same-size DE-capable device buffer allocated lazily on the receiver thread (where
the CUDA context is current). Before the pause the receiver copies the arena's used extent (the rank's inner payload
after outer Zstd decode, not the owner file size) into it, and the decoder reads device memory; `codec.prepare_batches`
accepts that CUDA input. Any other allocation error still fails.

The pattern is port-001's own for raw-byte tensors (`receiver.py`, `PreparedDelta`): CPU workers fill a host staging
buffer, one H2D copy on the receiver's stream before the pause moves it to device memory, and the copy is counted in
the receipt's `h2d_bytes` (as `raw_h2d_bytes`). fix-079 applies it to the compressed input and reuses port-001's
`memory.device_empty` allocator. What is new is the lifetime: the device image belongs to the arena's capacity
generation (allocated on first use, freed with the arena, replaced when the arena grows).

Cost, per rank per update: one H2D copy of the used extent (receipt `arena_staging_h2d_bytes`, included in
`h2d_bytes`), and HBM outside the KV pool equal to the arena capacity (`device_staged_arena_bytes`; capacity rounds up
and doubles on growth). `host_rank_device_staged` is 1 on every update served from a staged arena, reused or not.
Rejected: a smaller arena (it holds the rank's whole encoded publication), chunked host mappings (the cap's size is
unknown and may differ per host).

## How it is applied
`fork/deps/sglang/fix-079-gpu-delta-device-arena.patch` edits port-001's `gpu_delta/{memory,host,receiver,codec}.py`
and adds a host-arena test. Evidence, with commands and image identity: [evidence/fix-079/](evidence/fix-079/).

- Regression test `test_out_of_memory_host_mapping_stages_through_device_memory` (environment variable unset,
  `HostAllocation` failing through `memory._check` with code 2): passes with fix-079 and fails with its first
  revision, which matched `OUT_OF_MEMORY` in the error text. In our image the code prints as
  `<CUresult.CUDA_ERROR_OUT_OF_MEMORY: 2>` (`cuda-result.txt`), so that match held there; the code match does not depend
  on how the enum prints.
- GLM 4-layer FP8 benchmark (#145's `glm_bench.py`) on B300:8, this revision, one run each: with the staged arena
  forced, every rank copied its ~320 MB used extent into the device image and the run PASSED (selected outputs equal a
  fresh load on every route; warm pauses 68 and 40 ms); unforced, Modal admitted the 320 MB host mapping, nothing was
  staged, and it PASSED with warm pauses 62 and 75 ms. Pause is the maximum over ranks of the scheduler pause; the
  first update, which loads kernels, is excluded.
- The device-staged full-model split run (full FP8 GLM-5.3-Flash, the failing run's recipe with the staged arena
  forced, first revision) applied its step-1 delta in place on both engines (~530 MB per owner). It is not the same
  publication as the failing run's (~490 MB per owner), so the pair shows the fallback works at full size, not an
  A/B of fix-079 alone.

## Remove when
The PR (or SGLang) offers a device-staged or chunked input arena itself; upstream status: not filed, to be raised
on sgl-project/sglang#42210 (not on SGLang main `4709a3a24`, which has no `weight_sync/gpu_delta/`).
