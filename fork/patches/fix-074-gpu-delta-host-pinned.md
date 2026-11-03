# fix-074: GPU-delta host buffers fall back to plain pinned host memory on Modal

## Why
port-001 (sgl-project/sglang#42210) gives each receiver GPU a host-memory input buffer that Blackwell's
hardware decompression engine reads directly. It allocates it with `cuMemCreate` at location `HOST_NUMA` with the
`HW_DECOMPRESS` usage flag. In a Modal B300 container (gVisor sandbox, driver 610.57.04) every `HOST_NUMA`
`cuMemCreate` returns `CUDA_ERROR_INVALID_VALUE`, with or without the usage flag, although the device reports
HOST_NUMA VMM support and NUMA id 0. With the PR as is, its own codec test fails 4 of 4
(`test/manual/gpu_delta/test_codec.py`: `GPU-delta allocation cuMemCreate failed: CUDA_ERROR_INVALID_VALUE`).
A one-GPU allocation probe in the same container showed: location `HOST` plus `HW_DECOMPRESS` is created, mapped and
reported `CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 1`; `cuMemHostAlloc` memory and `HOST` without the flag are
not capable; device memory from `cuMemAlloc` is.

## Decision
On `CUDA_ERROR_INVALID_VALUE`, retry once with location `HOST` (id 0) and the same usage flag, and use that location
for the host access descriptor and the property check. The PR's mapped-pointer capability check still decides
admission, so a host where `HOST_NUMA` works is unchanged. This mirrors the PR's own pattern: probe the driver, then
admit by the mapped pointer's capability. Rejected: staging the input in device memory (uses GPU memory next to the
KV cache and adds a copy, which the PR's design avoids).

## How it is applied
`fork/deps/sglang/fix-074-gpu-delta-host-pinned.patch` edits port-001's `weight_sync/gpu_delta/memory.py` and its
CPU unit test `test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_memory.py` (the test driver stub gains the
`HOST` location and a `host_numa_refused` case). `apply.py` applies ports first, so this lands on top of port-001.

Verification:
- `python3 fork/deps/sglang/apply.py <SGLang checkout at 3a938213c1> 3a938213c1b349e1a8df0b5fdc57ef36a41c743b --check`:
  all 11 patches apply in order and revert.
- CPU, in `starlimobius/miles-mobius:main` with the patched `weight_sync/` overlaid:
  `python3 -m pytest test/registered/unit/gpu_delta_weight_sync/test_gpu_delta_memory.py`: 16 passed.
- One Modal B300, upstream image `radixark/miles:dev-202611021910` with both PR heads, nvCOMP 6.0.0.12 and this
  patch: `test_codec.py` 4 passed, `test_apply_cuda.py` 7 passed, Miles' `tests/manual/gpu_delta/test_gpu_delta_nvcomp.py`
  PASS (zstd, snappy, lz4 round trips), `test_gpu_delta_encoder.py` 42 passed. `test_device_allocator_cuda.py` and
  `test_host_cuda.py` need two GPUs and were not run.
- The launcher's `--check-only` from the stack head finds the `[fork fix-074]` marker in the image's `memory.py`.

## Remove when
The PR or SGLang allocates this buffer without requiring HOST_NUMA. Upstream status: not filed; the code is not on
SGLang main (checked `4709a3a24`, no `weight_sync/gpu_delta/`); to be raised on sgl-project/sglang#42210.
