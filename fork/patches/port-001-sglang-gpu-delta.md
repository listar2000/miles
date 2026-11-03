# port-001: SGLang GPU-delta receiver (sgl-project/sglang#42210)

## Why
Full-parameter runs in split placement (ext-036) have no NCCL link between trainer and engines; a full
sync through the results volume moves the whole served model (~321 GB for GLM-5.3-Flash FP8) every step.
GPU-delta (the humans& "weight teleportation" design) sends a compressed XOR delta instead and applies it in
place with Blackwell's hardware decompression engine. The PR is a WIP draft upstream; we need it before it merges.

## Decision
- Carried as a Port (fork/README.md, "Rules for ports"): the PR head `9747f67b11` diffed against its base
  `sglang-miles@fe89fc7a`, unchanged except one `[fork port-001]` line in each of the three edited files
  (`entrypoints/http_server.py`, `managers/scheduler.py`, `managers/tokenizer_control_mixin.py`).
- It applies cleanly at our pin `3a938213c1` (1,087 commits behind its base) together with our other patches,
  and every `sglang.*` name its new files import exists at the pin. No conflict resolutions were needed.
- Models: the PR maps DeepSeek V2/V3/V3.2 and `GlmMoeDsaForCausalLM` only. Our GLM-5.3-Flash (`glm5_next`)
  mapping is a separate change on top.
- Rejected: a `listar2000/sglang` fork (user prefers patch files); waiting for the upstream merge.

## How it is applied
`fork/deps/sglang/port-001-gpu-delta.patch`, applied by `apply.py` with the other patches. `apply.py` applies ports
first and then the rest in name order, one patch at a time, because a fix on top of a port (fix-074) edits a file the
port creates, which a single `git apply` of all files does not see; a failure reverts the patches already applied, so
the result is still all or none. Inert until a
caller uses `/update_weights_from_gpu_delta` (Miles side: port-002). Needs `nvidia-libnvcomp-cu13==6.x` at
runtime. Verify: `apply.py <sglang checkout> <SGLANG_COMMIT> --check`; the launcher's `--check-only` lists the
marker; the PR's own CUDA tests on a Modal B300 (results on fix-074's page). On a sync: if SGLang now has the PR, delete the patch and
retire port-001; if the PR head moved and we want it, refresh the patch in one commit with the new sha.

## Remove when
sgl-project/sglang#42210 is merged into `sglang-miles` and `SGLANG_COMMIT` moves past it; upstream status:
open draft PR (2026-11-03).
