# ext-053: full-parameter split placement with GPU-delta weight sync

## Why
Split placement (ext-036) carries only LoRA adapters through the results volume; a full-parameter sync would
move the whole served model each step (344 GB through the volume took 337 s: 113 s to write and commit, 222 s
to read, before any GPU load; probe on 2026-11-03). GPU-delta (port-001, port-002) sends a compressed XOR
delta instead; a GLM-5.3-Flash FP8 step is estimated at about 1.5 to 4 GB (CPU simulation on a release shard).

## Decision
- One recipe key, `rollout.weight_sync: tensor | gpu-delta` (default `tensor`: every existing recipe's argv is
  unchanged, checked by the placement golden test). `gpu-delta` needs `cluster.placement split`, `lora.rank 0`,
  `async.pause_generation retract` (Miles' admission) and a preset in `GPU_DELTA_PRESETS` (those with an SGLang
  receiver mapping: GLM-5.3-Flash, hook-018).
- The trainer publishes with Miles' own protocol to `/results/<run>/gpu_delta`; the engines are reached through
  the existing relays (`--rollout-external-engine-addrs`). Miles' `--custom-update-weight-post-write-path` hook
  (`volume_sync.gpu_delta_post_write`) has rank 0 commit the volume through the trainer's function process, as
  ext-037 does, and call the engine container agent's new `/reload`; Miles' collective check after the hook
  holds the other ranks. No custom protocol (hook-013 stays unused in this mode).
- `SGLANG_ENABLE_JIT_DEEPGEMM=0` for trainer and engines, so dense FP8 layers keep FP32 block scales on both
  sides (hook-018).
- Full-model defaults, from the first full GLM-5.3-Flash split runs (2026-11-03):
  `GPU_DELTA_CODEC` and `GPU_DELTA_INITIAL_SYNC_CODEC` are `lz4` (the outer Zstd pass asked for 46 GiB of
  scratch HBM beside a full-parameter trainer), `GPU_DELTA_DEVICE_ARENA=1` (fix-079: Modal cannot map
  HW-decompress host arenas past a small size) and `--update-weights-timeout 5400` (a first sync outlasts
  Miles' 600 s).
- The engines serve a trainer-format base, not the release (RFC 0004): against the release the first delta
  is ~240 GB. Recipe: `glm53-flash-split-gpud`: the trainer container asks for 2,560 GiB (Modal's per-container cap), since GPU-delta
  pins three served-model copies (~0.9 TB) beside Adam's derived host share (fix-073); 2,304 GiB ran out in step 2.
- Rejected: posting bytes through the tunnel (~10 MB/s); NCCL between the containers (ext-036).

## How it is applied
`miles_ext/recipes/recipe.py` (key, checks, `_split_args`, `trainer_env`), `miles_ext/launch/split/volume_sync.py`
(`gpu_delta_post_write`), `miles_ext/launch/split/agent.py` (`/reload`). Tests: `miles_ext/tests/test_split_gpu_delta.py`.
Recipe `miles_ext/recipes/configs/glm53-flash-split-gpud.yaml`; its base is built as RFC 0004 describes.

## Remove when
The launcher no longer needs Miles-side orchestration for separate engine placement (as ext-036).
