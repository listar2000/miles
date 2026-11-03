# port-002: Miles GPU-delta publication and receiver coordination (radixark/miles#3894)

## Why
Full-parameter runs in split placement (ext-036) need a weight sync that does not push the whole served model
through the results volume each step. radixark/miles#3894 is the trainer side of GPU-delta ("weight
teleportation"): PP-local owners XOR old and new canonical weights, compress on the trainer GPUs
(Snappy/LZ4 + Zstd) and publish framed files that SGLang (port-001) applies in place. It is a WIP draft upstream.

## Decision
- Carried as a Port: the PR head `7b2d749358` diffed against its base, upstream main `286634cc` (294 commits
  past our v0.1.3), applied with `git apply --3way`. The 30 new files are the PR's bytes (code, fast tests,
  `tests/manual/gpu_delta/` benchmark). `docker/Dockerfile` is left out: the nvCOMP 6 wheel goes into the image deps.
- Conflict resolutions, all toward our code:
  - `weight_update/protocol.py`: keeps hook-013's `custom_weight_transfer_protocol_path`; adds `gpu-delta` to the modes.
  - `utils/arguments.py`: keeps hook-014's imports; the PR's `utils/args/configs/{rollout,scaling}.py` hunks
    (the `gpu-delta` choice and help text) go inline, since our base has no `args/configs` layer.
  - `megatron_utils/actor.py`: keeps our `save(rollout_id, ...)` signature, `args.save`, `args.async_save` and
    our post-save-hook call; adds the PR's checkpoint-delta companion and `finish_checkpoint_delta`.
  - `utils/eval_config.py`: `uses_synced_eval` without the PR base's `args.custom_function` import.
- Base differences inside the PR's code are fixed on top, not in the port: fix-075.

## How it is applied
Opt-in: `--update-weight-transfer-mode gpu-delta` (plus `--update-weight-disk-dir`, a local HF checkpoint, Megatron
raw exporter, `retract` pause, no LoRA; `miles/utils/gpu_delta/admission.py`). Nothing changes for other modes.
Verify: the PR's fast tests on CPU in our image (`tests/fast/{backends/megatron_utils,backends/training_utils/weight_update,utils}/gpu_delta`),
gate E0 for the native path. On a sync: if upstream merged #3894, take upstream's side of every file listed in
`Touches:` and retire port-002 and fix-075; if the PR head moved and we want it, refresh in one commit.

## Remove when
radixark/miles#3894 is merged and our main has synced past it; upstream status: open draft PR (2026-11-03).
