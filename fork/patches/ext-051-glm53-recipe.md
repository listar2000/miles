# ext-051: GLM-5.3-Flash presets and a together-mode recipe

## Why
GLM-5.3-Flash (321 B, 34 KDA + 11 DSA layers, 288 experts) has upstream Miles support (radixark/miles#2786),
but only as a short-context, colocated math launcher (`scripts/run_glm5_3_flash.py`, 32 or 64 GB300s). Our
runs go through recipes, which had no GLM preset. The first target is full-parameter RL in one cluster
("together" mode): one B300:8 trainer node and one engine node, with the existing NCCL weight sync.
Full-parameter split mode is a separate line (its weight transport is not built yet).

## Decision
- Two presets. `glm-5.3-flash` serves the released checkpoint (FP8, 128x128 blocks). Miles' FP8
  quantizer reads that checkpoint's `quantization_config` (with its `modules_to_not_convert`) and re-encodes
  each sync. The trainer loads the same directory, and mbridge dequantizes it on load. So no BF16 copy
  and no custom quantizer (ext-007 is V4.1's FP8 + FP4 layout). `glm-5.3-flash-4layer` is upstream's BF16
  4-layer slice (`CharyZeng/GLM-5.3-Flash-4layer`), for smoke tests only.
- The flags are copied from the upstream launcher, which stays the source of truth: TITO `glm53`, parsers
  `glm47` / `glm45`, tilelang DSA backends, BF16 KV, `--moe-runner-backend triton` (R3 needs top-k ids that
  SM100's fused runner does not materialize), `--model-name glm5_next --qkv-format thd`, one sample per
  micro-batch. One upstream flag is dropped: `--sglang-disable-radix-cache`, because the session server
  shares one radix cache across an agent's turns.
- `glm53-flash-4layer-smoke.yaml`: 2 B300:8 nodes, trainer TP8 EP8 PP1 (the full model's one-node layout;
  CP stays 1 because GLM's DSA refuses context parallelism), two TP4 engines, fully async, R3, single-turn
  STAK prompts (no sandboxes).
- `glm53-flash-stak.yaml`: the full model, full-parameter, on the same 2-node shape. BF16 weights and bf16
  grads are ~160 GB per GPU. Adam is disk-streamed with bf16 moments, and `optimizer.host_gib` (170) of it per GPU
  stays in pinned host memory (fix-073), so only ~1.2 TB per node goes through the disk.
- Not done here: MTP speculative decoding (the release's NextN layer; no draft in the preset), the fused MoE
  runner, a BF16 rollout copy, and LoRA (upstream's native `glm5_next` LoRA spec would apply as is).

## How it is applied
`miles_ext/recipes/presets.py` (two presets), `miles_ext/recipes/recipe.py` (`optimizer.host_gib`),
`miles_ext/recipes/configs/glm53-flash-4layer-smoke.yaml` and `glm53-flash-stak.yaml` with their golden argv. Weights:
`prepare_weights("CharyZeng/GLM-5.3-Flash-4layer", "7fb7883207fe73ea04430ac392f6ffe2b9f936c4", ...)`.
The 4-layer slice has no `processor_config.json`; copy the full release's file into the snapshot dir (see the GPU-delta smoke notes).
Verify: `python -m pytest miles_ext/tests/test_placement_golden.py`, then the recipe's smoke run on Modal. On an upstream
sync, diff the preset flags against `scripts/run_glm5_3_flash.py`.

## Remove when
Upstream Miles ships recipe-level presets for GLM-5.3-Flash; upstream status: not applicable (recipes are ours).
