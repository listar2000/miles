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
  grads are ~160 GB per GPU. Adam is disk-streamed with bf16 moments. The pinned host tier (fix-073) is not a recipe
  key: `optimizer_host_gib` derives it from the trainer node's `cluster.memory_gib`, minus `TRAINER_HOST_RESERVE_GIB`
  (Ray, the trainer processes, checkpoint loading) and, with GPU-delta sync, three copies of the served model (the
  preset's `served_gib`: port-002's base, rolling and pending snapshots), split over the fullest trainer node's GPUs.
  It is an upper bound: whole buckets go there while they fit.
  - Evidence for the reserve: the [trainer-only GLM run](https://github.com/listar2000/miles-ops/blob/2e2313fe8594ef20163207e4817cd539e0722268/notes/glm53/README.md) ran with 2,304 GiB, a 170 GiB budget per GPU and 129 GiB
    per GPU resident, so the other host users took at most about 1,270 GiB; 1,100 keeps the derived 150 GiB per GPU at
    the same node size below the tested 170.
  - The GPU-delta term gives 35 GiB per GPU on 2,304 GiB. In the [overnight split GPU-delta runs](https://github.com/listar2000/miles-ops/blob/2e2313fe8594ef20163207e4817cd539e0722268/notes/glm53/README.md) on that node, a budget
    of 90 (49.5 GiB per GPU resident) had the trainer container killed for exceeding its memory request in step 2, after
    the third pinned served-model copy appeared; 40 was reached only through step 1 (those runs failed first in the
    engine's arena mapping, fix-079). So 35 is below the known failure but untested past one step.
  - Why derive: a hand-set value per recipe has to be re-done whenever another host user joins the node.
- Not done here: MTP speculative decoding (the release's NextN layer; no draft in the preset), the fused MoE
  runner, a BF16 rollout copy, and LoRA (upstream's native `glm5_next` LoRA spec would apply as is).

## How it is applied
`miles_ext/recipes/presets.py` (two presets), `miles_ext/recipes/recipe.py` (`optimizer_host_gib`),
`miles_ext/recipes/configs/glm53-flash-4layer-smoke.yaml` and `glm53-flash-stak.yaml` with their golden argv. Weights:
`prepare_weights("CharyZeng/GLM-5.3-Flash-4layer", "7fb7883207fe73ea04430ac392f6ffe2b9f936c4", ...)`. The slice has no
`processor_config.json`, and SGLang loads one for this multimodal architecture: without it the engines exit at startup.
Copy the full release's file into the snapshot dir, as upstream's `run_glm5_3_flash_lora.py prepare` does:
`hf download zai-org/GLM-5.3-Flash processor_config.json --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c --local-dir .`, then
`modal volume put <weights volume> processor_config.json CharyZeng--GLM-5.3-Flash-4layer/7fb7883207fe73ea04430ac392f6ffe2b9f936c4/`.
Verify: `python -m pytest miles_ext/tests/test_placement_golden.py miles_ext/tests/test_optimizer_host_gib.py`, then the recipe's smoke run on Modal. On an upstream
sync, diff the preset flags against `scripts/run_glm5_3_flash.py`.

## Remove when
Upstream Miles ships recipe-level presets for GLM-5.3-Flash; upstream status: not applicable (recipes are ours).
