# STAK overfit on DeepSeek-V4.1-Flash LoRA

Our only clean overfit so far, kept as a repeatable experiment: a rank-32 LoRA on DeepSeek-V4.1-Flash raises reward on 32
fixed prompts. Started as loose files on 2026-10-26; formalized here 2026-11-05.

## Data

`prompts/stak32.jsonl`: 32 made-up stack-language prompts (16 level 4, 16 level 5). With thinking off the base model passes
about 56%, and it passed each prompt some of the time (`metadata.base_pass8`). `stackvm.py` holds the grader, interpreter and
generator. The set was hand-picked from generated tasks, so regenerating gives different prompts. `metadata.task` is the
grader's dotted path, which `miles_ext.tasks.single_turn` imports.

Other single-turn recipes (`glm53-flash-4layer-smoke`, `glm53-flash-split-gpud`, `glm53-flash-stak`) can use this file as
their payload with `MILES_DATA` set to this folder. Their code does not import this experiment.

## Recipes

`stak-nosc.yaml` (PPO clip) and `stak-sc.yaml` (score centering), otherwise the same: 2 B300:8 nodes (1 trainer, 1 rollout),
LoRA r32 kept unmerged, LR 1e-4, DSpark, top-p 0.95 / top-k 32, 32 prompts x 8 tries, 8k reply cap, thinking off.

The only change from the 10-26 files: the `lora.serving` line is gone, since main removed that key on 2026-10-30 and LoRA
runs always serve the adapter. These files have not run on the current image; the results below used the image named there.

## Reproduce

    export MILES_WEIGHTS_VOLUME=miles-dsv41-ckpt MILES_RECIPE=experiments/dsv41_stak_overfit/stak-nosc.yaml
    modal run -m miles_ext.launch.modal_app --run-id <id>

## Results

| Run | Image | Reward | Notes |
| --- | --- | --- | --- |
| `overfit-stak-nosc-1026-a1` | `pr-ext038-d311deeb0f` | 0.58 to 0.97 by batch 11 | 14 steps, then stopped at the money cap; 0.78 at batch 14 as replies passed the 8k cap, which scores 0 |
| `overfit-stak-sc-1026-a1` | same | 0.58 to 0.93 by batch 6 | 13 steps; 0.29 at batch 10, 0.92 by batch 13 |

About 13.3 min per step and ~$485 per run. W&B `mobiuslab/miles-dsv41`, tag `exp:overfit-stak`.
