# STAK overfit on DeepSeek-V4.1-Flash LoRA

Our only clean overfit so far: training on 32 fixed questions until the model answers nearly all of them. An overfit
checks that the training loop can raise reward at all. STAK is a made-up stack language: each question is a short program,
and the answer is its final stack. The model is DeepSeek-V4.1-Flash with a rank-32 LoRA, a small trainable adapter over
frozen weights. First run 2026-10-26 from loose files; kept here as an example since 2026-11-05.

## Data

`prompts/stak32.jsonl`: 32 made-up stack-language prompts (16 level 4, 16 level 5). With thinking off the base model passes
about 56%, and it passed each prompt some of the time (`metadata.base_pass8`). `stackvm.py` holds the grader, interpreter and
generator. The set was hand-picked from generated tasks, so regenerating gives different prompts. `metadata.task` is the
grader's dotted path, which `miles_ext.tasks.single_turn` imports.

Other single-turn recipes (`glm53-flash-4layer-smoke`, `glm53-flash-split-gpud`, `glm53-flash-stak`) can use this file as
their payload with `MILES_DATA` set to this folder. Their code does not import this experiment.

To make a new task set: `python stackvm.py make --out tasks.jsonl --per-level 32 --seed 1026 --levels 4,5`.

## Recipes

`stak-nosc.yaml` (PPO clip) and `stak-sc.yaml` (score centering), otherwise the same: 2 B300:8 nodes (1 trainer, 1 rollout),
LoRA r32 kept unmerged, LR 1e-4, DSpark, top-p 0.95 / top-k 32, 32 prompts x 8 tries, 8k reply cap, thinking off.

The only change from the 10-26 files: the `lora.serving` line is gone, since main removed that key on 2026-10-30 and LoRA
runs always serve the adapter. These files have not run on the current image; the results below used the image named there.

## Reproduce

    export MILES_WEIGHTS_VOLUME=miles-dsv41-ckpt MILES_RECIPE=experiments/dsv41_stak_overfit/stak-nosc.yaml
    modal run -m miles_ext.launch.modal_app --run-id <id>

## Results

| Score centering | Reward | Notes |
| --- | --- | --- |
| off | 0.58 to 0.97 by batch 11 | 14 steps, then stopped at the money cap; 0.78 at batch 14 as replies passed the 8k cap, which scores 0 |
| on | 0.58 to 0.93 by batch 6 | 13 steps, then stopped at the money cap; 0.29 at batch 10, 0.92 by batch 13 |

Both ran on image `pr-ext038-d311deeb0f`. About 13.3 min per step and ~$485 per run. W&B `mobiuslab/miles-dsv41`, tag `exp:overfit-stak`.
