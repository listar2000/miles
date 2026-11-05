# STAK overfit on DeepSeek-V4.1-Flash LoRA

RL on 32 fixed questions until the model answers nearly all of them, to check that the training loop can raise reward.
STAK is a made-up stack language: each question is a short program and the answer is its final stack. The model is
DeepSeek-V4.1-Flash with a rank-32 LoRA.

| Run (2026-10-26) | Reward |
| --- | --- |
| Batch 1 | 0.58 |
| Batch 11 | 0.97 |
| Batch 14 | 0.78, as more replies hit the 8k cap |

14 steps of about 13.3 min, about $485, on image `pr-ext038-d311deeb0f`; W&B `mobiuslab/miles-dsv41`, tag
`exp:overfit-stak`. This config is that run's, minus the `lora.serving` key that main has since removed; it has not run
on the current image.

## Data

[prompts/stak32.jsonl](prompts/stak32.jsonl): 32 prompts (16 at level 4, 16 at level 5), each passed some of the time by
the base model with thinking off (about 56% overall, `metadata.base_pass8`). The set was generated and hand-picked
elsewhere. Each row's `metadata.task` names the grader, [stackvm.py](stackvm.py)'s `grade`. The `glm53-flash-*` STAK
recipes read the same file with `MILES_DATA` set to this folder.

## Run

Needs the preset's weights and their BF16 copy on `MILES_WEIGHTS_VOLUME` ([quickstart](../../fork/guide/quickstart.md#5-weights-and-tasks)).

```bash
export MILES_RECIPE=experiments/dsv41_stak_overfit/configs/dsv41-flash-stak.yaml
modal run -m miles_ext.launch.modal_app --run-id <run-id>
```

Training: 2 B300:8 nodes (1 trainer, 1 rollout), fully async, LoRA r32 unmerged, LR 1e-4, PPO clip 0.2 / 0.28, DSpark,
32 prompts × 8 tries, T 1.0, top-p 0.95, top-k 32, 8k reply cap (a capped reply scores 0), thinking off.
