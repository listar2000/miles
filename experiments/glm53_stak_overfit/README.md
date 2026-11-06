# STAK overfit on GLM-5.3-Flash LoRA

RL on 16 fixed STAK questions until the model answers most of them, to check that the training loop raises reward on
GLM. DeepSeek did this with thinking off ([dsv41_stak_overfit](../dsv41_stak_overfit/README.md)); GLM can't turn
thinking off, so this runs at `reasoning_effort: low` with a longer reply cap on an easier prompt set.

## Data

[prompts/stak_l4_16.jsonl](prompts/stak_l4_16.jsonl) is the 16 level-4 rows of `dsv41_stak_overfit/prompts/stak32.jsonl`.
The rows' `metadata.base_pass8` is DeepSeek's base rate, not GLM's. The grader is that experiment's `stackvm.grade`.

Why level 4 only, from the saved batches of `glm53-lora-dflash2-1105-a1` (4k cap, 4 prompts x 8 per step):

| Level, batches | Samples | Pass | Capped |
| --- | --- | --- | --- |
| L4, first batch | 32 | 43.8% | 3.1% |
| L4, later batches | 80 | 47.5% | 22.5% |
| L5, later batches | 80 | 10.0% | 92.5% |

The first batch is the same four L4 prompts in all three runs of 2026-11-05, so the earlier "truncation jump" was prompt mix.

## Run

Needs the preset's weights and draft on `MILES_WEIGHTS_VOLUME`.

```bash
export MILES_RECIPE=experiments/glm53_stak_overfit/configs/glm53-flash-stak-l4.yaml
modal run -m miles_ext.launch.modal_app --run-id <run-id>
```

Split placement on two half nodes ([ext-068](../../fork/patches/ext-068-split-part-node.md)): B300:4 trainer (TP4 EP4),
B300:4 engine, about $75-78 per 8 GPU-hours. 16 prompts x 8 tries, T 1.0, top-p 0.95, top-k 32, 6k reply cap, LoRA r32,
lr 1e-4, PPO clip 0.2 / 0.28, MIS, staleness 2. `glm53-flash-stak-l4-sc.yaml` is the same run with score centering
instead of PPO clip.

## Success

Mean `rollout/raw_reward` of the last 3 batches at least 0.20 above the first 2, and at least 0.70, with 10 or more of
the 16 prompts improved.

## Results

Not run yet.
