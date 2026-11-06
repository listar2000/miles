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
lr 1e-4, `policy_loss` with MIS, staleness 2. `glm53-flash-stak-l4-sc.yaml` is the same run with score centering.

`policy_loss` here is effectively GRPO: group-centered advantages, no std division, per-token loss. Its PPO clip 0.2 / 0.28
never binds (`train/pg_clipfrac` 0.0 on every step): without `--use-rollout-logprobs` the baseline is the trainer's own
fresh forward, so the ratio is 1 on the single update per rollout.

## Success

Mean `rollout/raw_reward` of the last 3 batches at least 0.20 above the first 2, and at least 0.70, with 10 or more of
the 16 prompts improved.

## Results

`glm53ov-1106-a1` (2026-11-06, `policy_loss` + MIS, image from 0824d2ccf5, W&B `mobiuslab/miles-glm53`), stopped on request
after step 8, about $100:

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reward | 0.38 | 0.53 | 0.41 | 0.57 | 0.65 | 0.63 | 0.62 | 0.63 | 0.63 |
| Capped at 6k | 9% | 14% | 17% | 15% | 13% | 24% | 21% | 27% | 33% |

Short of the success bar (first two 0.46, last three 0.63). Pass rates rose most where replies stay short: L4-019 0/16 to
15/16, L4-014 0/16 to 8/16. L4-009 fell 5/16 to 0/16 as 13 of 16 replies hit the cap. Log-prob gap 0.028 to 0.055; volume
sync 13-35 s for a 252 MB adapter.

`glm53ov-sc-1106-a1` (score centering, otherwise the same): the engine crashed in batch 7 with a CUDA illegal address
(`dflash_worker_v2.py:2602`, cause unknown), about $72, no checkpoint saved:

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reward | 0.52 | 0.45 | 0.40 | 0.60 | 0.68 | 0.63 | 0.59 |
| Capped at 6k | 8% | 14% | 13% | 20% | 23% | 42% | 42% |

Both runs level off near 0.63 as capped replies grow. Batch 0 is the same zero-adapter policy in both (0.38 and 0.52), so
one batch moves by more than its binomial noise of 0.044. The log-prob gap reached 0.083, against 0.055 with `policy_loss`.
