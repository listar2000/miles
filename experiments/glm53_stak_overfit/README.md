# STAK overfit on GLM-5.3-Flash LoRA

RL on 12 fixed STAK questions, to check that the training loop raises reward on GLM. STAK is a made-up stack language
([dsv41_stak_overfit](../dsv41_stak_overfit/README.md), which overfit DeepSeek with thinking off). GLM can't turn thinking
off, so this runs at `reasoning_effort: low` with a 6k reply cap on an easier prompt set.

**Status: the loop runs and the policy changes, but no run here reached the success bar.** The clearest signal is one
prompt going from 0/16 to 15/16 in the first run.

## Data

[prompts/stak_l4_12.jsonl](prompts/stak_l4_12.jsonl) is 12 of the 16 level-4 rows of
`dsv41_stak_overfit/prompts/stak32.jsonl`, whose `stackvm.grade` is the grader. `metadata.base_pass8` is DeepSeek's base
rate, not GLM's. Level 5 is left out because it mostly hits the cap, measured at a 4k cap on 2026-11-05 (4 prompts x 8 per
step, score centering, one cluster):

| Level, batches | Samples | Pass | Capped |
| --- | --- | --- | --- |
| Level 4, first batch | 32 | 43.8% | 3.1% |
| Level 4, later batches | 80 | 47.5% | 22.5% |
| Level 5, later batches | 80 | 10.0% | 92.5% |

The first batch was the same four level-4 prompts in all three runs of that day, so the earlier "truncation jump" was
prompt mix. L4-001, 004, 009 and 018 were then dropped, because they capped early in the first run below. They were chosen
after seeing that data, so the 12-prompt numbers are flattering.

## Run

Needs the preset's weights and draft on `MILES_WEIGHTS_VOLUME`.

```bash
export MILES_RECIPE=experiments/glm53_stak_overfit/configs/glm53-flash-stak-l4-12-sc.yaml
modal run -m miles_ext.launch.modal_app --run-id <run-id>
```

[configs/glm53-flash-stak-l4-12-sc.yaml](configs/glm53-flash-stak-l4-12-sc.yaml) is the latest run's recipe: split placement
on two half nodes ([ext-068](../../fork/patches/ext-068-split-part-node.md)), B300:4 trainer (TP4 EP4) + B300:4 engine,
12 prompts x 8 tries, T 1.0, top-p 0.95, top-k 32, DFlash2, LoRA r32, score centering + MIS on group-centered advantages
with no std division, lr 5e-5, staleness 2, 32 samples in flight, 10 steps, about $75.

## Success

Mean `rollout/raw_reward` of the last 3 batches at least 0.20 above the first 2, and at least 0.70, with most prompts
improved. A batch of 128 samples has about 0.044 of binomial noise.

## Results

All 2026-11-06, W&B `mobiuslab/miles-glm53`. The earlier two runs used 16 prompts (the 12 plus the four dropped), batch
16 x 8, lr 1e-4 and 128 samples in flight.

**`policy_loss` + MIS, 16 prompts**, stopped on request after step 8, about $100:

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reward | 0.38 | 0.53 | 0.41 | 0.57 | 0.65 | 0.63 | 0.62 | 0.63 | 0.63 |
| Capped at 6k | 9% | 14% | 17% | 15% | 13% | 24% | 21% | 27% | 33% |

Short of the bar (first two 0.46, last three 0.63). Over the first and last two batches, L4-019 went 0/16 to 15/16 and
L4-014 0/16 to 8/16, while L4-009 fell 5/16 to 0/16 with 13 of 16 replies capped. Over the same batches, wrong answers fell
from 114 to 33 of 256 and capped failures rose from 25 to 62.

`policy_loss` here is effectively GRPO: its PPO clip 0.2 / 0.28 never binds (`train/pg_clipfrac` 0.0 on every step). Without
`--use-rollout-logprobs` the baseline is the trainer's own fresh forward, so the ratio is 1 on the single update per rollout.

**Score centering + MIS, 16 prompts**: the engine crashed in batch 7 with a CUDA illegal address (first reported at
`dflash_worker_v2.py:2602`, cause unknown), about $72, no checkpoint:

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reward | 0.52 | 0.45 | 0.40 | 0.60 | 0.68 | 0.63 | 0.59 |
| Capped at 6k | 8% | 14% | 13% | 20% | 23% | 42% | 42% |

Batch 0 is the same zero-adapter policy in both runs (0.38 and 0.52), so a batch moves by more than its binomial noise.

**Score centering + MIS, 12 prompts, lr 5e-5, 32 in flight** (this recipe), 10 steps, no crash, about $75:

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reward | 0.60 | 0.60 | 0.70 | 0.48 | 0.71 | 0.61 | 0.68 | 0.65 | 0.69 | 0.73 |
| Capped at 6k | 2% | 2% | 4% | 4% | 9% | 5% | 15% | 13% | 11% | 17% |

Pooled pass over the first and last 3 batches: 0.635 to 0.688, with 6 prompts up, 5 down and 1 level, small against the
noise. The crash did not repeat, at a quarter of the earlier in-flight load, so that doesn't rule it out.

**Log-prob gap and sample age.** `train/train_rollout_logprob_abs_diff` compares current trainer log-probs with those saved
at generation, so it includes staleness. At 128 in flight every sample was 2 versions old from batch 2 on, and the gap
stayed near 0.05 with `policy_loss` and 0.08 with score centering. At 32 in flight and half the lr, samples were 1.4-1.5
versions old and the gap held at 0.037; these runs don't separate those two changes.
