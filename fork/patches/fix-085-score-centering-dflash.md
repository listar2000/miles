# fix-085: score centering with DFlash2

## Why
fix-058 allows DSpark only, so score centering on GLM-5.3-Flash could not use any draft: the model has no DSpark
draft, and EAGLE stays refused. Plain decode is about half the speed of DFlash2 here.

## Decision
Allow DFLASH with acceptance thresholds at 1.0. Score centering needs each token to be an exact sample from the
distribution the rollout reports. At SGLang `3a938213c1`, DFlash2's selector accepts with DSpark's `accept_sampling`
(`dflash_worker_v2.py`, `_selector_sampling_accept`), and top-k and selected logprobs come from `compute_spec_logprobs`
as they do for DSpark, whose check is fix-058. Lower thresholds accept tokens by another rule, so they are refused.
Limit: Miles sees the algorithm, not the draft. A first-generation DFlash draft has no selector and was not checked;
no preset uses one.

## Evidence
[evidence/fix-085](evidence/fix-085/report.json), from [probe.py](evidence/fix-085/probe.py) on one B300:8: full
GLM-5.3-Flash, a DFlash2 TP4 engine beside a plain TP4 engine, both with LoRA on (no adapter loaded) and the FP32 LM
head, 64 samples x 512 tokens per engine, re-scored by prefill on the plain engine.

| Check | DFlash2 | Plain decode |
| --- | --- | --- |
| Calibration, mean p(sampled) - sum p^2 over the reported top-64 (0 when exact) | +0.0011, z +1.4 | -0.0012, z -1.4 |
| Calibration over the top-p 0.95 / top-k 64 support | -0.0012, z -1.6 | +0.0004, z +0.6 |
| Sampled token outside the support; largest abs log of the support's sum | 0; 2e-7 | 0; 2e-7 |
| All-zero R3 rows | 0 | 0 |
| Selected-token mean abs(logprob - prefill) | 0.086 | 0.100 |
| Tokens per verify | 3.9 | - |

No adapter was loaded in that check. End to end, with score centering, MIS, top-p replay, LoRA rank 32, lr 1e-4 and zero
staleness on one B300:8 (`glm53-flash-lora-dflash2.yaml`): 5 rollouts ran clean, the train/rollout log-prob gap went
0.026, 0.028, 0.035, 0.038, 0.039 as the largest adapter weight grew to 5e-4, 0 of 113,434 R3 rows were all zero on the
last rollout, and accept length held at about 3.9.

## How it is applied
`miles/utils/score_centering.py` and its test, marked `[fork fix-085]`. Verify:
`pytest tests/fast/utils/test_score_centering_dspark.py`.

## Remove when
Upstream's `validate_score_centering_args` accepts DFLASH; upstream status: not filed.
