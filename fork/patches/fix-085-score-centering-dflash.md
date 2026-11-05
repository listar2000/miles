# fix-085: score centering with DFlash2

## Why
fix-058 allows DSpark only, so score centering on GLM-5.3-Flash could not use any draft: the model has no DSpark
draft, and EAGLE stays refused. In an inference-only probe (one 180 s run per arm, no adapter, no top-p, no score
centering) plain decode gave about half DFlash2's token rate.

## Decision
Allow DFLASH with acceptance thresholds at 1.0. Score centering needs each token to be an exact sample from the
distribution the rollout reports. At SGLang `3a938213c1`, DFlash2's selector accepts with DSpark's `accept_sampling`
(`dflash_worker_v2.py`, `_selector_sampling_accept`), and top-k and selected logprobs come from `compute_spec_logprobs`
as they do for DSpark, whose check is fix-058. Lower thresholds accept tokens by another rule, so they are refused. A first-generation DFlash draft has no selector
and takes another accept path, so Miles reads the draft's `config.json` and refuses anything but `DFlash2DraftModel`.

## Evidence
[report.json](evidence/fix-085/report.json), from [the probe](https://github.com/listar2000/miles-ops/blob/main/notes/glm53/dflash2_probe.py) (private ops repo) on one B300:8: full
GLM-5.3-Flash, a DFlash2 TP4 engine beside a plain TP4 engine, both with LoRA on (no adapter loaded) and the FP32 LM
head, 64 samples x 512 tokens per engine, re-scored by prefill on the plain engine.

| Check | DFlash2 | Plain decode |
| --- | --- | --- |
| Calibration, mean p(sampled) - sum p^2 over the reported top-64 (0 when exact, plus the omitted tail's sum p^2) | +0.0011, z +1.4 | -0.0012, z -1.4 |
| Calibration over the top-p 0.95 / top-k 64 support | -0.0012, z -1.6 | +0.0004, z +0.6 |
| Sampled token outside the support; largest abs log of the support's sum | 0; 2e-7 | 0; 2e-7 |
| All-zero R3 rows | 0 | 0 |
| Selected-token mean abs(logprob - prefill) | 0.086 | 0.100 |
| Tokens per verify | 3.9 | - |

These are consistency checks: they would catch a sampler bias of this size, and they do not prove exactness. The top-64
statistic has a small positive offset from the tail it leaves out, which only pulls DFlash2's +1.4 down.

No adapter was loaded in that check. End to end, with score centering, MIS, top-p replay, LoRA rank 32, lr 1e-4 and zero staleness on one B300:8: see
[end-to-end.md](evidence/fix-085/end-to-end.md).

## How it is applied
`miles/utils/score_centering.py` and its test, marked `[fork fix-085]`. Rerun the probe with `python -m modal run dflash2_probe.py --run-id <id>`, with the fork commit set in the script. Verify:
`pytest tests/fast/utils/test_score_centering_dspark.py`.

## Remove when
Upstream's `validate_score_centering_args` accepts DFLASH; upstream status: not filed.
