# fix-058: score centering with DSpark

## Why

Upstream `validate_score_centering_args` refuses score centering (SC) with every
`--sglang-speculative-algorithm`, because "speculative candidate-logprob semantics
are not verified". SC runs therefore lose DSpark: a DeepSeek V4.1 rollout batch took
54 min without it versus about 30 min with it.

## Decision

Allow DSPARK only. SC needs each generated token to be an exact sample from the
distribution the rollout reports, and the reported top-k to come from the same
logits. In SGLang `sglang-miles` `3a938213c1` (v0.1.3's pin):

- DSpark accepts drafts by rejection sampling against the target's temperature and
  top-p probabilities (`dspark_accept.py`), so every emitted token, the bonus token
  included, is an exact target sample.
- `compute_spec_logprobs` (`layers/logprob_processor.py`) takes the selected logprob
  and the top-k from one `log_softmax(verify_logits / T)` per committed token, as
  plain decoding does; verify runs through the normal `LogitsProcessor`, so SC's
  FP32 LM head applies.
- Top-p sampling masks under DSpark come from SGLang #34201, in the pin.

Other modes (EAGLE trees, NGRAM, DFLASH) stay refused until checked the same way.
`SGLANG_SIMULATE_ACC_LEN > 0` is refused with DSpark: it forces acceptance, which
is not exact sampling.

## Evidence

GPU comparison on one B300:8 (DeepSeek V4.1-Flash-1017, DSpark TP4 engine beside a
plain TP4 engine, both with the FP32 LM head; 64 samples x 512 tokens per engine,
re-scored by prefill on the plain engine), PR #96:

| Check | DSpark | Plain decode |
| --- | --- | --- |
| Calibration, mean p(sampled) - sum p^2 over the reported top-64 (0 when exact) | -0.0012, z -1.5 | +0.0004, z +0.5 |
| Calibration over the top-p 0.95 / top-k 64 support | -0.0007, z -1.0 | -0.0014, z -2.0 |
| Selected-token mean abs(logprob - prefill) | 0.047 | 0.039 |
| Top-64 overlap with prefill | 93.0% | 93.9% |
| Accept length | 3.7 | - |

DSpark samples exactly from what it reports; the reported distribution sits about
0.008 nats further from prefill than plain decode's (verify-pass numerics), against
a rollout-vs-trainer gap of 0.031-0.036 on the same model without DSpark, which
SC's importance weights already correct. The reported measurements and coverage
limits are recorded in [PR #96](https://github.com/listar2000/miles/pull/96).

## How it is applied

`validate_score_centering_args` in `miles/utils/score_centering.py` skips the
speculative refusal for DSPARK (any case). Tests:
`tests/fast/utils/test_score_centering_dspark.py`.

On an SGLang bump that touches `speculative/dspark*` or `logprob_processor.py`,
re-run the calibration and logprob comparison before keeping this exception.

## Remove when

Upstream Miles allows speculative decoding with score centering. Upstream status:
not filed.
