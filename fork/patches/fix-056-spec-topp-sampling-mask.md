# fix-056: speculative decoding returns top-p sampling masks (backport of SGLang #34201)

## Why
Top-p (or top-k) rollout sampling turns on Miles' sampling-support replay: every training request asks SGLang
for `return_sampling_mask`, the set of tokens the sampler could have picked and their renormalized log-probs.
Our SGLang (`sglang-miles` `7b6993a`) rejects that request whenever speculative decoding is on
("return_sampling_mask is not supported with speculative decoding", `scheduler.py`), so top-p and DSpark
cannot run together. SGLang main fixed this in [#34201](https://github.com/sgl-project/sglang/pull/34201)
(merged 2026-09-30 as `201c4b9b2a`); `sglang-miles` branched from main on 2026-09-29 (`bd78095030`) and
does not have it.

## Decision
- Copy #34201's runtime changes verbatim: the new `speculative/spec_sampling_mask.py` (validation plus a capture
  that rebuilds the target policy after acceptance), the DSpark and DFlash worker calls, the per-token
  alignment in `batch_result_processor.py`, and the per-row mask `top_k` in `sampling_batch_info.py`. Its test
  files are left out. Acceptance and sampling are untouched; batches without a mask request do no extra work.
- Source: SGLang main commit `201c4b9b2a` (#34201), copied verbatim (397 added lines). The user approved
  carrying it on 2026-10-21.

## How it is applied
`fork/deps/sglang/fix-056-spec-topp-sampling-mask.patch`, applied after fix-053 by `fork/deps/sglang/apply.py`
to SGLang `7b6993a` (offsets only, no fuzz), one `[fork fix-056]` comment per edited file.

## Remove when
`stack.SGLANG_COMMIT` moves to an `sglang-miles` commit that contains `201c4b9b2a`. Upstream status, checked
2026-10-21: merged on SGLang main; not on `sglang-miles`.
