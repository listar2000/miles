# fix-016: score-centering candidates as flat base64 arrays

## Why
Score centering (`--loss-type score_centering`) asks every training chat call for `top_logprobs=k` (k=32 in our
recipes). SGLang's chat endpoint then runs `return_text_in_logprobs=True`, so for every generated token the engine's
HTTP process (the tokenizer manager, one Python process per engine) decodes k candidate strings, builds k
`TopLogprob` objects with their bytes, and serializes all of them twice (`choices[].logprobs` and
`meta_info.output_top_logprobs`). Evidence: one-node probe replaying 128 recorded score-centering sessions
(2026-10-19): fixed per-call overhead 20.4 s with top-k against 3.2 s without, calls in 16 min 4,940 -> 3,154, the
tokenizer manager at ~98% of a core while the GPUs held 10-18 requests and an empty queue. production runs
saw ~35-45 s per call. CPU replay of that code path (sglang-miles 7b6993a, k=32): 1.0 s and 8 MB per
2,000-token response, 4.3 s and 33 MB per 8,000 tokens, against 0.02 s and 0.06 s with no top-k.

## Decision
Request SGLang's own compact format, `return_flat_raw_output_top_logprobs` with
`return_flat_raw_top_logprobs_b64` (sgl-project/sglang#42890, on sglang-miles and in our image's engine): the
candidates come back once, as base64 int32 ids and float32 logprobs in `meta_info`, built from the arrays the
scheduler already has, with no per-candidate decoding or objects. The same CPU replay costs 0.02 s and 1 MB per
2,000 tokens, the same as no top-k. Values are the float32 the sampler computed, which is what Miles stores.
`choices[].logprobs.top_logprobs` is then empty; the session server never returns it to agents unless they ask
for logprobs, and the rollout args already override the agent's own top-k.
If the response comes back without the flat arrays (an older router dropping the flag, or SGLang's fallback),
sample collection raises instead of reading nested rows: the nested path is what made those runs unusable.
Alternatives that lost: decoding candidates in a worker pool or with `skip_tokenizer_init` (still builds and
serializes k objects per token); computing the score-centering statistics in the engine (an SGLang fork change,
and the trainer needs the candidates themselves).

## How it is applied
`configure_rollout_topk_logprobs_request` sets both flags on OpenAI (session server) training requests in
"selected" mode; `append_rollout_topk_logprobs(..., flat=True)` decodes them, called by the session sample
producer (`samples/merge.py`) when the backend request asked for them. The native `/generate` path is unchanged
(it never decodes candidate text). The router must forward the flags: radixark/sgl-router-for-miles#24
(2026-10-14). Verify: `tests/fast/rollout/session/test_score_centering.py` (flat and nested responses give the
same validated sample), `tests/fast/rollout/generate_utils/test_score_centering_contracts.py`.
On a sync, check upstream has not started sending these flags itself.

## Remove when
Upstream Miles requests the flat format for rollout top-k logprobs.

**Removed in sync-v0.1.3:** upstream v0.1.3 requests the flat format itself (radixark/miles#3952); the sync took
upstream's side in every file this fix touched and its commit carries `Retires: fix-016`.
