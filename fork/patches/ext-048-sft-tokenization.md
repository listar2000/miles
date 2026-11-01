# ext-048: SFT records tokenized like session-server rollouts

## Why
The stock SFT rollout (`miles/rollout/sft_rollout.py`, `MultiTurnLossMaskGenerator`) renders
each message alone with the native template, for three hard-coded Qwen types. That is not
the token layout the session server trains on in RL, and a conversation log alone cannot say
what context each model call had. The Qwen3 SWE replication (2026-10-30) found two silent
mismatches this way (missing tool schema, `\n\n<tool_call>` re-rendered as `\n<tool_call>`).
Design: [RFC 0002](../rfcs/0002-tito-aware-sft-tokenization.md).

## Decision
- Records are OpenAI messages used as a store, plus `steps` (each call's context as indices,
  and its reply). Without `steps`, the harness is taken as append-only.
- Steps are tokenized by calling session server v2's own code (`SessionTree.find_attach_point`,
  `session_state._render_token_ids`, `default_postprocess` for reply ownership), not a copy, so
  serving and SFT cannot drift apart. Tests replay the same steps through
  `prepare_token_ids_and_request_args` and `commit_generation` and compare tokens.
- `thinking_retention: all | template` declares how the model will be served. prime-rl's
  `tool_cycle` is what `template` does with a Qwen3/DeepSeek-style template, so it is not a
  third value.
- Rejected: extending `MultiTurnLossMaskGenerator` (per-message renders cannot follow
  `merge_tokens`), and hand-written per-family renderers as in prime-rl (a second owner for
  the template rules TITO already has).

## How it is applied
New files: `miles_ext/sft/` (record, postprocess, render, prepare, rollout, parity, README) and
`miles_ext/tests/test_sft_*.py`. It uses hook-017 (`TITOTokenizer.tokenize_completion`). Run
`python -m pytest miles_ext/tests/test_sft_record.py miles_ext/tests/test_sft_render.py
miles_ext/tests/test_sft_prepare.py`. On a sync, those tests catch API drift in the session
server v2 functions above.

## Remove when
Upstream Miles tokenizes SFT data through the session server's rules.
