# hook-017: `TITOTokenizer.tokenize_completion`, an assistant message as the tokens the engine samples

## Why
SFT data is text, but the session server trains on the tokens the engine sampled. To
lay out SFT samples like served ones ([RFC 0002](../rfcs/0002-tito-aware-sft-tokenization.md)),
each assistant message must become completion tokens: reasoning, content, tool calls
and the stop token, without the generation prompt before it or template text after it.
From text this is the template's canonical form, which can differ from what the engine
sampled; recorded `completion_token_ids` are used instead when a record has them. `TITOTokenizer` owns every other per-family
token rule (fixed template, `merge_tokens` boundaries), but it had no method for this
direction. The Qwen3 SWE SFT replication (2026-10-30) hard-coded it for `<|im_end|>`.

## Decision
- One method on the base class, so every TITO family has it and a family overrides
  it next to its other boundary rules. Default: render a probe prompt with and without
  the message, encode the text after the generation prompt, and drop one trailing
  token in `trailing_token_ids` (Qwen's `\n` after `<|im_end|>`).
- Rejected: a per-family table in `miles_ext/sft/` (a second owner for family token
  rules), and hand-written renderers per family as in prime-rl's `renderers` package
  (duplicates the templates TITO already uses).
- Families are SFT-supported once tests compare their output with a whole-trajectory
  render: Qwen3 (`qwennext`) and DeepSeek V4.1 (`deepseekv41`). The method's own tests are
  `tests/fast/utils/chat_template_utils/test_tito_tokenize_completion.py` (real
  Qwen3-30B-A3B-Thinking-2507 and DeepSeek-V4.1-Flash-1017 tokenizers); the SFT assembly
  tests are in `miles_ext/tests/test_sft_render*.py`. GLM closes a turn with the next role's
  opener and needs an override before it is used.

## How it is applied
`miles/utils/chat_template_utils/tito_tokenizer.py`: one method inserted after
`TITOTokenizer.merge_tokens`, between `[fork hook-017]` markers, and its test file
`tests/fast/utils/chat_template_utils/test_tito_tokenize_completion.py`. Caller:
`miles_ext/sft/render.py`. On a sync, keep it on the base class; if upstream adds an
equivalent, switch the caller and retire this hook.

## Remove when
Upstream `TITOTokenizer` can turn an assistant message into its sampled tokens.
Upstream status: not filed.
