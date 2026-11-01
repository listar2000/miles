# SFT data tokenized like served rollouts

SFT records are text (OpenAI chat messages plus the context of each model call). This
package turns them into training samples whose tokens and loss masks are what the Miles
session server would have recorded for the same calls, with the RL path's TITO tokenizer.
The design and its reasons are in [RFC 0002](../../fork/rfcs/0002-tito-aware-sft-tokenization.md);
the record format is in its "The record" section and in `record.py`.

| File | Role |
| --- | --- |
| `record.py` | the record format: `parse_record` validates a JSON record into an `SftRecord` of messages and steps |
| `postprocess.py` | `fn(record) -> record` processors by dotted path, and the built-in ones |
| `render.py` | `SftRenderer`: steps to samples with the session server v2 attach rule; `thinking_retention` `all` or `template`; `load_renderer` |
| `prepare.py` | the offline CLI: records to token shards, `prompts.jsonl`, `samples.jsonl`, `manifest.json` |
| `rollout.py` | `--rollout-function-path miles_ext.sft.rollout.generate_rollout`: prepared samples for `--debug-train-only` training |
| `parity.py` | `compare`: a served sample against the render of the same record |

```bash
python -m miles_ext.sft.prepare --input data.jsonl --output /data/sft-run \
    --hf-checkpoint /weights/Qwen3-30B-A3B-Thinking-2507 --tito-model qwennext \
    --postprocess miles_ext.sft.postprocess.normalize_openai_messages --max-tokens 65536
# then train with: --debug-train-only --rollout-function-path miles_ext.sft.rollout.generate_rollout
#                  --prompt-data /data/sft-run/prompts.jsonl --input-key text
```

Supported families are those whose tests in `miles_ext/tests/test_sft_render.py` and
`test_sft_render_hub.py` pass: Qwen3-Thinking (`qwennext`) and DeepSeek V4.1 (`deepseekv41`). Another TITO
family needs those tests, and an override of `TITOTokenizer.tokenize_completion` if its
turn close is not part of the assistant render (GLM).

Not here yet (a later RFC): adapters that build records from Harbor ATIF or from rollouts,
and launching SFT through the fork's recipes on Modal.

Tests: `python -m pytest miles_ext/tests/test_sft_record.py miles_ext/tests/test_sft_render.py
miles_ext/tests/test_sft_prepare.py` run offline; `test_sft_render_hub.py` needs the Hugging Face
Hub for the real Qwen3 and DeepSeek V4.1 tokenizers.
