# SFT data tokenized like served rollouts

SFT records are text (OpenAI chat messages plus the context of each model call). This
package turns them into training samples with the RL path's TITO tokenizer and the session
server's tree rules, so prompts and sample splits are laid out as in serving. A text reply
becomes the template's canonical form of it, which can differ from what the engine sampled
(for example in whitespace); give `completion_token_ids` to keep sampled tokens exactly.
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
| `config.py` | the SFT run config (YAML): preset, pinned image, records and shards on a volume, schedule; the Miles argv |
| `train.py` | the driver (`python -m miles_ext.sft.train <Miles args>`): Miles' loop for `--debug-train-only`, plus held-out NLL |
| `val.py` | `SftTrainRayActor`: validation ids run forward-only and log `val/nll` |
| `modal_prepare.py`, `modal_train.py` | the config's records to shards on a Modal CPU container, then SFT on one H200:8 node |

## Training on Modal

A run config (YAML, `config.py`) drives both steps. Keys, with defaults where they have one:

```yaml
model: qwen3-30b-a3b-thinking-2507   # a preset (miles_ext/recipes/presets.py) with a megatron_model_type
image: docker.io/starlimobius/miles-mobius@sha256:...   # pinned by digest; or commit: <pushed fork commit>
data:
  volume: miles-rl-datasets
  records: sft/<dataset>/records     # *.jsonl train records, *.val.jsonl held out, _* ignored
  shards: sft/<dataset>/<name>       # modal_prepare writes train/ and val/ here
  max_tokens: 131072                 # longer samples are dropped whole
train: {epochs: 2, batch: 128, lr: 2.0e-5, decay: linear, warmup: 0.03, cp: 2}  # the defaults; decay linear (to 0) or cosine (to lr/10)
val: {interval: 20}                  # held-out NLL every 20 steps (default), before the first and after the last
tracking: {wandb_project: miles-sft} # smoke-/debug- run ids log to miles-smoke
```

```bash
export MILES_SFT_CONFIG=<config.yaml> MILES_WEIGHTS_VOLUME=... MILES_RESULTS_VOLUME=... MILES_DOTENV_DIR=<dir with .env>
python -m modal run -m miles_ext.sft.modal_prepare [--tests]             # records -> shards (CPU)
python -m modal run --detach -m miles_ext.sft.modal_train --run-id <id>  # [--num-rollout N] for a short run
```

- Layout: one H200:8 node, TP1, CP `train.cp`, EP8, full recompute, `max_tokens / cp` packed tokens per GPU.
  Exercised: Qwen3-30B-A3B-Thinking-2507 at 128K with CP4. A model needs a preset with a Megatron definition
  and packed batches (DeepSeek V4.1's one-sample micro-batches are refused for now), and a supported family below.
- Weights: `/models/<preset dir>` on `MILES_WEIGHTS_VOLUME`. One save per epoch.
- Outputs on `MILES_RESULTS_VOLUME` under `<run id>/`: `checkpoints/` (Megatron, weights only),
  `hf/iter_NNNNNNN/` (an HF export per save, with the base's tokenizer and configs), `argv-*.json`.
- The same run id again resumes from the last checkpoint's weights with a fresh optimizer state.
- Code: the image (or one Modal builds from `commit`) plus this checkout's `miles_ext/` and `experiments/`, laid
  over it at launch, so the image digest alone does not identify the code; `argv-*.json` keeps the config and argv.
  The image's `miles/` must have hook-019 (`--custom-trainer-actor-class`); `modal_train` refuses an older one.

## Offline

```bash
python -m miles_ext.sft.prepare --input data.jsonl --output /data/sft-run \
    --hf-checkpoint /weights/Qwen3-30B-A3B-Thinking-2507 --tito-model qwennext \
    --postprocess miles_ext.sft.postprocess.normalize_openai_messages --max-tokens 65536
# then train with: --debug-train-only --rollout-function-path miles_ext.sft.rollout.generate_rollout
#                  --prompt-data /data/sft-run/prompts.jsonl --input-key text
```

Supported families are `SUPPORTED_FAMILIES` in `render.py`: Qwen3-Thinking (`qwennext`) and
DeepSeek V4.1 (`deepseekv41`); `SftRenderer` refuses others. Adding one needs its cases in
`tests/fast/utils/chat_template_utils/test_tito_tokenize_completion.py`,
`miles_ext/tests/test_sft_render.py` and `test_sft_render_hub.py`, and an override of
`TITOTokenizer.tokenize_completion` if its turn close is not part of the assistant render (GLM).

## Connectors

A connector turns another data format into records ([RFC 0003](../../fork/rfcs/0003-sft-record-connectors.md)).
It yields record dicts or `Dropped(source, reason, detail)` items, and
`connectors.base.write_records` validates the records with `parse_record`, writes them as JSONL
and writes a `.report.json` beside it with drops counted by reason.

| File | Role |
| --- | --- |
| `connectors/base.py` | `Dropped`, `write_records`, the report |
| `connectors/atif.py` | Harbor trial directories in ATIF to records, by the spec alone: continuations, `context_management` boundaries, subagents as their own records |

```bash
python -m miles_ext.sft.connectors.atif --input /jobs/my-harbor-job --output /data/atif-records.jsonl \
    --chat-template-kwargs '{"enable_thinking": true}'
# then: python -m miles_ext.sft.prepare --input /data/atif-records.jsonl ...
```

`--model` names the model trained on (default: each trial's `agent.model_name`); steps by other
models stay in the context untrained. `--token-ids` copies ATIF `completion_token_ids`; use it
only when the trajectories were sampled by a model with the trained model's tokenizer.

Not here yet: other connectors (Nemotron-style datasets, Miles rollout dumps), multi-node SFT,
and evaluating an SFT export (serve `hf/iter_*` with an RL recipe's eval).

Tests: `python -m pytest miles_ext/tests/test_sft_record.py miles_ext/tests/test_sft_render.py
miles_ext/tests/test_sft_prepare.py miles_ext/tests/test_sft_connector_atif.py miles_ext/tests/test_sft_config.py` run offline; `test_sft_render_hub.py` needs the Hugging Face
Hub for the real Qwen3 and DeepSeek V4.1 tokenizers.
