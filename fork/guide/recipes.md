# Recipes: what a run is

A recipe is one YAML file and the only input that changes between runs. The image
and the code stay fixed. `miles_ext/recipes/recipe.py` resolves the recipe against
its defaults and a model preset. It then turns the result into the Miles command
line and the env vars the rollout workers read.

- **Every field, its default and what it does:** `DEFAULTS` in
  `miles_ext/recipes/recipe.py`. Each group has a comment there. The allowed
  values are in `FIELD_RULES`, and an unknown key is an error.
- **Model presets** (Hugging Face repo, parsers, backend, model flags, the shapes it
  has run on): `PRESETS` in `miles_ext/recipes/presets.py`. A recipe names one
  with `model.preset`.
- **Starting points:** `miles_ext/recipes/configs/`. Copy the closest one into
  your own project and edit it there. The header comment of each says what it
  was made for and what it needs. Golden files in `miles_ext/tests/golden/` show
  the exact Miles arguments each one produces.

## The groups at a glance

| Group | Decides |
| --- | --- |
| `model` | preset, backend (`fsdp` or `megatron`), the weights `revision` |
| `cluster` | GPU type, nodes, trainer GPUs, `placement` (`one_cluster` or `split`), timeouts |
| `taskset` | where the tasks and the prompt file are, and the `harness` |
| `agent` | step and time limits for each trial |
| `sandbox` | `modal` or `sail`, lifetime and idle caps |
| `algorithm` | mode (`fully_async` or `sync`), batch shape, learning rate, clipping, advantage estimator, loss, staleness bounds, group filter |
| `rollout` | GPUs per engine, served precision, context length |
| `optimizer`, `parallel`, `lora`, `async`, `checkpoint` | memory and layout knobs, LoRA, fully async buffer, saves |
| `tracking` | W&B project (required), entity, display name, tags |
| `extra_args` | raw Miles flags, appended last (prefer a recipe field when one exists) |

## Inputs

- `taskset.tasks_dir` and `taskset.prompts` may use `$MILES_DATA` and are otherwise
  relative to the recipe file. With `taskset.tasks_volume: <volume>:/<path>`, the
  task dirs come from a Modal volume instead, so your machine keeps only the prompt file.
- In the container, tasks are at `/inputs/tasks`, prompts at `/inputs/prompts.jsonl`,
  weights at `/models/...` and outputs at `/results/<run_id>`.
- Prompt rows for the Harbor harnesses: `{"prompt": ..., "metadata": {"instance_id": "<task dir>"}}`.
  Upstream's [SWE agent example](../../docs/examples/swe-agent-harbor-docker.md)
  describes the same format. Rows for `single-turn` carry `metadata.task` and
  `metadata.label` instead (`miles_ext/tasks/single_turn.py`).

## Overrides at launch

`modal_app` takes `--num-rollout`, `--rollout-batch-size`, `--n-samples`,
`--save-interval` and `--segment-hours` on the command line. Anything else is a
recipe edit.
