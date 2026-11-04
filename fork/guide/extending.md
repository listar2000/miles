# Extending: algorithms, harnesses, rewards and data

Most training work needs no code: a recipe picks the model, the tasks, the
harness and the algorithm knobs ([recipes.md](recipes.md)). This page covers the
cases that do, and where the code goes.

## Where your code runs

The trainer, the rollout workers and the agent functions run inside the fork's
image. Code reaches the image in one way only: as a commit of this repo
(`MILES_FORK_COMMIT`, which fetches the commit from `stack.FORK_REPO`). So:

- **Data and task sets** stay in your own project. They are inputs, not code.
- **New code** lives under `miles_ext/` on a branch of this repo: a draft PR, or
  a branch you launch from with `MILES_FORK_COMMIT`. Follow the Ext rules in
  [fork/README.md](../README.md) ("Kinds of change"). They come down to new
  files only, plugged in through extension points. The
  [miles-fork skill](../../.claude/skills/miles-fork/SKILL.md) has the steps.
- Loading a package from outside this repo is not supported yet. If you need it,
  open an issue ([help.md](help.md)).

## By kind

| You want | Use | Code lives in |
| --- | --- | --- |
| A different algorithm setting (estimator, clipping, loss, KL, staleness, filters) | recipe `algorithm.*` fields | nothing new |
| A new loss | Miles' `--loss-type custom_loss` with `--custom-loss-function-path`. `algorithm.loss` allows only `policy_loss` and `score_centering` today, so this also needs a new recipe field (an Ext change to `recipe.py`) | `miles_ext/<area>/` |
| A new agent harness | a new `taskset.harness` value in `HARNESSES` (`miles_ext/recipes/recipe.py`) and the agent function it maps to (today: mini-swe-agent in the sandbox or on the host, Terminus-2 and DeepSeek's dsh in the sandbox, and `single-turn`) | a new `miles_ext/harnesses/<harness>/` (its Harbor agent class, pins and assets; `harnesses/common.py` has the in-sandbox setup) |
| A new reward for agent tasks | the task's own Harbor verifier (its tests). `miles_ext/harbor/reward.py` just passes Harbor's verdict on | the task set (yours) |
| A new single-turn task type | a verifier in `VERIFIERS` (`miles_ext/tasks/single_turn.py`) | `miles_ext/tasks/` |
| A new task set | Harbor task dirs plus a prompt file | your project; `taskset.tasks_volume` to keep it on a Modal volume |
| A new sandbox provider | Harbor environment through `import_path` | `miles_ext/sandboxes/`; see `docs/developer/adding-a-sandbox-provider.md` |
| A new model | a preset in `miles_ext/recipes/presets.py` | maintainers: open an issue first |
| SFT data from another format (Harbor ATIF, ...) | a connector that yields records ([RFC 0003](../rfcs/0003-sft-record-connectors.md)); ATIF: `python -m miles_ext.sft.connectors.atif` | `miles_ext/sft/connectors/` ([README](../../miles_ext/sft/README.md)) |
| SFT training | an SFT run config: `python -m modal run -m miles_ext.sft.modal_prepare`, then `-m miles_ext.sft.modal_train` | your config; see [miles_ext/sft/README.md](../../miles_ext/sft/README.md) |
| SFT data, or SFT support for another model family | records as in [RFC 0002](../rfcs/0002-tito-aware-sft-tokenization.md), prepared by `python -m miles_ext.sft.prepare`; a family needs the tests in `miles_ext/tests/test_sft_render.py` (and maybe a `tokenize_completion` override) | your records; `miles_ext/sft/` ([README](../../miles_ext/sft/README.md)) |

Miles' own extension points and their signatures are documented upstream in
[docs/user-guide/customization.md](../../docs/user-guide/customization.md). The
flags themselves are defined in `miles/utils/arguments.py`.

## Before you send a PR

```bash
python -m pytest miles_ext/tests -q --continue-on-collection-errors   # CPU only
python fork/tools/ledger.py check         # fork conventions; needs a full clone
```

On a machine without the image's packages (torch, Harbor, mini-swe-agent), some
test modules fail to import. Compare the result with `main`: it should be the same
or better. GitHub Actions do not run on this repo, so run these yourself. Then do a
`--check-only` dry run with your commit (`MILES_FORK_COMMIT`). Every PR goes
through the [miles-review checklist](../../.claude/skills/miles-review/SKILL.md).

## Not yours to change

Edits to upstream Miles files, SGLang patches, kernels, the image build and the
launcher's internals belong to the maintainers. Open an issue saying what you
tried to do and where it broke.
