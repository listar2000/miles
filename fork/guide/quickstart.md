# Quickstart: a first run from your own machine

From a fresh dev machine with your own Modal account to a training run. Everything
runs on Modal; your machine only launches runs and holds credentials. Read
[running.md](running.md) before a long run.

## 1. Machine and accounts

- Python 3.10+, git, and a full clone of this repo (`main`, or a `stable-*` tag).
- `pip install modal pyyaml huggingface_hub`. Nothing else is needed locally: the
  trainer runs in the fork's image, which already holds Miles, SGLang, Harbor and
  mini-swe-agent.
- `modal token new` (a Modal account with GPU access; DeepSeek V4.1 recipes need B300s).
- A W&B account. A Hugging Face token if the model repo is gated.
- A Sail account only if a recipe sets `sandbox.provider: sail` (the default is Modal sandboxes).

## 2. Credentials

The launcher turns a `.env` file into a Modal Secret at launch time
(`MILES_DOTENV_DIR`, default: the current directory). Keep it outside the repo.

```
WANDB_API_KEY=...
HF_TOKEN=...          # gated model repos only
SAIL_API_KEY=...      # sandbox.provider: sail only
```

The keys a run needs are in `miles_ext/platform/secrets.py` (`required_keys`).

## 3. Your Modal names

The launcher's Modal names default to this repo's maintainers' names (`miles-fork`
app, `miles-fork-results` and `miles-swe-models` volumes; `miles_ext/platform/target.py`).
In a Modal workspace you share with anyone, pick your own so runs, cleanup and
volumes never collide. The launcher refuses names outside
`OWN_APP_NAMESPACES` in `miles_ext/recipes/recipe.py` (for example `miles-`), so
use that prefix for all of them:

```bash
export MILES_APP_NAME=miles-<you>            # the GPU app
export MILES_RESULTS_VOLUME=miles-<you>-results
export MILES_WEIGHTS_VOLUME=miles-<you>-weights
```

Volumes are created on first use. The full list of names a run creates is in
the `miles_ext/launch/modal_app.py` docstring.

## 4. Pick a trainer version

A trainer version is `latest` (the newest build of `main`) or a `stable-YYYY-MM-DD`
from `miles_ext/stable.json`. Resolve it to an image and a commit:

```bash
python -c "from miles_ext.platform.versions import resolve; print(resolve('latest'))"
git checkout <commit>                # the recipe code runs locally, so it must match the image
export MILES_IMAGE=<image>           # repo@sha256:...
```

Always set `MILES_IMAGE` or `MILES_FORK_COMMIT`. Without either, the launcher uses
`stack.FORK_IMAGE`, an older release kept for existing callers.
`MILES_FORK_COMMIT=$(git rev-parse HEAD)` builds an image on Modal from that
pushed commit instead. It is slow the first time, so use it only when you changed the fork.

## 5. Weights and tasks

- **Weights.** Set `model.revision` (a commit sha of the preset's Hugging Face repo) in
  the recipe, then put that snapshot on your weights volume once:

  ```bash
  python -c "from miles_ext.platform import prepare_weights; \
    print(prepare_weights('Qwen/Qwen3.5-9B', '<sha>', {'weights_volume': '$MILES_WEIGHTS_VOLUME'}))"
  ```

  DeepSeek V4.1 Flash also needs the trainer's BF16 copy on that volume
  (`train_dir_name` in `miles_ext/recipes/presets.py`). The repo has no
  one-command way to make it yet ([limits.md](limits.md)).
- **Tasks.** A task set is a directory of Harbor task dirs plus a `prompts.jsonl` whose
  rows carry `prompt` and `metadata.instance_id` (the task dir's name). Point the
  recipe at them with `$MILES_DATA` ([recipes.md](recipes.md)). Task sets live
  outside this repo.

## 6. Dry check, then train

Copy the closest shipped recipe into your own project first. In the copy, set
`tracking.wandb_entity` to your W&B entity (or delete the line to use your API
key's default) and `tracking.wandb_project` to your own project. The shipped
recipes log to the maintainers' entity, which your key cannot write to.

To watch the run in the monitor, also start it as step 7 describes. `MILES_MONITOR_DIR` below
makes the launch register itself there:

```bash
export MILES_RECIPE=<your copy>.yaml MILES_DATA=<your task sets> MILES_MONITOR_DIR=~/miles-monitor
modal run -m miles_ext.launch.modal_app --run-id smoke-<you>-1 --check-only    # one GPU, minutes
modal run --detach -m miles_ext.launch.modal_app --run-id smoke-<you>-1 > smoke-<you>-1.log 2>&1   # train
```

Redirecting the launch output to a file lets the monitor tail it for startup
stages, per-step rewards and errors. The launcher finds that file through `/proc`,
so on macOS register it by hand after launching:
`python -m miles_ext.monitor.register smoke-<you>-1 smoke-<you>-1.log <your copy>.yaml`.

The dry check builds the run's arguments in the image, parses them with Miles,
and checks the tasks and weights, all on one GPU. Fix whatever it reports before
you spend a cluster. Run ids starting with `smoke-` or `debug-` log to the
W&B project `miles-smoke`. Launching the same run id again resumes from its
last save.

Each shipped recipe's `cluster` block gives its hardware, from one H200:8 node
for the Qwen3.5-9B recipes to several B300:8 nodes for DeepSeek V4.1 Flash.
Its header comment says what it was made for and what it needs.

## 7. Watch the run

The run monitor is a read-only web page over your runs, served from your machine:

```bash
pip install wandb                                # Python 3.11+ (the config reader uses tomllib)
export MILES_MONITOR_DIR=~/miles-monitor         # the same dir your launches use
miles_ext/monitor/start.sh                       # http://127.0.0.1:8787
miles_ext/monitor/start.sh --tunnel              # a public URL behind a password (needs cloudflared)
```

Each launch with `MILES_MONITOR_DIR` set shows up within seconds. Config, run
groups, the compare page and everything else are in
[miles_ext/monitor/README.md](../../miles_ext/monitor/README.md).
