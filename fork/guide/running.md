# Running, monitoring and debugging a run

What a run looks like from outside, after `modal run --detach -m miles_ext.launch.modal_app --run-id <id>`
([quickstart.md](quickstart.md)).

## Where things are

| What | Where |
| --- | --- |
| The GPU work | Modal app `MILES_APP_NAME` (default `miles-fork`). `modal app list`, then `modal app logs <app id>` |
| Task sandboxes | Modal app (or Sail app) `<sandbox.app_prefix><run_id>`, one per run, cleaned up when the run exits |
| Outputs | results volume, `/<run_id>/`: `checkpoints/`, `trials/` (one Harbor trial dir per rollout sample, with the agent log and verifier result), `rollout_data/` when `checkpoint.save_rollout_data` is on, `hf/` when `checkpoint.save_hf` is on. `modal volume ls $MILES_RESULTS_VOLUME <run_id>` |
| Curves | W&B `<tracking.wandb_entity>/<tracking.wandb_project>`, group = run id. Tags record the recipe, model, sandbox, harness, image version and commit. Per-GPU-node system metrics are labelled `train` or `rollout` |
| The exact Miles arguments | the first lines of the app log; for a shipped recipe, also `miles_ext/tests/golden/<recipe>.json` |

## Lifecycle

- **Resume:** launch the same run id again. It loads the last save in `checkpoints/`.
- **Stop:** `modal app stop <app id>` for the GPU app, by id. The run's sandbox app
  has its own lifetime and idle caps (`sandbox.lifetime_secs`, `sandbox.idle_secs`),
  so sandboxes end even when the trainer is killed. To stop everything a run
  created, `miles_ext.platform.teardown(run_id, recipe, target)` does it in one call.
- **Cost:** GPUs are billed from container start, including weight loading and
  engine start. On large clusters that takes minutes. `--check-only` first is
  always cheaper than a failed launch.

## When something fails

Read the app log from the first `Traceback` or `ERROR`, not the last line. Then:

| Symptom | First check |
| --- | --- |
| `--check-only` reports field errors | the field named in the message, against `DEFAULTS` / `FIELD_RULES` in `recipe.py` |
| `unset variable ... (export MILES_DATA)` | `MILES_DATA` in your shell |
| secret or key errors at start | the `.env` in `MILES_DOTENV_DIR` ([quickstart.md](quickstart.md)) |
| weights not found under `/models` | `model.revision` and `prepare_weights` on the same weights volume the run mounts |
| reward stays 0 on every trial | one trial dir under `trials/`: the verifier output and the agent log. Gold patches scoring 0 point at the task set, not the trainer |
| many trials end in setup or timeout errors | `agent.setup_timeout`, `agent.trial_timeout` and the sandbox provider's capacity |
| the rollout stalls with no errors | log lines from the loop watchdog (`miles_ext/obs/loop_watchdog.py`), which print every thread's stack when the rollout loop stalls 5 s or more |
| an OOM, NCCL, kernel or SGLang crash | [limits.md](limits.md), then open a run-failure issue ([help.md](help.md)) |

Anything below the recipe, such as kernels, SGLang, memory layout or weight sync,
is for the maintainers. Open an issue with the evidence rather than patching around it.
