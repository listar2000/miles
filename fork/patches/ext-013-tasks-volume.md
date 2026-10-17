# ext-013: task dirs on a Modal volume (`taskset.tasks_volume`)

## Why
A hand launch baked `taskset.tasks_dir` into the image (`add_local_dir` to `/inputs/tasks`), so
the task dirs had to live on the launching machine and went into every launch's image (Run 4's
52 TB4 tasks: 364 MB). The launch host is meant to stay thin: repo, `.env`, `modal` CLI and small
prompt files only.

## Decision
- New recipe key `taskset.tasks_volume: <volume>:/<path>`, used instead of `tasks_dir`. Every
  launcher function mounts that volume read-only at `/datasets`; `HARBOR_TASKS_DIR` points at
  `/datasets/<path>`. The volume name must be in our namespaces (`OWN_APP_NAMESPACES`).
- The prompt file stays local and is still added to the image at `/inputs/prompts.jsonl`: it is
  small, it is the run's own task selection, and Miles reads it as `--prompt-data` unchanged.
- A dataset on the volume is written once and never changed; a new version gets a new path
  (`miles-datasets:/tb4/v1/tasks`, then `/tb4/v2/...`).
- The deep check now fails when a prompt row's `metadata.instance_id` has no task dir, for both
  `tasks_dir` and `tasks_volume`.

Rejected: the platform's per-run copy to `<results>/<run_id>/inputs` for hand launches (a copy of
the whole task set per run, and it still needs the tasks somewhere to copy from); filtering a
volume-side prompt file by a task-id list in the container (moves the run's selection off the
recipe's reviewable inputs for no gain while prompt files are KB-sized).

## How it is applied
`miles_ext/recipes/recipe.py` (`tasks_volume`, `tasks_path`, validation), `miles_ext/launch/modal_app.py`
(the mount, carried into the container through `MILES_EXT_TASKS_VOLUME`), `miles_ext/platform/validate.py`
(`_inputs(tasks=False)`, the id check); tests in `miles_ext/tests/test_tasks_volume.py`.
`tasks_dir` recipes and platform specs (no `tasks_volume`) resolve and launch as before.
Upload a task set once: `modal volume put miles-datasets <local dir> /<name>/v<N>/tasks`.

## Remove when
Not planned: this is the default way to point a recipe at a task set.
