# ext-010: training-task filter sweep

## Why
Before training on a generated taskset (first: `tb4-targeted-model-evaluated-20261016`,
1330 tasks) we drop tasks that cannot give a clean reward signal: the task needs GPUs,
the reference solution fails its own verifier on Sail, the training harness
(mini-swe-agent) cannot be installed in the task's sandbox, or the verifier is slow
enough to stall rollouts. Past platform evals covered only 331 of the 1330 tasks at
their current version, so every task is checked again.

## Decision
- Filter 1 (GPU) is read from `task.toml` without a sandbox. Filters 2-4 share one Sail
  sandbox per task, built as a training rollout builds it: the recipe's Harbor
  environment (`recipe.harbor_env`, Sail provider, lifetime/idle caps) and the trainer's
  own `agent_function.build_trial_config`. Only the agent and the caps differ.
- The agent is Harbor's oracle with the training harness's setup in front of it
  (`probe_agent.ProbeOracleAgent`: `SandboxMiniSweAgent` setup with public network for
  setup only, as ext-009, then a `mini-swe-agent --help` smoke). An install failure is
  recorded, never raised, so the oracle and verifier still run.
- The drop rules live in one place, `summary.reasons`; `final` applies them with the
  verifier cap each run recorded.
- Alternatives: the platform's harbor-eval oracle runs (they use the platform's
  `ManagedSetupAgent`, not our harness, Harbor pin or Sail provider, so they do not
  show what training sees); separate sandboxes per filter (three builds per task);
  filtering at launch time inside the trainer (each run pays for the broken tasks again).

## How it is applied
- Files: `miles_ext/taskfilter/` (`tasks.py` input builder, `mobius.py` platform reads,
  `sweep.py` Modal app `miles-taskfilter`, `probe_agent.py`, `summary.py`, `final.py`).
  Tests: `miles_ext/tests/test_taskfilter_{summary,final}.py`.
- Run, from the launch folder (its `.env` gives `SAIL_API_KEY`, the only key sent to the
  containers) with a `mobius login`:
  1. `python -m miles_ext.taskfilter.tasks --taskset <slug> --project <id> --out tasks.jsonl`
     (GPU tasks go to `tasks.gpu.jsonl`).
  2. `modal run -m miles_ext.taskfilter.sweep --check-only`, then
     `modal run --detach -m miles_ext.taskfilter.sweep --tasks tasks.jsonl --run full1`.
     The sweep is spawned, so a lost client does not cancel it. Rerun with a new `--run`
     on a JSONL of the transient failures.
  3. `python -m miles_ext.taskfilter.final --taskset <slug> --tasks tasks.jsonl --out <dir> <sweep dirs, oldest first>`.
- Output on Volume `miles-fork-results`, `taskfilter/sweep-<run>/`: `summary.jsonl`,
  `launch-*.json` (recipe, image, concurrency) and small trial files. Sail app:
  `miles-mobius-taskfilter-<run>`, terminated by exact name when the sweep ends.
- Image: `stack.IMAGE_REPO:stack.LATEST_TAG` (the image training runs), or
  `TASKFILTER_IMAGE` to pin one. Pin `<repo>:<sha10>` for a campaign so the full sweep and
  its retries run on the same build; each `launch-*.json` records the ref used. Only `miles_ext/taskfilter/` is added over it; all other
  `miles_ext` code is the image's. The first sweep ran on
  `miles-mobius@sha256:5734ab8f…` (198600a509, ext-009).
- Task files come through the Mobius API from the `frontierbench` storage connection on
  staging (`mobius.CONNECTION`) onto each container's disk and are deleted after the
  trial; nothing is downloaded to the launch host. Concurrency: `TASKFILTER_CONCURRENCY`
  (default 450).
- On a sync: nothing upstream is touched. If Harbor's `OracleAgent`, `TrialPaths` or
  `result.json` change, `--check-only` and the summary tests show it.

## Training notes from the first sweep
- Harbor's 600 s default image-build cap applies in training too. Under about 450
  concurrent cold Sail builds, 88 tasks hit it; at concurrency 100 all but 9 built.
  Sail caches built images (a rebuild takes about 2 s), so warm the cache before a
  large run.
- Sail rejects images from `mcr.microsoft.com`, any `# syntax=` directive, build
  contexts over 10000 files and CPU requests above the largest Sailbox (8 vCPUs;
  two tasks ask for 16).

## Remove when
The filter results are folded into the platform's task metadata, or a later sweep
replaces this one. Upstream status: not applicable (fork-only tooling).
