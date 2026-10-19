# ext-006: platform API (miles_ext.platform): validate, launch, observe, teardown, per-run secret, weights cache, checkpoints

## Why
The platform (Line 2, "Platform–Trainer Interface",
https://claude.ai/artifact/CiiRmYMbbAsM7Kv8KGMXi6) supervises runs it does not
understand: every Miles and Harbor detail stays in the fork, behind a few calls
that take a `target` (where the run's resources live). This page covers the
design's fork requirements R2, R4, R7, R8 and R9, plus a minimal R1 (schema).
r5 adds R3 (`launch`), R5 (rank-0 supervisor, `observe`) and R6 (exit record).
r6 adds named training knobs (the platform set five Miles flags through
`extra_args`, which a named-fields API must not need) and the rest of the r5
adversarial review (https://claude.ai/artifact/RVodMdDTRvXQ9z3mSNAtud).

## Decision
- **Target** (`target.py`): `modal_environment`, `app_name` (GPU app),
  `sandbox_app_prefix`, `results_volume`, `weights_volume`, `secret_name`.
  Every name must be in our namespaces (`miles-`, `mdev-`, `mrl-`), so no verb
  can reach another team's apps in the shared Modal workspace or Sail org.
  Defaults are the hand-launched setup; the test launcher reads the same values
  from env (`MILES_APP_NAME`, `MILES_RESULTS_VOLUME`, `MILES_WEIGHTS_VOLUME`,
  `MILES_SECRET_NAME`, `MODAL_ENVIRONMENT`).
- **R2 validate** (`validate.py`): instant checks (recipe, names, reserved
  flags in `extra_args`) then the deep check in the fork image, which is the old
  `check` body made structured: errors as `{path, message}` instead of prints,
  each step recorded and the independent ones still run. FSDP on CPU; Megatron on
  one H100, because its argument parser loads `libcuda.so.1`. `libcuda_stub=True`
  puts the CUDA toolkit's stub on `LD_LIBRARY_PATH` for a CPU run instead (result
  below). The test launcher's `check` now calls the same function, so there is
  one code path.
- **R4 teardown** (`teardown.py`): stops the GPU work (cancel the call; stop the
  app by exact name only when the name is the run's own, e.g. `mdev-<run_id>`,
  never the shared `miles-fork`), terminates the sandbox apps' sandboxes through
  one adapter per provider (`ModalSandboxes`, `SailSandboxes`), deletes the
  secret, and reports `{released, remaining}`. Exact names only: Modal
  `App.lookup` in the target's environment, never created; Sail `App.list`
  filtered by equality, never minted. Sail has no environment or project in its
  SDK; the org of `SAIL_API_KEY` is the scope. The launcher's exit cleanup uses
  the same adapters (the Sail helper moved here from `miles_ext/harbor/sail`).
  The SDK has no public app stop; teardown does what `modal app stop` does.
- **R7 secret** (`secrets.py`): created by the caller from the credential pool,
  only `WANDB_API_KEY`, `HF_TOKEN`, `SAIL_API_KEY`; Sail runs require the Sail
  key. Name `<namespace of app_name>secret-<run_id>` unless the target names one.
- **R8 weights** (`weights.py`): `<org>--<name>/<sha>` on the weights volume, a
  `.ready` marker written after the download's commit, a lock per key in the Dict
  `<weights_volume>-locks`; the holder refreshes it every 5 min while the download
  runs and a lock 30 min without a refresh is taken over (r6, review M5). Branches resolve to a sha first.
  `list_weights` and `delete_weights` (the user's follow-up: cached weights must
  be listable and deletable); delete refuses anything but a keyed snapshot and a
  key being prepared. Recipes read the keyed path when `model.revision` is set,
  else the preset directory as before (listed as `legacy`, never deleted here).
- **R9 checkpoints** (`checkpoints.py`): results volume as a target field;
  `checkpoint.save_hf` (Megatron: `--save-hf <run>/hf/iter_{rollout_id:07d}`);
  `finalize_hf` runs upstream `tools/convert_fsdp_to_hf.py` on FSDP's final save
  in a CPU container; `retention_plan` keeps the newest `checkpoint.keep` full
  saves and the resume point while training, and at the end the final HF export
  and, if `checkpoint.keep_final_full` (the user's binary knob), the final full
  save; pinned steps are never deleted. Miles past v0.1.2 also writes rollout-side
  state to `<run>/checkpoints/rollout/<rollout_id>/`; retention deletes the ones
  older than every kept full save, keeping from the oldest kept step minus one
  (a resume loads rollout `start_rollout_id - 1`: N for Megatron's `iter_N`, N - 1
  for FSDP's), and none when no full save is left. `delete_checkpoint` builds the path from a
  validated run id, an integer step and a kind, so only `<run>/checkpoints/iter_N`
  and `<run>/hf/iter_N` can be deleted, and it refuses the save the run resumes
  from. At the end, the final full save also stays while no HF export exists (FSDP
  before `finalize_hf`), whatever `keep_final_full` says. The test launcher applies
  final retention after a successful run only when standalone: with a per-run
  secret (a platform caller) it leaves checkpoints to the caller, which runs
  `finalize_hf` and then its own retention with its pinned steps
  (`MILES_FINAL_RETENTION=1/0` overrides; r4 fix of two r3 bugs found by the platform).
- **R3 launch** (`launch.py`): `launch(recipe, run_id, target, *, segment_seconds,
  lease_seconds, inputs, from_commit, final_retention)` spawns one detached segment and
  returns `{launched, complete, app_id, call_id, names, timeout}`. Modal binds a
  clustered function's shape at definition, so launch runs the test launcher module in
  a child interpreter with the configuration in `MILES_EXT_LAUNCH`; the caller reads no
  env. It returns without spawning when the run is already complete.
- **Completion** (`progress.py`): from `latest_checkpointed_iteration.txt` and
  `num_rollout` alone. FSDP names a save after the next rollout (`iter_{id+1}`),
  Megatron after the completed one, so `next_rollout_id` is per backend. The trainer
  also checks it on every rank before Ray starts, so a relaunch of a finished run costs
  a container start (r5 fix: a Stage 3 run relaunched once after its final save,
  because the platform's timer killed the container before that save was committed).
- **R5 supervisor** (`supervisor.py`, rank 0): runs Miles as a child; commits the
  results volume after every save; touches `--save-trigger-sentinel` before the
  segment deadline (counted from container start) or lease expiry, waits for Miles to
  remove it (save written), stops Miles and exits cleanly; stops Miles on SIGTERM;
  heartbeat with step and Miles metrics (from `{save}/events/*.jsonl`) to the Dict
  `<ns>progress` key `run:<run_id>`; lease at `lease:<run_id>` (`renew_lease`).
  `observe` reads it back with completion and checkpoints.
- **Multi-node commits** (`shards.py`, r6, review M7): before rank 0 commits a new
  tracker it asks every worker to commit (`commit:<run_id>` with a fresh token) and
  waits for their acks (queue partition `ack-<run_id>`, 300 s at most), so a durable
  tracker never names a save whose other-node shards are not durable. The wait is best
  effort (120 s at most, under the heartbeat's 300 s staleness limit) and never keeps rank
  0 from committing. Workers also commit every 2 min and on exit.
- **One segment at a time**: `launch` claims `active:<run_id>` (refuses while the call it
  names is running), then writes `run:<run_id>` as `pending` with the call id. Heartbeats and
  exit records carry `call_id`; `observe` reports `ended` only when the exit record's call id
  matches the heartbeat's, so a relaunched run reads `pending`, not the previous exit.
  Phases: `pending`, `starting`, `training`, `checkpointing`, `stopping`, `ended`,
  `no_heartbeat`. Rank 0 clears its completion-queue partition before the Ray head starts,
  so a worker never reads a dead segment's message. `create_run_secret` replaces the secret
  (rotated keys apply on relaunch). Checkpoints carry `rollouts_done` (FSDP `iter_N` holds N
  rollouts, Megatron N + 1). (Adversarial review H1-H4, M3, M4, L3, L5, M9.)
- **R6 exit record**: `<run>/exit.json`, `<run>/exits.jsonl` and Dict `exit:<run_id>`,
  reason `success|segment_end|lease_expired|stopped|oom|nccl_timeout|error`, with
  `retryable`, `complete` and the output tail on failure. Teardown deletes the Dict keys.
  r6: `note` holds the first line of the output that names the error (for `oom`, the first
  OOM sign), kept from the whole output, not only the 60-line tail; a Ray worker that died
  with "connection error code 2" (the kernel OOM killer, by Ray's own account) reads as
  `oom` (platform field report, Megatron run 267df4a9), unless an NCCL watchdog timeout came
  first in the output: that worker was taken down by NCCL, so the reason is `nccl_timeout`
  (a DeepSeek-V4.1 weight save, where one rank's closing barrier timed out while the others still wrote).
- **Memory defaults** (r6): `algorithm.max_tokens_per_gpu` defaults to 16384 (was 32768).
  It caps the tokens packed into one training micro-batch, and FSDP holds that micro-batch's
  full-vocabulary logits (about 1 GiB per 1k tokens in fp32 at a 248k vocabulary): 32768
  OOMed Qwen3.5-9B on 4 H200s. `algorithm.max_response_len` caps each model turn (SGLang's
  `max_new_tokens`); a multi-turn trajectory's training response (all turns plus tool
  output) is capped by `algorithm.max_seq_len` instead, which is why mean response lengths
  run past `max_response_len`.
- **Named training knobs** (r6, platform review P-M1): `algorithm.advantage_estimator`
  (`grpo` | `gspo` | `reinforce_plus_plus` | `reinforce_plus_plus_baseline`; not `ppo`,
  which needs a critic the recipe does not build), `algorithm.kl_loss_coef` (> 0 adds
  `--use-kl-loss`; FSDP then also loads the base weights as `--ref-load`, Megatron
  already passes it), `algorithm.entropy_coef`, `algorithm.seed` (null: Miles' 1234),
  and a new `async` section: `over_sampling_batch_size`, `max_concurrent_samples`
  (null: batch x n), `unused_samples` (`drop` | `retry`), `pause_generation`
  (`in_place` | `retract`; Miles rejects `abort` and `--partial-rollout` in fully async
  mode, so neither is offered). Defaults reproduce the r5 argv, and `Target.apply` and
  the trainer fill defaults, so a recipe resolved by r5 (no `async`) still runs.
- **Field checks** (r6, review M1): `recipe.FIELD_RULES` gives each field a type and,
  where it matters, a range or choice list (GPU types, estimators, minimums);
  `field_errors` reports every bad field as `{path, message}` before anything computes
  with the values, so a string where a number belongs is an error, not a TypeError.
  Miles' own asserts on the async sizes are mirrored.
- **Reserved flags** (r6, review M2): `extra_args` may not repeat any flag `miles_args`
  emits for the recipe (computed from its output), nor a named knob's flag; the hand list
  `RESERVED_FLAGS` only names the key to use instead.
- **Teardown reports GPU work** (r6, review M8): it also cancels the call named in
  `active:<run_id>`, stops every live app named exactly like the per-run app even when
  given an app id, and lists calls and apps not yet stopped in `remaining`. The trainer
  runs as an ephemeral app, which a name lookup never finds (the platform saw one survive
  a cancel during launch), so those ids come from the environment's app list filtered
  by equality, as `modal app list` shows it. A test imports the Modal internals the
  app stop relies on, so an SDK upgrade fails there first.
- **Layout** (`layout.py`, r6, review L1): run paths, tracker reads and volume handles
  in one module; nothing else parses the tracker.
- **R1** (`schema.py`): JSON Schema derived from `recipe.DEFAULTS` with the field rules'
  types, enums and bounds; defaults, presets, reserved flags, namespaces.
- **Trainer versions** (`versions.py`): a run names its trainer once, `latest` or a
  stable version `stable-YYYY-MM-DD`; `versions.resolve` turns that into a fixed
  `repo@sha256:...` and the commit inside it (latest: the registry's `main` tag and
  the image's own `MILES_FORK_COMMIT`; stable: `miles_ext/stable.json` read from
  `main` on GitHub, the packaged copy as fallback). The platform stores the result
  and passes `launch(image=...)` to every segment, so a resume never changes build.
  Without `image`, launch still runs `stack.FORK_IMAGE`.
- **Names, option A** (`target.py`): a target with `name_id` (6-16 lowercase
  letters/digits, unique per environment, kept so by the platform) names the run's
  resources `rl-<env>-<id>` (GPU app), `<app>-sbx` (sandbox app, recipe key
  `sandbox.app_name`, set by `Target.apply`), `<app>-secret`, and the shared
  `rl-<env>-progress`; the run id still keys files, W&B and progress keys. `rl-`
  names require a name id. The `mdev-`/`mstg-`/`mrl-` names stay accepted until
  dev is cleaned up.
- **miles-mobius names** (2026-10-15): images carry `MILES_MOBIUS_VERSION` and
  `MILES_MOBIUS_COMMIT` (and the old `MILES_FORK_*` pair; readers take either,
  `images.image_version`/`image_commit`) and `miles.mobius.*` labels, in
  `starlimobius/miles-mobius`. `launch(trainer=...)` tags W&B `miles-mobius:<trainer>`
  plus `upstream:<tag>` (was `fork:<release>`), and the launch and exit records gain
  `trainer` and `version` (the old `stack`/`release` keys repeat `version`).
  `tracking.wandb_name` sets the W&B display name. With a name id the helper app is
  `rl-<env>-helpers` and the completion queue `rl-<env>-completion`.

Alternatives: separate packages per requirement (one dependency web, more
churn for the platform); retention by Miles flags (Miles has none for full
saves); a Megatron Hook for HF export (Miles already has `--save-hf`).

## How it is applied
New files under `miles_ext/platform/`; `miles_ext/launch/modal_app.py` reads the
target from env and uses `validate.run_deep_check`, `teardown.sandbox_adapter`
and `checkpoints.apply_retention_local`; recipe keys `model.revision`,
`checkpoint.keep`, `checkpoint.keep_final_full`, `checkpoint.save_hf`;
`recipe.from_dict` and `recipe.validate(local_inputs=False)` for specs whose
taskset is on the results volume (`<results_volume>/<run_id>/inputs/`).
Tests (CPU only): `miles_ext/tests/test_platform.py`, `test_supervisor.py`,
`test_shards.py`, `test_sandbox_cleanup.py`, `test_vendored_overrides.py`.
The trainer function lives in `miles_ext/launch/modal_app.py`, split by rank
(`_head`, `_worker`); moving it under `platform/` with one config object is open.
The package re-exports functions named like their modules (`schema`, `validate`,
`teardown`, `launch`; review L2); the platform imports the functions, so the
names stay, and tests import modules with `importlib`.
Live checks on the dev environment are in the r3 release note.

## Checked on miles-dev (2026-10-15, commit dc768d5df2)
- `prepare_weights("Qwen/Qwen3.5-9B", "main")` into `mdev-weights`: resolved to
  `c202236235…`, 19.3 GB in 101 s; a lock left by a killed prepare blocked the
  next one as designed (released by hand).
- `validate`: FSDP Sail recipe on CPU ok in 60 s; the 2-node Megatron recipe with
  `save_hf` on one H100 ok in 88 s (`save_hf` resolved to
  `/results/<run>/hf/iter_{rollout_id:07d}`).
- libcuda stub (one try, as planned): fails. The stub gets past loading
  `libcuda.so.1`, but Megatron's `validate_args` queries device properties
  (`cudaGetDeviceCount`: "CUDA driver is a stub library"). Megatron validates on a GPU.
- `teardown` of a run with two live Modal sandboxes in `miles-mobius-<run>`:
  released both, `remaining` empty; a second call released nothing.
- Final retention on dummy saves (`keep_final_full: false`): left only the final
  HF export and `latest_checkpointed_iteration.txt`.
- Not yet exercised on a real save: `finalize_hf` and Megatron's `--save-hf` output.

## Remove when
Not an upstream gap: this is our platform's API. It changes with the platform
(r5: launch, observe, supervisor, exit record; r6: named knobs and the review backlog).
