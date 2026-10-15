# ext-006: platform API (miles_ext.platform): validate, teardown, per-run secret, weights cache, checkpoints

## Why
The platform (Line 2, "Platform–Trainer Interface",
https://claude.ai/artifact/CiiRmYMbbAsM7Kv8KGMXi6) supervises runs it does not
understand: every Miles and Harbor detail stays in the fork, behind a few calls
that take a `target` (where the run's resources live). This page covers the
design's fork requirements R2, R4, R7, R8 and R9, plus a minimal R1 (schema).
R3 (`launch`), R5/R6 (`observe`, supervisor, exit record) come in r4.

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
  `<weights_volume>-locks` (stale after 2 h). Branches resolve to a sha first.
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
  save; pinned steps are never deleted. `delete_checkpoint` builds the path from a
  validated run id, an integer step and a kind, so only `<run>/checkpoints/iter_N`
  and `<run>/hf/iter_N` can be deleted, and it refuses the save the run resumes
  from. The test launcher applies final retention after a successful run.
- **R1 (minimal)** (`schema.py`): JSON Schema derived from `recipe.DEFAULTS`,
  defaults, presets, reserved flags, namespaces. The full spec model is r4.

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
Tests: `miles_ext/tests/test_platform.py`, `test_sandbox_cleanup.py` (CPU only).
Live checks on the dev environment are in the r3 release note.

## Remove when
Not an upstream gap: this is our platform's API. It changes with the platform
(r4: launch, observe, supervisor, exit record).
