# hook-006: `--custom-lora-spec-path` and LoRA resume for native LoRA

Its other half, `--lora-merged-sync` (full weights with the adapter folded in at every sync), was
removed by hook-015 on 2026-10-30: no checked-in recipe used it, and the checked-in LoRA recipes serve the adapter
(hook-010). The page keeps its old file name so links still work.

## Why
The native registry lists only the architectures Miles ships, so V4.1 could not be described
without editing it. Native LoRA also had no resume: a LoRA save dir has no Megatron tracker, so a
restart replayed the data from rollout 0 with a reset optimizer.

## Decision
- `--custom-lora-spec-path <module>` registers an architecture from outside Miles: `MODEL_TYPE`,
  `ARCH_SPEC` (a `LoRAArchSpec`), `HF_LAYER_PREFIX`, `HF_UNEMBED`, `hf_lora_groups(hf_config)`. It
  never replaces a spec Miles ships. First user: ext-030, `miles_ext.lora.dsv41`.
- LoRA resume works like a full checkpoint. When `--load` holds the run's own LoRA saves, the
  latest iteration with both `iter_N/adapter` and `rollout/N` resumes: base weights from
  `--ref-load`, adapter and optimizer from `iter_N/adapter`, the rollout side from `rollout/N`, and
  the model companions (weight version, sample consumptions) in full. `--lora-adapter-path` alone
  seeds a new run: fresh data, and only the weight version restored, since sample counts would
  collide with the new numbering.

Rejected: editing the registry for V4.1 (an upstream file per architecture, conflicts at each sync).

## How it is applied
`miles/utils/lora/arguments.py` (flag, `load_custom_lora_spec`),
`miles/backends/megatron_utils/model.py` (spec loaded before the LoRA provider wrap),
`miles/backends/megatron_utils/lora/utils.py` (companions in `training_state`),
`miles/backends/megatron_utils/megatron_config.py` (resume from `--load`) and
`miles/ray/rollout/rollout_executor.py` (rollout state from the LoRA run's dir). Every spot is
marked. Test: `tests/fast/backends/megatron_utils/test_lora_resume.py`. Checkpoints saved before
the hook have no companions and restore as before.

## Sync risks
- `load_custom_lora_spec` writes into upstream's private `_HF_LORA_MODELS` and builds a private
  `_HfLoraModelSpec`. An upstream rename or a change to that dataclass breaks registration at
  startup (loudly). On a sync, check both still exist with the same fields.
- The companions key in the LoRA `training_state` is written for every LoRA run. Loading ignores
  its absence, so older checkpoints still resume. If upstream adds its own companion state to the
  LoRA checkpoint, drop ours.
- The LoRA resume branch in `resolve_args_checkpoint_load` keys on the save layout `iter_NNNNNNN/adapter`
  and `rollout/N`. If upstream adds a tracker or renames either, resume falls back to a fresh run
  (quietly): on a sync, check `save_checkpoint_with_lora` and `compute_rollout_checkpoint_dir`.
- Since v0.1.3, upstream's resume selection runs first: an explicit `--ckpt-step` or
  `--exit-on-missing-checkpoint`, or a Megatron tracker under `--load`, returns before the LoRA
  branch. LoRA saves write no tracker, so a LoRA resume reaches it only without those flags.

## Remove when
Upstream accepts architecture specs from outside its registry (Miles main has a built-in V4.1
spec, `f3088f8a8`, #4205: the v0.1.4 sync can drop the spec half) and resumes a LoRA run from its
own saves; upstream status: not filed.
