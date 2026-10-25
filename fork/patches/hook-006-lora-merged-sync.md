# hook-006: `--lora-merged-sync` and `--custom-lora-spec-path` for native LoRA

## Why
Upstream native LoRA (#1792) serves the trained adapter through SGLang LoRA. SGLang has no LoRA
path for DeepSeek-V4.1 (its attention and the MXFP4 routed experts), and the LoRA rollout path
turns off the weight-version, staleness and sample-ownership checks the fully async recipes rely
on. The native registry also lists only the architectures Miles ships, so V4.1 cannot be
described without editing it.

## Decision
- `--lora-merged-sync`: every weight sync sends the full weights with each adapter folded in, using
  upstream's own `merge_lora_into_weights` (fp32 `W + alpha/r * B @ A` from the frozen base). The
  engines see a full-parameter model, so the full-parameter checks stay on: the engine version
  check, `assert_weight_version_is_published` and the sample-ownership checker. The version comes
  from the model companion the raw provider installs (bumped every optimizer step); the hook saves
  the companions in the LoRA checkpoint, so the version continues across a resume.
- LoRA resume works like a full checkpoint (native LoRA had none: a LoRA save dir has no Megatron
  tracker, so a restart restarted the data at rollout 0 and reset the optimizer). When `--load` holds
  the run's own LoRA saves, the latest iteration with both `iter_N/adapter` and `rollout/N` resumes:
  base weights from `--ref-load`, adapter and optimizer from `iter_N/adapter`, the rollout side from
  `rollout/N`, and the companions in full. `--lora-adapter-path` alone seeds a new run: fresh data, and
  only the weight version restored, since sample counts would collide with the new numbering.
- Refused with it: bridge mode (no native adapters), `--lora-train-only`, multi-LoRA, `--colocate`
  and `--offload-train` (the merge reads the live trainer weights at each sync). The placement check runs
  after Megatron's validation, once `--offload`, `--use-critic` and `--colocate` have set `offload_train`.
- `--custom-lora-spec-path <module>` registers an architecture from outside Miles: `MODEL_TYPE`,
  `ARCH_SPEC` (a `LoRAArchSpec`), `HF_LAYER_PREFIX`, `HF_UNEMBED`, `hf_lora_groups(hf_config)`. It
  never replaces a spec Miles ships. First user: ext-030, `miles_ext.lora.dsv41`.

Rejected: SGLang LoRA for V4.1 (not implemented upstream in SGLang). Rejected: editing the
registry for V4.1 (an upstream file per architecture, conflicts at each sync).

## How it is applied
`miles/utils/lora/arguments.py` (flags, `load_custom_lora_spec`, `_validate_lora_merged_sync`),
`miles/utils/lora/utils.py` (`lora_merged_sync_enabled`; `lora_rollout_enabled` is false with it;
`lora_without_weight_versions`, the one condition the version checks, the engine version check and
the ownership checker read), `miles/utils/weight_version.py` and `miles/utils/arguments.py` (checks
stay on),
`miles/backends/megatron_utils/actor.py` (merged weights getter, engine version check),
`miles/backends/megatron_utils/model.py` (spec loaded before the LoRA provider wrap),
`miles/backends/megatron_utils/lora/utils.py` (companions in `training_state`),
`miles/backends/megatron_utils/megatron_config.py` (resume from `--load`) and
`miles/ray/rollout/rollout_executor.py` (rollout state from the LoRA run's dir) and
`miles/rollout/session/config.py` (the session server's config carries the flag, so agentic requests
name no adapter). Every spot is
marked. Test: `tests/fast/backends/megatron_utils/test_lora_merged_sync.py`. Checkpoints saved
before the hook have no companions and restore as before.

## Sync risks
- `load_custom_lora_spec` writes into upstream's private `_HF_LORA_MODELS` and builds a private
  `_HfLoraModelSpec`. An upstream rename or a change to that dataclass breaks registration at
  startup (loudly). On a sync, check both still exist with the same fields.
- The companions key in the LoRA `training_state` is written for every LoRA run, not only merged
  ones. Loading ignores its absence, so older checkpoints still resume. If upstream adds its own
  companion state to the LoRA checkpoint, drop ours. A good candidate for an upstream PR.

## Remove when
Upstream syncs merged LoRA weights through the full-parameter path and accepts architecture specs
from outside the registry.
- The LoRA resume branch in `resolve_args_checkpoint_load` keys on the save layout `iter_NNNNNNN/adapter`
  and `rollout/N`. If upstream adds a tracker or renames either, resume falls back to a fresh run
  (quietly): on a sync, check `save_checkpoint_with_lora` and `compute_rollout_checkpoint_dir`.
- Since v0.1.3, upstream's resume selection runs first: an explicit `--ckpt-step` or
  `--exit-on-missing-checkpoint`, or a Megatron tracker under `--load`, returns before the LoRA
  branch. LoRA saves write no tracker, so a LoRA resume reaches it only without those flags.
