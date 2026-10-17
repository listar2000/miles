# ext-012: DeepSeek V4.1-Flash training recipe

## Why
Recipes could only build 1-GPU engines and Qwen-sized trainers, so the DeepSeek runs were
hand-launched with argv copied from `scripts/run_deepseek_v4_1.py`. The first V4.1 RL run
needs 8-GPU engines serving the released FP8 + FP4 checkpoint (ext-007), a 16-GPU Megatron
trainer with CP and EP, Adam streamed to disk, R3 and expert QAT (fix-006), all as a recipe.
The first config is an overfitting run (the same 16 TB4 tasks every step) that checks the
stack end to end before a run on the full task set. The reason for each value: Linear MOB-187.

## Decision
- **Model-specific flags live in the preset**, copied from the upstream launcher, which stays
  the source of truth: `env`, `sglang_args`, `megatron_args`, `megatron_cp_args`, rollout
  `precisions` (format -> engine env), `train_dir_name` (the trainer's BF16 copy when the
  engines serve a quantized checkpoint), `packed_batches` (V4.1: `--micro-batch-size 1`,
  bshd) and `attention_backend` (V4.1: none, its spec decides). Recipe code stays generic.
- **New recipe fields, each defaulting to the old argv** (the three Qwen recipes resolve to
  byte-identical argv, env and Harbor env): `rollout.*` (engine size, precision, context
  length with auto-truncate, running requests, expert QAT), `optimizer.*` (disk streaming,
  moment dtype, gradient dtype: `bf16` emits upstream's `--grad-reduce-in-bf16`), `parallel.ep`,
  `algorithm.tis_clip_low`, `algorithm.routing_replay`,
  `algorithm.group_filter` (`nonzero_std` drops groups that all scored the same),
  `checkpoint.save_optimizer`, `sandbox.build_timeout_multiplier`.
- **Validation** refuses: a precision the preset lacks; engines that cross a node or don't
  split the rollout GPUs; FP8 + FP4 without Megatron or a BF16 copy, or with `save_hf`
  (the HF export would use the quantized config); expert QAT without disk streaming or
  with expert DP > 1; an EP that does not divide a pipeline stage; Megatron-only fields on FSDP.
- **Rejected: launching with `scripts/run_deepseek_v4_1.py` directly.** It cannot run the
  fork's agentic stack (Harbor agent function, Sail sandboxes, session server, quantizer, our
  Modal launcher and platform contract), so every run would be a hand-edited copy of it.
- **Rejected: model flags in the recipe YAML.** Every V4.1 config would carry its own copy of
  the same ~40 flags; the preset holds them once and the YAML keeps only run decisions.
- **Task list as a file next to the config** (`dsv41-flash-tb4-overfit16.tasks`): the ids are
  the source, the prompt file is derived from Run 4's with the command in its header.
- No colocated V4.1 settings (engram host table, CPU peak device): these engines never sleep.

## How it is applied
`miles_ext/recipes/presets.py` (`deepseek-v4.1-flash`), `miles_ext/recipes/recipe.py`,
`miles_ext/recipes/configs/dsv41-flash-tb4-overfit16.yaml` and `.tasks`, tests in
`miles_ext/tests/test_dsv41_recipe.py`. Verify with `--check-only` on the host before the
launch. On a sync, diff the preset against `scripts/run_deepseek_v4_1.py` (`_train`).

## Remove when
Not an upstream change: kept for as long as recipes are the fork's run format.
