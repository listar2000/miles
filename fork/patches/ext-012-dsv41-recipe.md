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
  `checkpoint.save_optimizer`, `sandbox.build_timeout_multiplier`, `cluster.disk_gib` (Modal's
  `ephemeral_disk`; required with disk streaming, since the default 512 GiB quota is too small).
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
- **Text-only engines** (`--sglang-json-model-override-args '{"vision_n_layers": 0}'` in the
  preset). With the released checkpoint's vision tower loaded, every MoE layer routes through
  SGLang's `vision_topk`, which never captures routed experts, so R3 got an all-zero payload
  (dsv41-tb4-overfit16-2). Text tokens route the same either way (the VL bias applies only to
  image tokens); RL here is text-only and the BF16 copy has no vision tensors. Rejected:
  patching `vision_topk` to capture (a fork patch on the SGLang base image for a path we never
  use); SGLang's `--language-model-only` (the DSV4 model never reads it: the VL bias is gated on
  `vision_n_layers` alone, `models/deepseek_v4.py`).
- No colocated V4.1 settings (engram host table, CPU peak device): these engines never sleep.
- **The preset serves DeepSeek-V4.1-Flash-1017**, a weights-only update of the first release:
  its `config.json`, `model.safetensors.index.json` and tokenizer files are byte-identical
  and every shard has the same size. The overfit config pins the HF commit (`model.revision`),
  so the engines read the `prepare_weights` snapshot; the BF16 cast is `DeepSeek-V4.1-Flash-1017-bf16`.
- **Engine weight checksum audit off in `miles_args`, not in the recipe.** `--save` derives an
  event dir, which defaults `--log-inference-engine-weight-checksums` on; the post-sync checksum
  crashed the FP8 + FP4 engines (dsv41-tb4-overfit16-1). Every launch passes the `--no-` flag, so a
  recipe's `extra_args` may not repeat it (reserved-flag check).
- **Trainer allocator follows the mode.** Fully async trainers get
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; colocated ones keep `False`, because
  torch_memory_saver rejects expandable segments. With it off, overfit-1018-a1's step-0 cache
  (72 GiB reserved, unallocated) fragmented step 1 into an OOM; two 256k steps fit with it on
  (replay-r3-2, with fix-014).
- **`checkpoint.save_rollout_data`** writes each trained batch to `<run>/rollout_data/<id>.pt`
  (Miles `--save-debug-rollout-data`, now reserved), so a trainer failure replays on trainer
  nodes alone. On in the overfit config.

## How it is applied
`miles_ext/recipes/presets.py` (`deepseek-v4.1-flash`), `miles_ext/recipes/recipe.py`,
`miles_ext/recipes/configs/dsv41-flash-tb4-overfit16.yaml` and `.tasks`, `dsv41-flash-overfit8.yaml` and `.tasks` (8 tasks x 8 samples, 3 nodes, 2 x TP4 engines), tests in
`miles_ext/tests/test_dsv41_recipe.py`. Verify with `--check-only` on the host before the
launch. On a sync, diff the preset against `scripts/run_deepseek_v4_1.py` (`_train`).

## Remove when
Not an upstream change: kept for as long as recipes are the fork's run format. Drop the
text-only override when SGLang's `vision_topk` captures routed experts (check
`multimodal/dsv41/vl_routing.py` on each base-image bump) or a V4.1 run needs image inputs.
