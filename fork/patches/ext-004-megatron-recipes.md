# ext-004: Megatron trainer in recipes, including trainers that span nodes

## Why
Recipes could only produce FSDP argv on one trainer node (`--actor-num-nodes 1`,
FSDP-only flags always on). Miles defaults to Megatron, and the DeepSeek run
needs Megatron on several trainer nodes, so the fork needs a recipe path for it.

## Decision
- `model.backend: megatron` picks the backend per recipe. The preset names the
  model's Megatron definition (`megatron_model_type`, i.e. `scripts/models/<type>.py`),
  and the recipe sets `parallel: {tp, pp, cp}`. DP is whatever trainer GPUs remain.
- Weights load straight from HF: `--ref-load <hf dir>` is the fresh-start load
  path (`megatron_config.resolve_args_checkpoint_load`), and mbridge converts on
  load. With `kl_coef 0` no reference model is built. There is no `torch_dist`
  conversion step, as in the archived DeepSeek launcher. Resume uses `--load` as before.
- A trainer of more than 8 GPUs spans nodes. Miles gives the trainer the first
  GPUs in (node IP, GPU id) order and uses `actor_num_nodes * actor_num_gpus_per_node`
  only to size the placement group, so 12 trainer GPUs on 2 nodes are declared
  as 2 x 6 and placed 8 + 4. This is the cheapest shape that puts Megatron
  traffic across nodes. The alternative (3 nodes, 16 + 8) is Miles' usual shape
  and costs 1.5x; the smoke chose 12 + 4 (design: https://claude.ai/artifact/Ee8jnfHfL9fHbdNvgUgJn6).
- `--override-opt_param-scheduler`: the LR scheduler's length is num_rollout x
  batch, and Megatron refuses a resume whose length differs from the checkpoint's
  (smoke launch B: 240 vs 144 after raising num_rollout from 3 to 5). This
  launch's schedule wins; with the recipes' constant LR nothing else changes.
- `recipe.load` rejects layouts Megatron would reject minutes into a cluster
  start: tp*pp*cp must divide the trainer GPUs, TP must stay inside a node and
  divide the heads and KV groups, PP must divide the layers, and DP must divide
  `rollout_batch_size * n_samples`, because Miles splits each step's samples
  into equal parts per DP rank.

## How it is applied
`miles_ext/recipes/presets.py`, `miles_ext/recipes/recipe.py` (`backend`,
`trainer_layout`, `megatron_parallel`, `_validate_trainer`, `_backend_args`,
`trainer_env`: Megatron needs `CUDA_DEVICE_MAX_CONNECTIONS=1` in every process
that parses args, so the launcher sets it before Ray starts; the launcher also
puts `/root/Megatron-LM` on PYTHONPATH as upstream launchers do, and the dry
check imports what the trainer imports when it builds the model),
recipe `configs/qwen35-9b-swegym8-megatron-2node.yaml`. The launcher's dry check
prints the layout. FSDP recipes produce byte-identical argv. Verify with
`--check-only`, then a two-launch smoke (save, then resume with the same run id).
On a sync, recheck `resolve_args_checkpoint_load` and the placement-group
ordering in `miles/ray/placement_group.py`.

Validated 2026-10-14 by run `smoke-megatron-2node-v1` on 2 x H200:8 (W&B
miles-smoke c9gnl5in, then resumed in k85jd69f). The trainer was placed 8 + 4,
the trainer/rollout logprob gap was 0.0074 to 0.0079 over 5 steps (one-node FSDP
baseline: 0.007 to 0.012), weight pushes took 0.5 to 4 s, and checkpoints
iter 2 and iter 4 each had 12 shards. Launch log:
`run-logs/2026-10-14-megatron-2node-smoke.md` in the launch folder.

## Remove when
Not an upstream change: kept for as long as recipes are the fork's run format.
