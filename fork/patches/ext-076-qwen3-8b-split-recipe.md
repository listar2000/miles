# ext-076: qwen3-8b trains on Megatron, and its split recipe replaces Qwen3.5-9B's

## Why
- `qwen35-9b-split-elastic` selects the GDN `linear_attn` projections through LoRA's default targets, and native LoRA has no
  adapters for them. Its `parse_args` asserts in `resolve_adapter_targets`, seen with the same model and defaults in a CPU
  check on 2026-11-07. Upstream's `examples/lora/run_qwen3_lora_native.py` also warns raw mode is unstable on these layers.
- `qwen3-8b` had only run on FSDP. Under Megatron at TP 2, its vocabulary of 151936 pads to 152064 and trainer init fails:
  `ProcessGroupNCCL::scatter: invalid tensor size at index 0 (expected (76032, 4096), got (75968, 4096))`.

## Decision
The recipe is the one a run from an outside launcher completed (W&B `mobiuslab/miles-e2e/tg0m1ckc`): 100 of 100 steps, `complete: true`,
reward 0.27 over the first 20 steps and 0.48 over the last 20, log-prob gap 0.009 to 0.013. Its lr is the official 1e-5:
the same run at 1e-4 reached a gap of 0.05 with a third of replies capped (W&B `mobiuslab/miles-e2e/bv0g0pov`). The vocabulary flag moved from that
run's `extra_args` to the preset, since every Megatron run of this model needs it. The grader lives in `miles_ext/tasks`
because a recipe here reads it; preparing the prompts stays outside this repo. The run's grader was the same code outside
the repo, so its prompt rows named another path.

## How it is applied
`miles_ext/recipes/presets.py`, `miles_ext/recipes/configs/qwen3-8b-split-elastic.yaml`, `miles_ext/tasks/dapo_math.py`.
The recipe's header gives the prompt row format.

## Remove when
The preset flag: when Megatron's HF load accepts a padded embedding. The recipe: when a newer split recipe replaces it.
