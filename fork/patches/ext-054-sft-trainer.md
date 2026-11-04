# ext-054: a dataset-agnostic SFT trainer and Modal launch, driven by one run config

## Why
`miles_ext.sft` turned records into token shards but had no way to train on them. The tmax SFT runs
(2026-11-01, local branch `sft/qwen36-tmax-tb21`) carried their own driver, validation, Modal launch and model
table under `miles_ext/sft/tmax/`, so the next dataset (GLM-5.3-Flash Pi traces) would have trained through
`tmax.train`.

## Decision
- One YAML run config (`miles_ext/sft/config.py`) names the preset, the pinned image, records and shards on a
  data volume, and the schedule; `modal_prepare` and `modal_train` both read it, so a run is one file.
- Values that move together are derived: tokens per GPU = `data.max_tokens / train.cp`; weights at
  `/models/<preset dir>`; the layout is fixed to one H200:8 node, TP1 CP<cp> EP8 (the tmax layout).
- Model facts come from `miles_ext/recipes/presets.py` (new preset `qwen3-30b-a3b-thinking-2507`), not a second
  table. Validation ids live in `rollout.py` (`VAL_BASE`; `MILES_SFT_VAL_DIR` is set by the launcher only), padded to
  a multiple of `--global-batch-size` with untrained copies because Miles trims rollouts to that multiple.
- `SftTrainRayActor` is selected with hook-019's `--custom-trainer-actor-class`.
- Fixed, not configurable: full recompute (selective recompute of MoE activations and norms ran out of memory at
  32K tokens per GPU), one save per epoch, the whole held-out set.
- The image is the config's published image (`image`, digest-pinned) or one Modal builds from a pushed fork commit
  (`commit`, for code no published image has yet), with the checkout's `miles_ext/` and `experiments/` laid over it,
  as the tmax runs did, so an SFT change runs without an image build. Lost: the image alone no longer pins the
  code; `argv-*.json` in the run dir keeps the config and argv. The overlay cannot add `miles/` code, so
  `modal_train` refuses an image without hook-019.
- Presets with one sample per micro-batch (`packed_batches=False`, DeepSeek V4.1) are refused until SFT is
  verified for them; the renderer supports more families than the trainer.
- Rejected: an SFT mode of the RL recipe (`recipe.py`): most of its keys (taskset, rollout, harness) mean
  nothing for SFT.

## How it is applied
`python -m modal run -m miles_ext.sft.modal_prepare` then `-m miles_ext.sft.modal_train --run-id <id>`, with
`MILES_SFT_CONFIG` (`miles_ext/sft/README.md`). Verify: `miles_ext/tests/test_sft_config.py`, then a two-step
smoke (`--num-rollout 2`) and the same run id again with `--num-rollout 3` for resume. Nothing on a sync beyond `train.py`'s use of `create_rollout_components`,
`create_training_models` and the Megatron actor's `train` signature.

## Remove when
upstream Miles ships an SFT entry point for prepared samples; upstream status: not filed
