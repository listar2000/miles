# fix-006: MXFP4 QAT for routed experts and the EVEN E8M0 scale rule (upstream port)

## Why
An FP4 rollout engine re-encodes the trainer's BF16 routed experts on every weight sync. An update
smaller than one MXFP4 step (all of them at lr 1e-6) rounds back to the old value, so the engine
keeps serving the old experts while the trainer's move. Upstream measured this on MiMo-V2.6:
train/rollout KL reached 3.28e-3 by step 14, against 7.3e-4 for a BF16 engine. The ceil scale rule
also doubles a block's scale on a one-ULP drift above 6 * 2^s, requantizing the whole block.

## Decision
Port upstream's fix verbatim rather than write our own. Source: radixark/miles branch
`jiajun/mimo26-mxfp4-qat` by Jiajun Li, commits c300c42bc, 1b3a99708 and ddb2b211c.
- `quantize_mxfp4` uses the EVEN E8M0 rule (torchao's `ScaleCalculationMode.EVEN`). This applies
  to every caller, including Kimi K3's compressed-tensors MXFP4 sync, as upstream intends.
- `project_mxfp4`: the BF16 values an MXFP4 engine decodes.
- `--mxfp4-qat-routed-experts`, off by default: the NVMe optimizer store copies the routed experts
  into the params as their MXFP4 decoding, and the fp32 mains keep the full update. It requires
  `--stream-optimizer-state-to-disk` with Adam and expert data parallel size 1, checked at startup.

Not ported: 61f7ff8b1 (MXFP4 detection in the FP8 dispatch for MiMo checkpoints), since nothing
here calls it; the DeepSeek V4.1 layout goes through ext-007 instead.

## How it is applied
`miles/utils/mxfp4.py`, `miles/utils/fork_arguments.py` (flag, hook-014), `miles/utils/arguments.py` (validation),
`miles_plugins/optimizers/nvme_stream.py` (projection in `_copy_main_to_model_params`); every spot
is marked. Verify: `tests/fast/backends/megatron_utils/test_quantizer_mxfp4.py`,
`tests/fast/optimizers/test_nvme_stream_mxfp4_qat.py` (CPU) and
`tests/fast-gpu/test_nvme_optimizer_streamed_grads.py` (GPU). On a sync where upstream main has
these commits, take upstream's side and drop the markers.

## Remove when
Upstream main carries `--mxfp4-qat-routed-experts` and the EVEN rule. Status 2026-10-16: on the
branch, not merged.
