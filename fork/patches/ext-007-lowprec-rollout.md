# ext-007: DeepSeek V4.1 rollout in its released FP8 + FP4 format

## Why

Run 4 served DeepSeek V4.1-Flash from its released checkpoint: dense layers in FP8 (32x32
power-of-two scales) and routed experts in FP4 (MXFP4: E2M1 plus one E8M0 scale per 32). That was
about 2x faster per request than our BF16 copy, and the weights per GPU fell from about 164 GB to
about 70 GB. RL could not use it:

- No Miles quantizer writes that layout. Dispatch picks one method per checkpoint. The FP8 quantizer
  asserts on 32x32 blocks on Blackwell, and on Hopper it sends scales that are not powers of two. It
  would also send the experts as FP8 and silently corrupt V4.1's indexer `wk` (BF16 in the engine)
  and engram `wkv` (FP8 in the engine).
- Re-encoding BF16 experts to FP4 after every step rounds away small updates; that half is
  fix-006.

The engine side already works in our image: SGLang `d1faf82` reloads DSV4 MXFP4 experts in place
on the FlashInfer TRT-LLM path.

## Decision

- **ext-007** `miles_ext/lowprec/checkpoint_quantizer.py`: each tensor is sent in the format the
  rollout checkpoint (`--hf-checkpoint`) stores it, read from its safetensors headers.
  - `F8_E4M3` becomes FP8 with power-of-two block scales.
  - `I8`/`U8` with a `.scale` becomes MXFP4.
  - Anything else goes as BF16, except `attn.wo_a`, which the engine holds in BF16.
  - Scales are sent as fp32 powers of two.
  - Names and shapes are checked against the headers, so a layout mismatch fails at the first sync
    instead of corrupting silently.
  - Alternatives: name lists per model (what the built-in quantizers do; they are wrong for V4.1's
    `wk`/`wkv` today), and upstream's MiMo `quantizer_fp8_mxfp4.py` (fused-qkv and MiMo names, and
    it excludes DSV4).
- Plugged in through hook-001 (`--custom-weight-quantizer-path`); expert QAT is fix-006
  (`--mxfp4-qat-routed-experts`).

Out of scope, by the user's call (2026-10-16): off-policy corrections (TIS/MIS, clip ranges,
staleness). They are independent of the drift source and belong to RL recipe design.

## How it is applied

No launcher change. With `scripts/run_deepseek_v4_1.py`:

```
--hf-checkpoint <released DeepSeek-V4.1-Flash dir>    # the engine loads it, engram tables included
(no --load-from-hf: --ref-load stays <model-dir>/DeepSeek-V4.1_torch_dist, built from the BF16 copy)
--extra-env-vars 'SGLANG_DSV4_FP4_EXPERTS=1'          # overrides the script's 0
--extra-args '--custom-weight-quantizer-path miles_ext.lowprec.checkpoint_quantizer.quantize_like_checkpoint
              --check-weight-update-allow-quant-error [--mxfp4-qat-routed-experts]'
--disk-offload                                         # QAT needs --stream-optimizer-state-to-disk
```

- The engine cannot start on dummy weights: the engram tables are never synced.
- Keep the script's `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`, `SGLANG_OPT_FP8_WO_A_GEMM=0` and
  `SGLANG_OPT_FUSE_WQA_WKV=0`: SGLang refuses or mishandles online updates otherwise.
- Use broadcast or broadcast_packed transfer. SGLang's `update_weights_from_disk` does not restore
  the FP4 expert layout, so disk-delta is unsafe here.

Verify:
- `miles_ext/tests/test_lowprec_checkpoint_quantizer.py`: CPU tests that FP8 and FP4 tensors from
  the checkpoint re-encode to the same values, that the QAT-projected experts arrive exactly, and
  per-tensor format, name and shape checks.
- On GPU: a 4-layer V4.1 run with `--check-weight-update-equal`.

## Remove when

Upstream syncs V4.1 in its FP8 + FP4 format natively. Status: not filed.
