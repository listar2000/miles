# fix-082: a hybrid-model DP rank with no rows stays idle in speculative verify (SGLang)

## Why
GPU-delta serving (ext-053) runs data-parallel attention: each GPU of an engine is its own attention rank. With EAGLE
drafting (the GLM-5.3-Flash preset's NextN layer, ext-051) on a hybrid linear-attention model (GLM-5.3-Flash's KDA
layers), `ForwardBatch.prepare_mlp_sync_batch` turns an IDLE rank into a TARGET_VERIFY batch whenever the global
batch verifies, even when the rank has no padded rows (SUM_LEN padding). The 0-row verify batch reaches the DSA
backend, whose `init_forward_metadata` reads `seq_lens_cpu.max()` of an empty tensor, and the scheduler dies
(`RuntimeError: max(): Expected reduction dim ... numel() == 0`, `dsa_backend.py:1063`). Any engine with fewer
running requests than attention ranks hits it, so real runs would too.

## Decision
Keep a rank with no padded rows IDLE, as non-hybrid models already do. Ranks with padded rows (MAX_LEN padding) keep
the verify layout the hybrid branch exists for. One condition on SGLang's own branch.
- Rejected: guarding the `max()` in the DSA backend. The rest of its verify metadata also assumes at least one row,
  and the IDLE path already joins the collectives correctly.

## How it is applied
`fork/deps/sglang/fix-082-hybrid-spec-idle-rank.patch`, applied by `fork/deps/sglang/apply.py` to SGLang `3a938213c1`:
`model_executor/forward_batch_info.py`, marked `[fork fix-082]`. Verify: `python -m modal run -m
miles_ext.gpu_delta.run_glm_bench --tag <tag> --model /w/gpudelta/GLM-5.3-Flash-fp8-4layer-mtp --draft` (one B300:8; the slice
with the NextN layer comes from `make_glm_fp8_slice.py --mtp`). Without the patch every rank crashed in the first
verify; with it 3 GPU-delta versions applied and the comparison passed with every DP rank exact against the oracle.

## Remove when
SGLANG_COMMIT keeps zero-row idle ranks idle in hybrid speculative verify; upstream status: not filed; the same
branch is on sgl-project/sglang main `045e0e2e2d` (2026-11-04), where this patch applies clean.
