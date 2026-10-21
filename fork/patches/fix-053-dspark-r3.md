# fix-053: DSpark speculative decoding passes the verify pass's R3 routing through

## Why
Routing replay (R3) records, for every token the engine produces, which experts it was routed to, so the
trainer can replay the same routing. With `--speculative-algorithm DSPARK` the decoded tokens come from
DSpark's verify forward, and DSpark dropped that forward's routing: `TargetVerifyResult` held only the logits and
the CUDA-graph flag, and `dspark_worker_v2.py` built `GenerationBatchResult` without `routed_experts_output`,
`indexer_topk_output` or `kv_weight_version_record`. Every decoded token got an all-zero routing row. On
DeepSeek-V4.1-Flash (2xTP4, B300, replayed agent traffic) 1046 of 1052 and 1326 of 1436 measured requests had
zero rows (itr-1021-r1). DSpark is worth having: +42% decode speed per agent at 128 sessions per node, judged
against the non-speculative engines (itr-1021-r3).

## Decision
- Mirror DFlash. `dflash_worker_v2.py` (line 2897 at the pinned `7b6993a`) passes the three fields from the
  target forward into `GenerationBatchResult`; DSpark now does the same through `TargetVerifyResult`, which gains
  the three fields with default `None`. Nothing reads them but the existing result processing, so sampling,
  acceptance and attention are untouched.
- An SGLang patch file (rule: fork/README.md, "SGLang patches"), 11 added lines. The applier the overnight
  research used (`investigations/inference-throughput/patches/dspark_r3.py`, edits at image build) stays an
  investigation tool and never lands.

## How it is applied
`fork/deps/sglang/fix-053-dspark-r3.patch`, applied by `fork/deps/sglang/apply.py` to SGLang `7b6993a`:
`dspark_components/dspark_verify.py` (the struct and where it is built) and
`dspark_components/dspark_worker_v2.py` (the `GenerationBatchResult`), each with a `[fork fix-053]` comment.

Evidence, same bench and image, the patch the only difference:

| | DSpark, unpatched (r1) | DSpark + fix-053 (r3) |
| --- | --- | --- |
| requests with zero R3 rows | 1046/1052, 1326/1436 | **0** at every level |
| decode vs prefill routing overlap | n/a | **0.948** (min 0.935, 4088 rows) |
| per-agent tok/s, 128 sessions/node | 78.4 | 78.5 |

The engine-vs-trainer logprob gap held in Miles with DSpark and LoRA weight syncs: 0.0359, 0.0358, 0.0354 over
three steps (smoke-dspark-1021-a1, the a6 recipe plus `--sglang-speculative-algorithm DSPARK`).

## Remove when
SGLang fixes DSpark's verify result. Upstream status, checked 2026-10-21: not filed; the bug is also on
sgl-project/sglang main `ed268dd32`, and the patch applies there unchanged. Listed in fork/UPSTREAM.md.
