# Upstream candidates

Our bug fixes that upstream Miles ([radixark/miles](https://github.com/radixark/miles))
or SGLang ([sgl-project/sglang](https://github.com/sgl-project/sglang)) could take,
and whether upstream still has each bug. New features, hooks and extensions are
not listed. This page only records candidates: filing them is a separate decision.

Each fix's problem, evidence and remove condition stay in its commit trailers and
patch page (`python fork/tools/ledger.py list`, [patches/](patches/README.md)); a
row here links to them and owns only the upstream judgement: current status,
porting effort and worth. A commit's `Upstream-Status:` trailer is the status on
the day it landed; this page is the current one.

Last checked: 2026-10-21, against Miles main
[`ee36bb2a01`](https://github.com/radixark/miles/tree/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e)
(151 commits past our base `69a147dfc`), SGLang main `4b1a44bc77` and SGLang
`sglang-miles` `203f4bbc70` (the branch Miles builds its image from,
[docs/developer/versions.md](../docs/developer/versions.md)). "Applies clean"
means `git cherry-pick` of our commit onto that Miles main had no conflict.
"Open upstream PR" is an open upstream PR for the same bug: back it instead of
filing our own. Found by `gh pr list --state open --search` on 2026-10-21 and
matched by title; "likely" and "possible" rows need a read of the PR before acting.

**Worth:** High = upstream users hit it on common setups (OOM, crash, lost
work, large waste) and the fix is small. Medium = real, but narrower setups or
a larger change. Low = cosmetic or specific to our stack.

## Keeping it current

- A fix (`Fork-Fix`, or an SGLang patch we carry anywhere) adds its row here in
  the same PR, or the PR says why it is not a candidate (a feature or our-stack only).
- Every upstream sync re-checks the rows for files it touched and moves any
  fix upstream made to "Fixed upstream" with the commit; the sync's conflict
  handling (`.claude/skills/miles-fork/SKILL.md`, Sync) already takes upstream's side there.
- Filing a candidate updates its row's Status with the PR link.

## Miles

| Id | Problem (one line) | Upstream main | Open upstream PR | Port effort | Worth |
| --- | --- | --- | --- | --- | --- |
| [fix-014](patches/fix-014-r3-pad-spread.md) | Routing-replay pad rows all go to experts 0..topk-1, so one expert-parallel rank holds every pad token (25-40 GiB extra, OOM at 256k) | still present, [replay_base.py:104](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/utils/replay_base.py#L104) | [#2697](https://github.com/radixark/miles/pull/2697) "stride the pad-token replay fill across experts" (opened 2026-08-21), related [#2770](https://github.com/radixark/miles/pull/2770): back it rather than file our own | 9 lines + test, applies clean | High |
| [fix-008](https://github.com/listar2000/miles/commit/c476838e01) | FlashMLA sparse prefill pads 16 TP-local heads to 64 for the whole sequence at once (6-8 GiB transient, OOM at 192k+) | still present, [sparse_attention.py:106](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/kernels/attention/dsa/sparse_attention.py#L106) | none found | 23 lines, applies clean | High |
| [fix-011](patches/fix-011-recompute-release.md) | Full recompute keeps each layer's detached input and its grad alive until the next forward (~3.6 GiB per DeepSeek-V4.1 layer at 192k) | still present (Miles side unchanged); may belong in Megatron-LM instead | none found | 40 lines + test, applies clean | High |
| [fix-007](patches/fix-007-dsv41-candidate-block-ids.md) | DeepSeek-V4.1 indexer hands later layers a dense `[q, n_kv]` candidate mask (32 GiB at 256k, plus per-layer carry) | still present, [indexer.py:77](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles_plugins/models/deepseek_v4_1/ops/indexer.py#L77) | possible: [#2793](https://github.com/radixark/miles/pull/2793) "chunk indexer queries on demand", [#3879](https://github.com/radixark/miles/pull/3879) (freezes the indexer) | 44 lines + test, applies clean | High |
| [fix-015](patches/fix-015-bshd-pad-per-micro-batch.md) | bshd pads every micro-batch to the rollout batch's longest sample (mean 139k padded to 210k; ~half of forward/backward is padding) | still present, [rollout.py:92](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/backends/training_utils/data/rollout.py#L92) | [#2698](https://github.com/radixark/miles/pull/2698) "pad bshd samples to their own length, not the rollout-global max": back it | ~75 lines + test, applies clean; touches the DeepSeek-V4 rope cache | High |
| [fix-010](patches/fix-010-nvme-mains-once.md) + [fix-021](patches/fix-021-nvme-lazy-mains.md) | NVMe-streamed optimizer writes every fp32 main from the init params, then again from the load (~10 min of DeepSeek-V4.1 trainer init each) | still present, [nvme_stream.py:590](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles_plugins/optimizers/nvme_stream.py#L590) | none found | ~70 lines + tests, one PR; fix-021 needs fix-010 first | High |
| [fix-044](https://github.com/listar2000/miles/commit/e26c044757) | Raw-mode (mbridge) HF load fails at startup when native LoRA adapters are attached | still present, [checkpoint.py:254](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/backends/megatron_utils/checkpoint.py#L254) | possible: [#4205](https://github.com/radixark/miles/pull/4205), [#3098](https://github.com/radixark/miles/pull/3098) (native LoRA for new models) | 21 lines + test, applies clean | High |
| [fix-017](patches/fix-017-session-collect-timeout.md) | Collecting a finished session's samples has a fixed 120 s bound; a timeout drops the whole group | still present, [openai_endpoint_utils.py:24](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/rollout/generate_utils/openai_endpoint_utils.py#L24) | possible: [#2574](https://github.com/radixark/miles/pull/2574), [#2801](https://github.com/radixark/miles/pull/2801) | 19 lines + test, applies clean | Medium |
| [fix-051](patches/fix-051-r3-wire-int16.md) | R3 routing crosses the samples wire as int32 (165-200 MB replies; a stalled reader loses the try) | still present, [codec.py:43](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/rollout/session/samples/codec.py#L43) | likely: [#4035](https://github.com/radixark/miles/pull/4035) "Decode R3 with SGLang's codec, keep narrow rows" | 22 lines + test, applies clean | Medium |
| [fix-050](patches/fix-050-debug-dump-off-loop.md) | `--save-debug-rollout-data` writes on the rollout event loop (minutes-long stalls at 20 GB per batch) | still present, [rollout_executor.py:219](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/ray/rollout/rollout_executor.py#L219) | none found | 28 lines + tests, applies clean | Medium |
| [fix-046](patches/fix-046-async-checkpoint-buffer.md) | Fully async save pickles every buffered group (tokens, logprobs, routing) on the rollout loop: 50 GB, 22 min blocked, then a keepalive timeout | still present, [fully_async_rollout.py:266](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/rollout/fully_async_rollout.py#L266) | possible: [#2287](https://github.com/radixark/miles/pull/2287), [#2288](https://github.com/radixark/miles/pull/2288) (fully async ownership series) | 32 lines + test; conflicts only on fix-004's flag lines | Medium |
| [fix-048](patches/fix-048-retry-aborted-tries.md) | Fully async mode drops a whole group when one try aborts, discarding its finished tries | still present, [fully_async_rollout.py:192](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/rollout/fully_async_rollout.py#L192) | likely: [#3702](https://github.com/radixark/miles/pull/3702) "keep partial groups on abort"; also [#4167](https://github.com/radixark/miles/pull/4167) | 82 lines + test, opt-in flag; conflicts only on fix-004's flag lines | Medium |
| [fix-004](patches/fix-004-async-single-pass.md) | A fully async eval hands prompts out again after one pass, so fast tasks' repeats can end the eval before slow tasks finish | still present (fully_async_rollout.py unchanged) | none found | ~115 lines + tests, opt-in flag, applies clean | Medium |
| fix-001 ([commit](https://github.com/listar2000/miles/commit/e10a41c109)) | FSDP DCP save/load planning collectives run on NCCL, which fails when colocated SGLang holds the GPU memory | still present, [checkpoint.py:217](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/backends/fsdp_utils/checkpoint.py#L217) | none found (weak: [#3583](https://github.com/radixark/miles/pull/3583)) | 40 lines, applies clean | Medium |
| fix-002 ([commit](https://github.com/listar2000/miles/commit/7db92aa662)) | FSDP resume dies when saved optimizer state lacks params that never got a gradient (Qwen3.5 vision tower) | still present, [checkpoint.py:130](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/backends/fsdp_utils/checkpoint.py#L130) | none found | 12 lines; needs fix-001 first | Medium |
| fix-012 ([commit](https://github.com/listar2000/miles/commit/17664dcd2f)) | Session rollback log has no session id or cause, and nothing reaches the sample | still present, [linear_trajectory.py:348](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/rollout/session/linear_trajectory.py#L348) | possible: [#889](https://github.com/radixark/miles/pull/889), [#3307](https://github.com/radixark/miles/pull/3307) | ~80 lines incl. metrics, applies clean | Low |
| fix-005 ([commit](https://github.com/listar2000/miles/commit/126aad4a20)) | Mismatch diagnostic logs a traceback when DeepSeek-V4.1 text holds the image placeholder SGLang's encoder refuses | still present, [linear_trajectory.py:436](https://github.com/radixark/miles/blob/ee36bb2a015bcceeb34f9fce1d7cd89676747c8e/miles/rollout/session/linear_trajectory.py#L436) | none found | 32 lines + test, applies clean | Low |
| [hook-011](patches/hook-011-native-lora-disaggregated.md) | Native LoRA refuses adapter serving to non-colocated engines, though the broadcast protocol carries adapters | still present, `validate_native_lora_args` in `lora/native/registry.py` on Miles main | none found | 1 condition + test | Medium: needed by every disaggregated native-LoRA run. Goes upstream with hook-012 (the adapter checksum check, `--check-lora-weight-equal`, also accepts PP 1 without PP gathering, one condition in `weight_update/updater.py`) |

## SGLang

| Id | Problem (one line) | Upstream | Open upstream PR | Port effort | Worth |
| --- | --- | --- | --- | --- | --- |
| [fix-053](patches/fix-053-dspark-r3.md) | DSpark speculative decoding drops the verify pass's `routed_experts_output`, `indexer_topk_output` and `kv_weight_version_record` when it builds `GenerationBatchResult`, so overlap scheduling (Miles' default) loses decoded-token routing (DFlash passes these outputs through) | still present on SGLang main `ed268dd32` (2026-10-21) and `sglang-miles` `203f4bbc70`: `TargetVerifyResult` in [dspark_verify.py](https://github.com/sgl-project/sglang/blob/4b1a44bc77c8f473b543468016cecbb6c3fbb794/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L76) has only `logits_output` and `can_run_cuda_graph` | none found on SGLang | 11 added lines ([fix-053-dspark-r3.patch](deps/sglang/fix-053-dspark-r3.patch)), applies unchanged to SGLang main `ed268dd32` | High: R3 with DSpark is silently wrong without it |
| [hook-010](patches/hook-010-dsv4-lora-serving.md) | SGLang has no LoRA path for DeepSeek-V4/V4.1 attention or the shared expert, so a V4.1 LoRA run must merge the adapter into the FP8 weights, where most of the update rounds away | still missing on SGLang main `aa10927271` (no `should_apply_lora` in `models/deepseek_v4.py`) | [sgl-project/sglang#45231](https://github.com/sgl-project/sglang/pull/45231) into `sglang-miles` (we carry it); [#36579](https://github.com/sgl-project/sglang/pull/36579), [#36578](https://github.com/sgl-project/sglang/pull/36578) on main | ours is #45231 placed on `3a938213c1` (the `deepseek_v4.py` and `utils.py` hunks by hand) plus a guard against CP decode attention TP | High: back #45231 |

## Fixed upstream

| Id | Upstream fix | Note |
| --- | --- | --- |
| fix-003 (find the Ray head through the GCS) | [radixark/miles#3640](https://github.com/radixark/miles/pull/3640), commit `951ad8e547`, 2026-10-20: `ray.nodes()` + `HEAD_NODE_RESOURCE_NAME` | same approach; dropped in sync-v0.1.3 (`Retires:`) |
| [fix-016](patches/fix-016-flat-topk-logprobs.md) (flat base64 top-k logprobs) | [radixark/miles#3952](https://github.com/radixark/miles/pull/3952), commit `0cc65167f9`, 2026-10-20 | upstream keeps the nested rows as a fallback where ours raises; dropped in sync-v0.1.3 |
| [fix-055](patches/fix-055-lora-apply-gate.md) (SGLang: `should_apply_lora` gates the LoRA wrap) | [sgl-project/sglang#39798](https://github.com/sgl-project/sglang/pull/39798), on `sglang-miles` `3a938213c1` (v0.1.3's pin) | patch deleted in sync-v0.1.3 |
| fix-056 (SGLang: speculative top-p sampling masks, draft #84, never landed) | [sgl-project/sglang#34201](https://github.com/sgl-project/sglang/pull/34201), on `sglang-miles` `3a938213c1` | draft #84 closes unmerged after sync-v0.1.3 lands |

## Not listed

- fix-006 is a port of upstream's own branch `jiajun/mimo26-mxfp4-qat` (head
  `ddb2b211c8`, still unmerged into main): nothing to send; drop it when main has it.
- fix-049 only serves ext-034 (our group log).
- Features carried as fixes: fix-022 (trainer init alongside the engines),
  fix-045 (optimizer host-memory tier), fix-047 (recompute reuses the indexer top-k).
- Hooks (hook-001, 002, 005, 006, 007, 008) exist for our extensions.
