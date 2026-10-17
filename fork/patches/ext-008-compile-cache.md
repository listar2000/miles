# ext-008: compiled-kernel cache kept on a Volume between runs

## Why
Every container starts with an empty `SGLANG_CACHE_DIR` (`/root/.cache/sglang`), so the
engine re-tunes and recompiles every kernel at every start. Measured on DeepSeek-V4.1-Flash,
1 x B300:8 (TP8/EP8, FP8 dense + FP4 experts, SGLang 0.5.22.dev46+g7b6993a), from
`sglang.launch_server` to ready, without Ray or Miles:

| | weight load | FlashInfer autotune | to ready |
| --- | --- | --- | --- |
| cold (empty cache) | 176 s | 510 s (2 passes) | 965 s (16.1 min) |
| warm (cache restored) | 98 s | 15 s | 237 s (4.0 min) |

Run 4's launch shows the same cold profile inside Miles (autotune 15:34-15:40 UTC). The
cache is ~1 GB in ~1,500 files: CUDA's compute cache (`nv`, 930 MB), Triton, TileLang,
DeepGEMM, FlashInfer JIT and the FlashInfer autotune results.

## Decision
- **Stage, don't mount.** Every node unpacks all of `<volume>/<gpu>/sglang-<version>/node-<rank>.tar`
  into the local cache before Ray starts, oldest save first. Each node packs its own local cache
  back into `node-<rank>.tar` when it has changed and no file in it was written in the last
  60 s. It checks every 5 min and saves once more on exit. Kernels compile and take file
  locks on local disk only.
  - One archive per node because rollout engines can be placed on any node; saving only
    rank 0's cache would drop the tuning of engines on the other nodes.
  - Pointing `SGLANG_CACHE_DIR` at the mounted Volume was rejected for three reasons. DeepGEMM
    and inductor take file locks there. Writes reach other containers only after a commit. And
    on a multi-node run several nodes would write the same files.
- **One tar file, not a tree.** Copying the 1,500 files one by one off a Volume took 126 s.
  One archive is read sequentially. Each save is a whole snapshot, so FlashInfer's per-rank
  tuning files never come back half old and half new (archives are unpacked whole, one
  after another, so a later node's set replaces an earlier one's): SGLang discards disagreeing rank files
  and retunes (`_drop_diverged_autotune_cache`).
- **Key = GPU type + SGLang version.** Inside it each cache keys its own entries, so runs of
  different models share one archive. FlashInfer keys by model path, dtype, quantization, MoE
  backend and TP/PP/DP/EP, but not by mem fraction or context length. A new SGLang starts a
  new archive rather than trusting old binaries.
- **Volume name derived from the target**, like the completion queue: `miles-compile-cache`
  by hand, `rl-<env>-compile-cache` for option-A targets. It is a v2 volume, shared by every
  run of the environment and never deleted by teardown.
- **Env for the whole run:** `SGLANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `TILELANG_CACHE_DIR` and `CUDA_CACHE_PATH` under the local cache. The trainer's Triton,
  inductor and CUDA kernels are kept too: SGLang redirects those itself only inside its own processes.
- A failed restore or save is logged and never fails a run; the fallback is a cold start.
- **`SGLANG_DG_CACHE_DIR_PER_PROCESS=0` for the whole run.** Miles' engine env
  (`compute_inference_engine_env_vars`, `miles/ray/specs/inference.py`) defaults it to `1`
  unless the environment sets it, and the pinned SGLang then writes DeepGEMM kernels under
  `pid_<pid>`, which no later launch reads. SGLang reads only `1`/`true`/`yes` as on
  (`srt/layers/deep_gemm_wrapper/compile_utils.py`).

Recipes must not override these directories. Upstream's `scripts/run_deepseek_v4_1.py`
sets `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and `TILELANG_CACHE_DIR` to `/tmp`, and
`SGLANG_DG_CACHE_DIR_PER_PROCESS=1`, which puts DeepGEMM's cache under `pid_<pid>`. A run
with any of those set gets no reuse for those caches. Leave them out of fork recipes: the
per-process DeepGEMM directory is for colocated engines sharing a host, and DeepGEMM
already serializes its ranks with its own `locks/` directory.

## Measured, and what did not lead to code
DeepSeek-V4.1-Flash (FP8 + FP4), 1 x B300:8, TP8/EP8, engine alone, one run each:
cold 16.1 min to ready (weight load 176 s, FlashInfer autotune 510 s), warm 4.0 min
(98 s, 15 s), warm with `--weight-loader-prefetch-checkpoints` 5.3 min (109 s, 23 s).
- The prefetch flag gives nothing on one node: the load already runs at the Volume's
  limit, about 2.9 GB/s for 510 GB.
- The pinned SGLang has no `instanttensor` load format.
- GCS from a Modal container reads at 0.15 to 0.24 GB/s, for raw ranged GETs and for
  SGLang's `runai_streamer` at TP1 and TP4 with 16 or 64 streams, so checkpoints stay on Volumes.
- Volume v2 reads 61 GB at 2.95 GB/s against 2.1 to 2.5 GB/s for v1, and writes plus
  commits about twice as fast.

What is not measured yet: the timings above are plain `sglang.launch_server`, not an engine
launched by Miles. The launcher smokes (Qwen3.5-9B, 1 x H200:8) show the archive restored and
saved around a real run, but that model does not use DeepGEMM and ran before the DeepGEMM
setting above, so DeepGEMM reuse and the warm start through Miles' engine launch are first
seen on the second DeepSeek launch.

## How it is applied
`miles_ext/platform/compile_cache.py` (restore, save, `Saver`, env).
`miles_ext/launch/modal_app.py` mounts the volume at `/compile-cache` on `train`. Every rank
restores before Ray starts, `_head` and `_worker` each run a `Saver` for their node, and `_ray_env` adds the env.
`miles_ext/platform/target.py` has `compile_cache_volume_name`.
`miles_ext/platform/teardown.py`'s `resource_names` lists the volume. Tests:
`miles_ext/tests/test_compile_cache.py`.

To verify, a second launch of the same recipe logs `compile cache: restored ... MB`. On a
DeepSeek engine, `FlashInfer autotune completed` should come seconds after `Running
FlashInfer autotune`, not minutes. FlashInfer's key includes the model path, so a cache
tuned with the weights at another path retunes once; the next launch is the warm one.

On a sync, recheck `sglang/srt/environ.py` (`SGLANG_CACHE_DIR`,
`redirect_third_party_caches`) and `flashinfer_autotune_cache_path`. A new cache that SGLang
keeps outside `SGLANG_CACHE_DIR` is not covered.

To clear the cache, delete the archives for that key from the volume. Archives are never
pruned: each node saves what it restored as well, so every `node-<rank>.tar` grows toward the
union of all caches under the key, and a `node-<rank>.tar` left by an earlier, wider run stays
until it is deleted. Restore unpacks the newest save last, so its copy of a file wins.

## Remove when
Modal keeps compiled kernels across containers by itself, or the engine stops tuning at
start. Not an upstream change. Upstream status: not filed.
