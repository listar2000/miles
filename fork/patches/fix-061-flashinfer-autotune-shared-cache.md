# fix-061: one FlashInfer autotune cache file per TP group, written by TP rank 0

## Why
SGLang keeps the FlashInfer tactics it tunes at engine start in one file per TP rank
(`rank_tp<N>_pp<P>_dp<D>.json`) and, before tuning, discards all of them if the ranks' files differ
("per-rank caches disagree"), because a cache hit on one rank but not another desyncs the timing all-reduce.
Our engine nodes run two TP4 engines that share one cache directory (and one compile-cache tarball), so engine A's
TP0 and engine B's TP0 write the same file, and each file ends up holding whichever engine wrote last. The two
engines tune independently, so the four files of a TP group come from different engines, disagree, and the next
start discards them and tunes from scratch. On DeepSeek-V4.1-Flash with DSpark the draft CUDA-graph step then took
117-140 s instead of 3-4 s: 4 of 9 V4.1 launches from 10-20 to 10-27 (two overfit runs, a DSpark smoke and a split
smoke), about 2 min per launch it hits.

## Decision
- The TP ranks all-reduce their timings before choosing (`_autotune_tactic_sync_group`), so one TP group tunes one
  set of tactics. Keep one file per TP group (`rank_tpall_pp<P>_dp<D>.json`), loaded by every rank and written only
  by TP rank 0, through a temp file and a rename so another engine never reads half a file. Two engines sharing the
  directory now overwrite one whole file, and every rank of the next engine loads the same one. Each rank copies
  the file once and runs both the digest check and the load on that copy, so a rename by the other engine in
  between cannot give the ranks different tactics.
- The write copies SGLang's own pattern for a cache file other processes read: `HiCacheFile.set`
  (`mem_cache/hicache_storage.py` lines 546-548 at `3a938213c1`) writes `.{uuid}.tmp` and `os.replace`s it into
  place; the per-rank snapshot uses the same uuid-named private file.
- MegaMoE keeps its per-rank, per-GPU files (its records need one writer per file).
- Lost: giving each engine its own cache dir (doubles the cache and still lets a TP group's four files come from
  different starts); keeping per-rank files but writing them all from rank 0 (same effect, more files).

## How it is applied
`fork/deps/sglang/fix-061-flashinfer-autotune-shared-cache.patch`, applied by `fork/deps/sglang/apply.py` to SGLang
`3a938213c1`: `model_executor/runner/flashinfer_autotune.py`, every spot marked `[fork fix-061]`. The first start
on a new image tunes once and writes the shared file (the old per-rank files are ignored). Verify: an engine start
that restores the compile cache logs no "per-rank caches disagree" and its draft decode CUDA graph takes seconds.
Measured 10-28 on two split DeepSeek-V4.1 LoRA smokes on one new image: the first start tuned once (draft decode
graph 81 s; one engine logged "per-rank caches disagree" while the shared file was first written), the second start
logged none and took 3.8 s.
On a sync: SGLang main has moved the context lines around the save, so re-apply the same three edits.

## Remove when
SGLang stops keeping per-rank tactic files for TP-reduced tuning. Upstream status: not filed; still per-rank on
SGLang main `8136cbce3d` (the patch context has moved).
