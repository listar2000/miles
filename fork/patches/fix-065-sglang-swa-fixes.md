# fix-065 to fix-069: five SGLang main commits for V4.1 SWA and prefix caching, carried on our pin

## Why
Our SGLang is `sglang-miles` `3a938213c1`. SGLang main has since fixed four bugs in the
sliding-window (SWA) KV and prefix-cache code DeepSeek-V4.1 uses, and removed a per-layer host
sync in V4.1 prefill. Long agent prompts that share prefixes hit exactly this code. None of the
five commits is on `sglang-miles` (head `5403606db9`, 2026-10-30), and upstream Miles takes them
only when it re-ports `sglang-miles` onto SGLang main (started 2026-10-29).

| Id | SGLang commit | What it fixes |
| --- | --- | --- |
| fix-065 | `d4daf5236` (#44347) | Prefill CUDA graph tail: MoE routing reads the GPU non-padded counts, now clamped to the tail's live rows |
| fix-066 | `47eeddcd1` (#43168) | Prefix-sharing SWA eviction keeps max(window, page) rounded up to whole pages, so a cached key keeps a full window |
| fix-067 | `8b1d1e0d7` (#45223) | Request-window history check is a device-side invariant instead of a `torch.equal` host sync per layer |
| fix-068 | `99cf403c9` (#45586) | Idle forwards write SWA through their own idle layout, not stale or unset metadata |
| fix-069 | `e7c55f695` (#46700) | SWA tombstone recovery never starts below the request's protected prefix |

## Decision
Carry each commit as its own patch, byte for byte except one `[fork fix-NNN]` comment per file and
without its tests. They mirror SGLang's own code, so they are the fix SGLang chose, not ours.
Rejected: waiting for the `sglang-miles` re-port (no date; these are correctness fixes on our hot
path). Rejected: a `listar2000/sglang` branch now (fork/README.md suggests one past about five
live patches; this makes nine, but all five of these drop by themselves at the next SGLang move,
so the count falls back to four). Revisit if the re-port slips past the v0.1.4 sync.

## How it is applied
`fork/deps/sglang/fix-065-*.patch` to `fix-069-*.patch`. fix-065 and fix-068 both edit
`deepseek_v4_backend.py`, fix-067 and fix-068 both edit `dsv41_request_window.py`: each patch is
the diff on top of the lower-numbered ones, so `apply.py` (one `git apply`, files in name order)
must keep them in id order. Check: `python fork/deps/sglang/apply.py <sglang at 3a938213c1>
3a938213c1 --check`. On a sync whose `SGLANG_COMMIT` contains a commit, delete its patch and mark
its row below removed.

Evidence: one B300 node, DeepSeek-V4.1-Flash as two engines with four-way tensor and expert
parallelism and routing replay (R3), at 64 and 128 concurrent agent sessions. Baseline image
`starlimobius/miles-mobius:main`; candidate `starlimobius/miles:pr-sglswa-7f1a89b0c6`, whose
`miles.mobius.sglang-patches` label lists fix-065 to fix-069. Both: no request errors, no empty
R3 rows, logprob gap 0.042, R3 overlap 0.95. At 64/128 sessions, baseline vs candidate:
prefix-cache hit 0.92/0.90 vs 0.92/0.91; median prefill time per new token 0.056/0.062 vs
0.052/0.058 ms (6 to 7% lower); decode throughput 4907/5620 vs 4729/5678 tok/s (noise).

## Remove when
`SGLANG_COMMIT` contains the SGLang commit in the table; upstream status: all five merged in
sgl-project/sglang main `80afdc128` (2026-10-29), none on `sglang-miles` `5403606db9`.
