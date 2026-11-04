# ext-056: recipe settings for which saves carry Adam's state and which are kept, and pruning while training

## Why
Modal containers live at most 24 h and GPU functions can be preempted, so long runs need regular saves, and RL runs
need older steps to backtrack to. On full-parameter GLM-5.3-Flash a weights-only save is ~627 GB and Adam's state adds
~2.6 TB (two B300:8 nodes, 2026-11-04: 3,410 GB in 2,608 s, training blocked throughout). The recipe could only save
Adam's state in every save or in none, and the launcher pruned saves only when a run ended.

## Decision
These are settings, not a save strategy: every save still goes straight to `--save`. How a save reaches the volume
(local staging, background writes) is separate work that builds on hook-020.
- `checkpoint.optimizer_every` (k): with `save_optimizer`, Adam's state goes in every k-th periodic save and the save
  gets `iter_N/.optimizer`. A save off the interval (the final one, one the supervisor requests before a deadline) holds
  weights only. A resume from a save without the marker passes `--no-load-optim`, so Adam's moments restart.
- `checkpoint.keep_every` (m): retention also keeps the save after every m-th rollout, counted like the save interval.
  It must be a multiple of `save_interval`, or it would match no save.
- With launcher-side retention on, retention also runs when the tracker moves. It keeps the newest `keep` saves, the
  tracker's save, the `keep_every` saves and, while training, the newest save with Adam's state.

## How it is applied
`miles_ext/checkpoint/save_policy.py` (`context` for hook-020, `with_optimizer`, `has_optimizer`), the recipe keys in
`miles_ext/recipes/recipe.py`, `_retention_while_training` and the resume check in `miles_ext/launch/modal_app.py`, and
`keep_every` in `miles_ext/platform/checkpoints.py`. Verify: `retention (tracker N): removed [...]` in the launcher log,
`iter_N/.optimizer` on the saves that carry Adam's state. Tests: `miles_ext/tests/test_save_policy.py`.

Limit: the marker exists only under `optimizer_every` > 1, so do not change that setting between segments of one run.

## Rerunning the probes
- `modal volume create --version=2 miles-ckpt-probe` once; the probes write fake data only there.
- `CKPT_PROBE_GPU=B300:1 modal run fork/probes/ckpt/ckpt_stream_probe.py`: N parallel files to the volume (raw,
  torch.save, multi-thread pwrite), node-local disk, local-to-volume copies. `CKPT_PROBE_GIB`, `CKPT_PROBE_STREAMS`
  and `CKPT_PROBE_VOL_ONLY=1` (vol-N cases only, with a 30 s rate ticker) size a long sustained write.
- `CKPT_PROBE_GPU=B300:8 CKPT_PROBE_MEMORY_GIB=2560 CKPT_PROBE_IMAGE=starlimobius/miles-mobius:main modal run
  fork/probes/ckpt/ckpt_async_probe.py --only sync-t1-78`: a real-size Megatron sync save; the CASE line times
  the host preload apart.

## Remove when
Upstream Miles has per-save optimizer selection and in-run retention; upstream status: not filed.
