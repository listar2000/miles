# fix-054: bound a group's token-weighted staleness as well as its oldest token

## Why

`--max-weight-staleness` measures a group by its oldest generated token. Long agent trajectories can span weight
updates while continuing to generate on newer weights, so a strict oldest-token cap drops them for their first
turn alone and can underrepresent long tasks. This change lets callers combine a relaxed oldest-token cap with a
bound on the mean lag of a group's recorded generated tokens.

## Decision

Two bounds, both must pass: `--max-weight-staleness` (oldest token, unchanged) and a new
`--max-token-weighted-staleness` (current version minus the token-weighted mean version, the quantity behind the
existing `token_weighted_staleness` metric). The oldest bound keeps a hard ceiling on the worst token; the mean
bound sets the typical freshness. The scheduler illustration in [PR #76](https://github.com/listar2000/miles/pull/76) uses the version at a group's
wall-clock midpoint as a proxy for its token-weighted mean; its projected waits and task shares do not measure the
predicate implemented here.

Rejected: a pluggable staleness predicate (a hook plus an extension): more code for one rule. Early stopping of
doomed groups: cuts waste, not the bias.

Caveat: under `--pause-generation-mode in_place` the engines keep their KV cache across a push
(`weight_update/session.py`), so a later turn's tokens carry the new version but attend over context encoded with
older weights. The mean bound is therefore somewhat optimistic; the oldest bound limits how far.

## How it is applied

`group_stale_reason` in `miles/rollout/filter_hub/common_filters.py` applies both bounds; the default data buffer
(`fully_async_data_buffer.py`) uses it in `num_ready` and `_drop_stale`. Dropped groups go to
`--async-unused-samples-handler` with `UnusedReason.STALE`, as before. New metric
`rollout/fully_async/token_weighted_stale_groups_filtered` counts the groups the mean bound dropped (also counted
in `stale_groups_filtered`). Recipes set it with `algorithm.max_token_weighted_staleness` (default none). Miles and the recipe reject a
NaN, negative or non-fully-async setting at startup. Off, the
buffer behaves exactly as upstream. On a sync, keep both call sites on the shared predicate.

## Remove when

Upstream Miles bounds staleness per token, or the fork stops using fully async mode.
