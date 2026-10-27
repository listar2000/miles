# fix-059: fork fixes have no off switch

## Why
Six fork fixes could be turned off with a flag or an env var, and turning one off brought its bug back. No recipe
or host run used the off value (flag prune plan, user decision 2026-10-27).

## Decision
- Always on: fix-011 (`MILES_RECOMPUTE_RELEASE`), fix-014 (`MILES_R3_PAD_SPREAD`), fix-015
  (`MILES_BSHD_PAD_PER_MICRO_BATCH`), fix-022 (`--trainer-init-with-rollout`, for non-colocated fully async runs
  with `--num-rollout`), fix-047 and fix-052 (`--dsv4-reuse-indexer-topk`, `--dsv4-indexer-tp-split`, on unless
  indexer replay is on).
- Constants: fix-008's FlashMLA row chunk (16384, was `MILES_FLASH_MLA_ROW_CHUNK`) and fix-017's 1,800 s collect
  timeout (was `--session-collect-timeout`).
- fix-048's `--async-retry-aborted-tries` stays, default 1: upstream's fully async tests run with 0, and a group
  reward or deterministic inference forces 0, because neither can retry one try alone.
- Rejected: compatibility shims for the removed names. A run from before reproduces from its image and
  `recipe-*.json` (the last pre-prune build is `main` at 7cc9e4706f).

## How it is applied
Each spot is marked `[fork fix-059]` next to the fix it changes. The tests of the off values are gone. The commit
carries `Releases:` for the files a flag left (`ledger.py`).

## Remove when
Never: it only removes switches from our own fixes.
