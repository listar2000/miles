# fix-014: spread routing-replay padding rows over all experts

## Why
With `--use-rollout-routing-replay`, `fill_replay_data` pads each sample's replayed routes with -1 rows up to the
microbatch length, and `_get_replay_result` turns every all-(-1) row into `arange(topk)`. Every padding token
therefore goes to experts 0..topk-1, and with expert parallelism those experts all sit on expert-parallel rank 0.
With `--qkv-format bshd` every sample is padded to the longest sample in the rollout batch, so a batch of mixed
lengths sends a large share of all tokens to one GPU. Evidence: DeepSeek-V4.1-Flash RL run on 2 nodes of 8 B300
(TP4 PP2 CP2 EP8, 384 experts, top-6), batch mean ~135k tokens padded to ~256k: local GPU 0 of each node ran
25-40 GiB above its seven peers through every forward/backward, peaked at 266.2 of 267.7 GiB in the first step and
ran out of memory in the second, asking 7.12 GiB in the MoE combine (`sort_chunks_by_idxs`). That request is ~746k
rows x 5120 x 2 bytes, 3.8x the ~197k rows an even split gives.

## Decision
Offset each filled row by `row * topk`, modulo the expert count, so pad rows go round-robin over all experts.
Real-token routes are untouched. Padding sits after the last real token and is masked from the loss, so loss and
gradients are unchanged; padding now also stops inflating the expert-bias load counts of experts 0..topk-1.
Alternatives that lost:
- Drop pad rows from dispatch: Megatron only honours `padding_mask` in the dispatcher with HybridEP, not alltoall.
- Pad each sample to its own length instead of the batch maximum: removes the padding compute too, but changes
  `max_seq_lens` for every consumer (batching, loss CP slicing, replay); a separate change.
- Use the router's own top-k for pad rows: identical pad tokens can still pile onto a few experts.

## How it is applied
`_get_replay_result` in `miles/utils/replay_base.py`; on by default, `MILES_R3_PAD_SPREAD=0` restores the old fill.
Verify: `tests/fast/utils/test_replay_base.py` (pad rows within one row per rank of an even split over 8
expert-parallel ranks, real rows bit-identical, the switch restores arange). On a sync, check the fill is still in
`_get_replay_result`.

## Remove when
Upstream spreads or drops replay padding itself. Status 2026-10-18: upstream fills pads with arange.
