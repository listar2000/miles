# ext-034: write-ahead group log for the fully async buffer

## Why

The default fully async buffer is checkpointed by pickling every buffered group into one `state.pt`, inside the
rollout save, which never awaits by design. With DeepSeek V4.1 each try carries its R3 expert routing
(int32, tokens x 43 layers x 6, about 175 MB at 170k tokens), so a 16-try group is about 2.8 GB. A 3-node
overfit run's save grew to about 50 GB, wrote at about 40 MB/s, and froze the rollout loop for 22 min until Ray
declared the executor dead. fix-046 (`--async-checkpoint-buffer prompts`) avoids the crash by saving only
prompts, but a resume then regenerates the whole buffer (about 75 min at that size) while the trainer waits.

## Decision

Plan approved 2026-10-20 (all five picks): write each group once, as it enters the buffer, and have a save
record only which groups are buffered.

- **On put**, after the default filters, a writer task hands the group to one of two writer threads, which
  write it to `<save>/rollout_groups/<uid>.safetensors`; `put` returns at once, so the producer never waits
  for storage (a group being written counts toward the buffer's capacity). With `--async-single-pass`, `put`
  waits for the write, so the pass cannot end before its last group is ready. The entry becomes visible to `get`
  only once its file is written, so nothing reads a group the trainer side may change, and a save never waits
  for a write. The CPU probe wrote a group in about 2 s, with the event loop's longest pause at 19 ms; each
  write is logged (`group log: wrote group`) with its size and time.
- **On save**, `state_dict` returns a manifest: each entry's uid and a reset copy of its prompts. A group still
  being written is listed with its prompts only.
- **On load**, the named files are read in parallel and the buffer is restored as it was. A group without a file
  (missing, failed write, or still being written at the save) is regenerated from its prompts, also under
  `--async-unused-samples-handler drop` (fix-049: it was never trained on). Resuming from another run's save
  copies the files into this run's `rollout_groups/`.
- **Format.** Per-token fields (the sample codec's tensor fields: tokens, loss mask, logprobs, top-k, R3
  routing, indexer top-k) are stored as arrays with the codec's dtypes; the rest of each sample is pickled.
  R3 routing is stored as uint16 when every id fits (384 experts in V4.1), halving the file, and read back as
  int32.
- **Cleanup.** After each save the writer deletes group files that no live entry and no published manifest
  (`<save>/rollout/*/generate_rollout/state.pt`) names. Storage is bounded once old `rollout/<id>` folders go;
  the platform's checkpoint retention removes them.
- Not chosen: the same pickle in a thread (pickling holds the GIL; the loop still stalls), Modal Queue or Dict
  (1 MiB items, 100 MiB values, not durable), recomputing routing on resume (changes the routing the rollout
  logprobs were sampled with).

## How it is applied

- `miles_ext/rollout/group_log.py`: `GroupLogDataBuffer`, a `DefaultDataBuffer` subclass. `put` repeats the
  default capacity wait and append around the write; the filters, `get`, staleness and metrics are inherited.
  It requires `--async-checkpoint-buffer log` (hook-008).
- `miles_ext/recipes/recipe.py`: `async.checkpoint_buffer` (default `log`) passes `--async-checkpoint-buffer log
  --custom-async-data-buffer-path miles_ext.rollout.group_log.GroupLogDataBuffer`; `prompts` keeps fix-046 as a
  fallback. Both flags are reserved.
- Single policy only: the buffer refuses a multi-policy run (`DefaultMultiDataBuffer` saves nothing either).
- Verify: `python -m pytest miles_ext/tests/test_group_log.py`. In a run, look for `group log: restored` after
  a resume and `group log: deleted` after each save.
- On a sync: if upstream changes `DefaultDataBuffer.put` (capacity wait and append) or the codec's
  `SAMPLES_VALUE_SPEC`, re-read `put` and `_ARRAY_FIELDS`.

## Remove when

Upstream Miles checkpoints buffered groups without one blocking write, or stops buffering finished groups across
saves. fix-046's `prompts` mode stays as a fallback (`async.checkpoint_buffer: prompts`).

## Since 2026-10-27

The recipe passes `--custom-async-data-buffer-path` for every fully async run; `async.checkpoint_buffer`,
`--async-checkpoint-buffer` and the `prompts` fallback are gone (fix-060).
