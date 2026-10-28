# ext-001 (task hash): Harbor's task-directory hashes computed once per task, in a worker thread

## Why
Harbor hashes a trial's whole task directory twice, synchronously, on the caller's event loop.
`Trial.__init__` writes the trial lock, whose digest is `Packager.compute_content_hash`, and the
trial result records `Task.checksum` (`dirhash`). In Miles every sample's trial runs on the rollout
manager's one event loop, so while one task is hashed no other trial starts or advances.

The filtered TB4 task set holds up to 4.5 GB per task (21 GB in all) on a read-only Modal Volume
read at about 20 MB/s. The first evaluation runs on it (one node, TP8 engine, 128 concurrent agents) started 11 trials
in 20 minutes and sent the engines no request. py-spy showed the loop in `compute_content_hash`.
A CPU probe (4 large tasks created at once, 50 ms ticker on the loop) measured one 70.2 s stall
lasting the whole wall time.

## Decision
- `warm(task_dir)` computes both hashes in a worker thread (`asyncio.to_thread`) before
  `Trial.create`. `Packager.compute_content_hash` and `Task.checksum` are replaced on their classes
  by lookups that return the stored value, so the trial's own calls are dict hits on the loop. The
  values are Harbor's own: the replacements call the upstream functions, and a directory `warm`
  never saw falls through to them.
- A task's attempts share one computation (a per-directory lock). Each `warm` re-checks every
  file's relative path, size and mtime, in its thread, and hashes again when anything changed.
- `prewarm(task_dirs)` hashes the whole prompt set at rollout setup, before the first weight
  sync, on 4 threads of its own, in the order the data source hands prompts out. It runs from
  `agent_function.prepare` (hook-005), in a daemon thread that first imports what the first
  `Trial.create` imports (Harbor's trial classes, the harness, the `HARBOR_ENV_IMPORT_PATH`
  provider). Without it the first wave does both at once: the hash threads hold the GIL while the
  loop imports, and the loop also builds 128 trial configs back to back. A trial whose dir is
  still being prewarmed waits on that dir's lock, so a set that outlasts the setup is still hashed
  once. `build_trial_config` reads swebench.yaml and the mini-swe-agent version once, not per trial.
- Rejected: a `listar2000/harbor` fork (fork/README.md "Harbor"). The fix changes no Harbor
  code path, only where and how often it runs, and a fork would add a second sync to every
  upstream release. Rejected: creating the `Trial` in a thread. The trial result still reads
  `Task.checksum` on the loop later, during `run`.
  Rejected: skipping the hashes. The lock digest and checksum are part of Harbor's result.

## How it is applied
- `miles_ext/harbor/task_hash.py` patches the two Harbor attributes at import. `lock.py` and
  `trial.py` look them up on the class at call time.
- `miles_ext/harbor/agent_function.py` `run()` imports it and awaits
  `asyncio.to_thread(task_hash.warm, config.task.path)` before `Trial.create`.
- Verify: `python -m pytest miles_ext/tests/test_task_hash.py`. The run test fails if a hash is
  computed on the loop thread.
- Measured after the change: 4 large tasks, max loop stall 3.8 s and 21.4 s wall; 8 tasks, max
  stall 4.3 s and 26.7 s wall. An evaluation at 128 agents on one 8-GPU node had 156 trials
  started about 30 minutes after the first.
- Prewarm, CPU probe with the trial config of overfit-1018-a2 (the 8-task
  DeepSeek-V4.1 overfitting run of 2026-10-18) (mini-swe-agent-sandbox harness, Sail environment
  constructed but not started; 16 groups x 8 tries of that run's 8 tasks, 10 ms ticker;
  a CPU-only Modal probe, not in this repo): without prepare, the loop
  was blocked once for 16.9 s and the first trial started at 16.9 s (the first `Trial.create`
  11.5 s, 128 config builds 5.2 s). With prepare (12.1 s, before the wave): max loop stall 1.1 s,
  first trial at 0.7 s, all 128 by 6.2 s. That run's stall was 7 min 43 s on its head node; the
  probe reproduces its shape (one block, then every trial at once), not its length.
- On a Harbor sync: check that `Packager.compute_content_hash` and the `Task.checksum` property
  still exist with the same signatures, and that the lock and result still read them from the class.
- Known cost: hashing the 21 GB task set takes about 30 minutes. It reads each directory twice.
  With prewarm it starts at setup, so a set that hashes faster than the engines load is free;
  a larger one still overlaps the first wave, where trial warms of not-yet-hashed dirs wait on
  their locks and the hash threads share the GIL with the loop. A child process for the hashing
  would remove that (follow-up, not needed for an 8-task set).

## Since 2026-10-28: one hash per node, not per process

The memo is per process, so the prewarm in the rollout process never reached a trial process (ext-033,
ext-042): every trial hashed its task again. `task_hash.warm` now keeps the values in a file cache,
`$TMPDIR/miles-task-hash/<sha256 of the dir>.json`, keyed by the directory and its stat signature. An
`flock` on the entry makes the first process compute and the rest wait and read it. Listing a dir for its
signature is itself up to 35 CPU-s for a large TB4 task, so an entry listed in the last `FRESH_S` (300 s) is read
without listing. A task dir edited in place mid-run is noticed up to 5 min late. CPU probe of 512 forked
trials with prewarm: with every trial listing its dir, trials took up to 35 CPU-s and
the last started at 53 s on 64 cores; with fresh entries 2 CPU-s at most and 28-30 s, the same as hashing off.
Without prewarm the last start went from 151 s to 53 s on 32 cores.
Check: `test_fresh_file_cache_avoids_rehashing_with_an_empty_memo` in `miles_ext/tests/test_task_hash.py`.

## Remove when
Harbor computes these hashes off the event loop or caches them per task; upstream status: not filed.
