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
- On a Harbor sync: check that `Packager.compute_content_hash` and the `Task.checksum` property
  still exist with the same signatures, and that the lock and result still read them from the class.
- Known cost: the first wave over the 21 GB task set still takes about 30 minutes per launch. It
  reads each directory twice, and at 128 agents it fills the default thread pool (32 workers).
  The event loop stays free.

## Remove when
Harbor computes these hashes off the event loop or caches them per task; upstream status: not filed.
