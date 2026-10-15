"""Multi-node commits: the tracker becomes durable only after every node's shards are.

On a multi-node Megatron run each node writes its own checkpoint shards to its view
of the results volume, and a v2 volume keeps only what that node commits. Rank 0
commits as soon as the tracker changes (supervisor), so before that commit it asks
every worker node to commit and waits for their acknowledgements:

- rank 0 writes ``commit:<run_id>`` in the progress Dict as ``{token, ts}`` (a fresh
  token per request, so a request or acknowledgement left by an earlier segment
  never matches);
- each worker polls that key every ``POLL_SECS``; on a new token it commits and puts
  ``(rank, token)`` in partition ``ack-<run_id>`` of the completion queue;
- rank 0 waits for ``nodes - 1`` acknowledgements of its token (``ACK_TIMEOUT_SECS``
  at most: a dead worker cannot hold the run, its shards are lost either way), then
  commits itself.

Workers also commit on a timer, and every rank commits on exit; this only closes the
window in which a killed cluster leaves a durable tracker naming an incomplete save.
"""

import threading
import time
import uuid
from typing import Callable

POLL_SECS = 5
ACK_TIMEOUT_SECS = 300


def ack_partition(run_id: str) -> str:
    return f"ack-{run_id}"


def head_commit(run_id: str, nodes: int, progress_dict, queue, commit: Callable[[], None], *,
                key: str, timeout: float = ACK_TIMEOUT_SECS, log=print) -> Callable[[], None]:
    """Rank 0's commit: workers first, then this node (plain ``commit`` on one node)."""
    if nodes <= 1:
        return commit

    def commit_all() -> None:
        token = uuid.uuid4().hex
        progress_dict[key] = {"token": token, "ts": time.time()}
        acked, by = set(), time.monotonic() + timeout
        while len(acked) < nodes - 1:
            left = by - time.monotonic()
            if left <= 0:
                log(f"shards: no commit ack from ranks {sorted(set(range(1, nodes)) - acked)} in {timeout:.0f}s; "
                    "committing anyway")
                break
            item = queue.get(partition=ack_partition(run_id), timeout=min(left, 30))
            if item and item[1] == token:
                acked.add(item[0])
        commit()

    return commit_all


def worker_commits(run_id: str, rank: int, progress_dict, queue, commit: Callable[[], None], stop: threading.Event,
                   *, key: str, every: float, poll: float = POLL_SECS, log=print) -> None:
    """A worker node's loop: commit on rank 0's request (and acknowledge it) and every ``every`` seconds."""
    seen = None  # answering a request left by an earlier segment costs one commit; rank 0 ignores its token
    last = time.monotonic()
    while not stop.wait(poll):
        try:
            token = (progress_dict.get(key) or {}).get("token")
            if token and token != seen:
                commit()
                queue.put((rank, token), partition=ack_partition(run_id))
                seen, last = token, time.monotonic()
            elif time.monotonic() - last >= every:
                commit()
                last = time.monotonic()
        except Exception as e:  # the next poll retries; the exit commit still runs
            log(f"shards: commit on rank {rank}: {type(e).__name__}: {e}")
