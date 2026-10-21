"""Size fully async rollout concurrency from measured group times and drain periods."""

import asyncio
import math
import statistics
import time
from collections import deque

MARGIN = 1.15  # headroom over exact balance
PERIODS = 3  # drain periods averaged
GROUPS_PER_BATCH = 4  # group times averaged, in batches


class AutoInFlight:
    def __init__(self, args, clock=time.monotonic):
        self._batch = args.rollout_batch_size
        self._clock = clock
        self._started: dict[asyncio.Task, float] = {}
        self._group_s: deque[float] = deque(maxlen=GROUPS_PER_BATCH * self._batch)
        self._period_s: deque[float] = deque(maxlen=PERIODS)
        self._last_drain: float | None = None
        self._target: int | None = None  # None until a batch of groups and one drain period are measured
        self._cap: int | None = None

    def on_submit(self, task: asyncio.Task) -> None:
        self._started[task] = self._clock()
        task.add_done_callback(self._on_done)  # at completion, not when the worker hands the group on

    def _on_done(self, task: asyncio.Task) -> None:
        if (start := self._started.pop(task, None)) is not None and not task.cancelled():
            self._group_s.append(self._clock() - start)

    def on_drain(self) -> None:
        now = self._clock()
        if self._last_drain is not None:
            self._period_s.append(now - self._last_drain)
        self._last_drain = now
        if len(self._group_s) >= self._batch and self._period_s:
            group_s, period_s = statistics.mean(self._group_s), statistics.mean(self._period_s)
            self._target = math.ceil(MARGIN * self._batch * group_s / max(period_s, 1e-9))

    def budget(self, cap: int) -> int:
        self._cap = cap
        return cap if self._target is None else max(min(self._batch, cap), min(cap, self._target))

    def metrics(self) -> dict[str, float]:
        prefix = "rollout/fully_async/in_flight_"
        out = {}
        if self._cap is not None:
            out[prefix + "budget_groups"] = self.budget(self._cap)
        if self._group_s:
            out[prefix + "group_time_s"] = statistics.mean(self._group_s)
        if self._period_s:
            out[prefix + "drain_period_s"] = statistics.mean(self._period_s)
        return out
