from types import SimpleNamespace

from miles_ext.rollout.in_flight import AutoInFlight


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Task:
    def __init__(self, cancelled=False):
        self._cancelled = cancelled

    def cancelled(self):
        return self._cancelled

    def add_done_callback(self, fn):
        self.finish = lambda: fn(self)


def run_step(ctl, clock, *, groups, group_s, period_s):
    for _ in range(groups):
        task = Task()
        ctl.on_submit(task)
        clock.now += group_s
        task.finish()
        clock.now -= group_s
    clock.now += period_s
    ctl.on_drain()


def test_the_budget_tracks_what_the_trainer_consumes():
    clock = Clock()
    ctl = AutoInFlight(SimpleNamespace(rollout_batch_size=8), clock=clock)
    assert ctl.budget(16) == 16  # nothing measured yet
    for _ in range(3):
        run_step(ctl, clock, groups=8, group_s=46 * 60, period_s=32 * 60)
    assert ctl.budget(16) == 14  # ceil(1.15 x 8 x 46 / 32)
    assert ctl.metrics()["rollout/fully_async/in_flight_budget_groups"] == 14
    for _ in range(3):
        run_step(ctl, clock, groups=8, group_s=46 * 60, period_s=20 * 60)
    assert ctl.budget(16) == 16  # a faster trainer: back to the cap
    for _ in range(4):
        run_step(ctl, clock, groups=8, group_s=10 * 60, period_s=60 * 60)
    assert ctl.budget(16) == 8  # never below one batch


def test_cancelled_groups_are_not_timed():
    ctl = AutoInFlight(SimpleNamespace(rollout_batch_size=1), clock=Clock())
    task = Task(cancelled=True)
    ctl.on_submit(task)
    task.finish()
    assert not ctl.metrics()
