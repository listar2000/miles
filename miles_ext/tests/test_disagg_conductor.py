"""The disagg conductor starts the trainer just in time, retries it, and stops every app it launched:

    python -m pytest miles_ext/tests/test_disagg_conductor.py
"""

from miles_ext.disagg.conductor import Config, Phase, State, projected_ready_in, step

CFG = Config(trainer_startup_s=100, max_attempts=2, ready_timeout_s=300)


class FakeTrainer:
    """Drives the state machine like the real loop: each launch becomes an app id, then ready or failed."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.launched, self.stopped, self.cleaned = [], [], False

    def run(self, s, event, now):
        s, actions = step(s, event, now, CFG)
        for a in actions:
            if a.kind == "launch_trainer":
                app = f"ap-trainer-{len(self.launched)}"
                self.launched.append(app)
                s, _ = step(s, {"kind": "trainer_app", "app": app}, now, CFG)
                outcome = self.outcomes.pop(0)
                s = self.run(s, {"kind": outcome, "reason": "boom"}, now + 50)
            elif a.kind == "stop_app":
                self.stopped.append(a.arg)
            elif a.kind == "cleanup":
                self.cleaned = True
        return s


def _up():
    s, _ = step(State(), {"kind": "rollout_up", "app": "ap-rollout"}, 0, CFG)
    return s


def test_projection():
    assert projected_ready_in(0, 32, 60) is None
    assert projected_ready_in(8, 32, 60) == 180
    assert projected_ready_in(32, 32, 60) == 0


def test_trainer_waits_until_batch_is_within_startup():
    fake = FakeTrainer(["trainer_ready"])
    s = fake.run(_up(), {"kind": "progress", "ready": 8, "wanted": 32, "elapsed": 60}, 60)  # eta 180 > 100
    assert s.phase is Phase.WAIT_BATCH and not fake.launched
    s = fake.run(s, {"kind": "progress", "ready": 20, "wanted": 32, "elapsed": 120}, 120)  # eta 72 <= 100
    assert s.phase is Phase.TRAINING and fake.launched == ["ap-trainer-0"]


def test_failed_trainer_is_stopped_and_relaunched_then_run_fails_and_cleans_up():
    fake = FakeTrainer(["trainer_failed", "trainer_failed"])
    s = fake.run(_up(), {"kind": "progress", "ready": 32, "wanted": 32, "elapsed": 10}, 10)
    assert s.phase is Phase.FAILED and s.attempts == 2
    assert fake.launched == ["ap-trainer-0", "ap-trainer-1"]
    assert set(fake.stopped) == {"ap-trainer-0", "ap-trainer-1", "ap-rollout"} and fake.cleaned


def test_trainer_that_never_becomes_ready_times_out_into_a_retry():
    s = _up()
    s, actions = step(s, {"kind": "progress", "ready": 32, "wanted": 32, "elapsed": 1}, 0, CFG)
    s, _ = step(s, {"kind": "trainer_app", "app": "ap-t0"}, 0, CFG)
    assert step(s, {"kind": "tick"}, 299, CFG)[1] == []
    s, actions = step(s, {"kind": "tick"}, 301, CFG)
    assert [a.kind for a in actions] == ["stop_app", "launch_trainer"] and s.attempts == 2


def test_rollout_done_stops_everything_once():
    fake = FakeTrainer(["trainer_ready"])
    s = fake.run(_up(), {"kind": "progress", "ready": 32, "wanted": 32, "elapsed": 1}, 0)
    s = fake.run(s, {"kind": "rollout_done"}, 500)
    assert s.phase is Phase.DONE and set(fake.stopped) == {"ap-trainer-0", "ap-rollout"} and fake.cleaned
    assert step(s, {"kind": "trainer_failed"}, 600, CFG) == (s, [])
