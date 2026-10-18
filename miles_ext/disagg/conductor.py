"""When to start, retry and stop the trainer of a split run: a pure state machine plus the loop that drives it.

The rollout app reports progress of batch 0 (groups ready, groups wanted, elapsed seconds) into the run's Dict; the
conductor starts the trainer so that it comes up as batch 0 completes (projected finish minus the trainer's measured
startup), relaunches it on failure up to ``max_attempts``, and at the end stops every app it knows by id.
"""

from dataclasses import dataclass, field, replace
from enum import Enum


class Phase(Enum):
    WAIT_ROLLOUT = "wait_rollout"
    WAIT_BATCH = "wait_batch"
    TRAINER_STARTING = "trainer_starting"
    TRAINING = "training"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class Config:
    trainer_startup_s: float = 180.0
    max_attempts: int = 3
    ready_timeout_s: float = 900.0


DEFAULT = Config()


@dataclass(frozen=True)
class State:
    phase: Phase = Phase.WAIT_ROLLOUT
    attempts: int = 0
    trainer_started_at: float | None = None
    rollout_app: str | None = None
    trainer_apps: tuple[str, ...] = ()
    reason: str = ""
    log: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Action:
    kind: str  # "launch_trainer" | "stop_app" | "cleanup"
    arg: str = ""


def projected_ready_in(groups_ready: int, groups_wanted: int, elapsed_s: float) -> float | None:
    """Seconds until batch 0 is complete at the rate seen so far; None before the first group lands."""
    if groups_ready >= groups_wanted:
        return 0.0
    if groups_ready <= 0 or elapsed_s <= 0:
        return None
    return (groups_wanted - groups_ready) * elapsed_s / groups_ready


def _launch(s: State, now: float, why: str) -> tuple[State, list[Action]]:
    s = replace(
        s,
        phase=Phase.TRAINER_STARTING,
        attempts=s.attempts + 1,
        trainer_started_at=now,
        log=s.log + (f"{now:.0f} launch trainer #{s.attempts + 1}: {why}",),
    )
    return s, [Action("launch_trainer")]


def _end(s: State, phase: Phase, reason: str) -> tuple[State, list[Action]]:
    stops = [Action("stop_app", a) for a in (*s.trainer_apps, *([s.rollout_app] if s.rollout_app else []))]
    return replace(s, phase=phase, reason=reason, log=s.log + (f"end {phase.value}: {reason}",)), stops + [
        Action("cleanup")
    ]


def step(s: State, event: dict, now: float, cfg: Config = DEFAULT) -> tuple[State, list[Action]]:
    """Advance on one event. Events: rollout_up(app), progress(ready, wanted, elapsed), trainer_app(app),
    trainer_ready, trainer_failed(reason), rollout_done, rollout_failed(reason), tick."""
    kind = event["kind"]
    if s.phase in (Phase.DONE, Phase.FAILED):
        return s, []
    if kind == "rollout_up":
        return replace(s, phase=Phase.WAIT_BATCH, rollout_app=event["app"]), []
    if kind == "trainer_app":
        return replace(s, trainer_apps=s.trainer_apps + (event["app"],)), []
    if kind == "rollout_done":
        return _end(s, Phase.DONE, "rollout finished")
    if kind == "rollout_failed":
        return _end(s, Phase.FAILED, f"rollout failed: {event.get('reason', '')}")
    if s.phase is Phase.WAIT_BATCH and kind == "progress":
        eta = projected_ready_in(event["ready"], event["wanted"], event["elapsed"])
        if eta is not None and eta <= cfg.trainer_startup_s:
            return _launch(s, now, f"batch 0 ready in ~{eta:.0f}s, startup {cfg.trainer_startup_s:.0f}s")
        return s, []
    if s.phase is Phase.TRAINER_STARTING and kind == "trainer_ready":
        return replace(s, phase=Phase.TRAINING, log=s.log + (f"{now:.0f} trainer ready",)), []
    if s.phase is Phase.TRAINER_STARTING and kind == "tick" and now - s.trainer_started_at > cfg.ready_timeout_s:
        event = {"kind": "trainer_failed", "reason": f"not ready after {cfg.ready_timeout_s:.0f}s"}
        kind = "trainer_failed"
    if s.phase in (Phase.TRAINER_STARTING, Phase.TRAINING) and kind == "trainer_failed":
        reason = event.get("reason", "")
        stops = [Action("stop_app", s.trainer_apps[-1])] if s.trainer_apps else []
        if s.attempts >= cfg.max_attempts:
            return _end(s, Phase.FAILED, f"trainer failed {s.attempts}x, last: {reason}")
        s, actions = _launch(s, now, f"retry after: {reason}")
        return s, stops + actions
    return s, []
