"""Split runs on Modal: a rollout app (engines + primary) and a trainer app that joins late or serves one step.

    MILES_DISAGG_RUN=<run> modal run --detach -m miles_ext.disagg.split --phase late|per-step --num-rollout 3

Apps: ``miles-disagg-<run>`` (conductor and rollout functions; ephemeral, detached) and ``miles-disagg-<run>-trainer``
(deployed by the entrypoint so the conductor can spawn it; stopped by id at the end). Both pin one cloud and region
(app.CLOUD, app.REGION) so their i6pn addresses reach each other. The run's Dict ``miles-disagg-<run>`` carries
addresses and events.
"""

import json
import os
import time
import uuid

import modal
from miles_ext.disagg import argv as A
from miles_ext.disagg import conductor as C
from miles_ext.disagg.app import (
    CLOUD,
    HOURS,
    INFERENCE_CONTROLLER_POOL,
    REGION,
    SHAPE,
    STATE,
    TRAINER_POOL,
    base_image,
    secrets,
    state,
    volumes,
    with_code,
)
from miles_ext.disagg.late_start import DICT_ENV, EVENTS_KEY, TRAINER_ADDR_KEY, mark

RUN = os.environ.get("MILES_DISAGG_RUN", "unset")
ROLLOUT_APP, TRAINER_APP = f"miles-disagg-{RUN}", f"miles-disagg-{RUN}-trainer"
HOOK = "miles_ext.disagg.late_start.start_rollout_before_trainer"
ENTRY = ["python", "-m", "miles.utils.workers.deployment_entrypoint"]
RUN_EXIT = ["python", "-m", "miles_ext.disagg.run_exit"]
ROLLOUT_KEY, STOP_KEY = "rollout", "stop"
LATE, PER_STEP = "late", "per-step"  # --phase: the trainer joins once, late, or serves one step at a time
SEGMENT_SIZE = "16gb"  # Mooncake memory for rollout batches (Qwen3-4B bench: 256 samples x 8k tokens, well under)
_NCCL_V6 = {"NCCL_SOCKET_FAMILY": "AF_INET6", "GLOO_SOCKET_FAMILY": "AF_INET6"}

image = with_code(base_image.env({"MILES_DISAGG_RUN": RUN, **_NCCL_V6}))
rollout_app = modal.App(ROLLOUT_APP)
trainer_app = modal.App(TRAINER_APP)
gpu_kw = dict(cloud=CLOUD, region=REGION, i6pn=True, secrets=secrets, volumes=volumes, image=image, timeout=2 * HOURS)


def _env(cluster, dict_name: str, cfg: A.RunConfig) -> dict:
    return {**cluster.env(), DICT_ENV: dict_name, **_NCCL_V6, **A.wandb_env(cfg, "split")}


@rollout_app.function(gpu=f"H100:{SHAPE['rollout_gpus']}", cpu=16, memory=131072, **gpu_kw)
def rollout(cfg: dict, dict_name: str, mode: str = LATE) -> dict:
    """Engines (inference deployment) + primary: train_async with the late-start hook (late) or the on-demand
    driver (per-step); returns when the primary exits."""
    from miles_ext.disagg import procs as P

    cfg, d = A.RunConfig.of(cfg), modal.Dict.from_name(dict_name)
    host = P.i6pn_addr()
    primary_c = P.RayCluster("primary", 1, host)
    infer_c = P.RayCluster("inference", 2, host, gpus=tuple(range(SHAPE["rollout_gpus"])))
    moon, drivers = P.start_mooncake_master(), []
    try:
        for c in (primary_c, infer_c):
            c.start()
        mooncake = A.host_port(host, P.MOONCAKE_PORT)
        segment = P.Driver(
            "segment",
            ["python", "-m", "miles_ext.disagg.segment_host", mooncake, host, SEGMENT_SIZE],
            primary_c.env(),
        ).start()
        drivers.append(segment)
        segment.wait_for(lambda dr: any("segment host:" in x for x in dr.lines), timeout=300, what="a mounted segment")
        d[ROLLOUT_KEY] = {"host": host, "mooncake": mooncake, "app_id": modal.current_function_call_id()}
        entry = RUN_EXIT + (["train_async.py"] if mode == LATE else ["-m", "miles_ext.disagg.ondemand"])
        late = ["--custom-pre-trainer-hook-path", HOOK] if mode == LATE else []
        # the trainer's address arrives later through the Dict; Miles' validator wants one at parse time
        primary = P.Driver(
            "primary",
            entry + A.build(cfg, A.PRIMARY, mooncake_master=mooncake, trainer_addr="[::1]:1") + late,
            _env(primary_c, dict_name, cfg),
        ).start()
        drivers.append(primary)
        ic_addr = A.host_port(*primary_c.pool_addr(INFERENCE_CONTROLLER_POOL, driver=primary))
        drivers.append(
            P.Driver(
                "inference",
                ENTRY + A.build(cfg, A.INFERENCE, mooncake_master=mooncake, inference_controller_addr=ic_addr),
                _env(infer_c, dict_name, cfg),
            ).start()
        )
        drivers.append(
            P.Driver("medic", ["python", "-m", "miles_ext.disagg.medic"], _env(infer_c, dict_name, cfg)).start()
        )
        mark(d, "rollout_up")
        rc = primary.proc.wait()
        mark(d, "rollout_done" if rc == 0 else "rollout_failed", returncode=rc, tail="".join(primary.lines[-30:]))
        return {"returncode": rc}
    except Exception as e:
        mark(d, "rollout_failed", error=f"{type(e).__name__}: {e}"[:500])
        raise
    finally:
        d[STOP_KEY] = True
        for x in reversed(drivers):
            x.stop()
        for c in (primary_c, infer_c):
            c.stop()
        moon.terminate()
        state.commit()


@trainer_app.function(gpu=f"H100:{SHAPE['trainer_gpus']}", cpu=16, memory=131072, **gpu_kw)
def trainer(cfg: dict, dict_name: str, attempt: int) -> dict:
    """The trainer deployment: checks it can reach the rollout app, serves its controller, waits for the stop."""
    from miles_ext.disagg import procs as P

    cfg, d = A.RunConfig.of(cfg), modal.Dict.from_name(dict_name)
    mark(d, f"trainer{attempt}_container_up")
    ro = d[ROLLOUT_KEY]
    try:
        P.wait_port(ro["host"], P.MOONCAKE_PORT, timeout=20)
    except TimeoutError:
        mark(d, f"trainer{attempt}_failed", reason="rollout app unreachable over i6pn")
        return {"ok": False, "reason": "unreachable"}
    host = P.i6pn_addr()
    c = P.RayCluster("trainer", 0, host, gpus=tuple(range(SHAPE["trainer_gpus"])))
    drv = None
    try:
        c.start()
        drv = P.Driver(
            "trainer",
            ENTRY + A.build(cfg, A.TRAINER, mooncake_master=ro["mooncake"], trainer_segment=False),
            _env(c, dict_name, cfg),
        ).start()
        d[TRAINER_ADDR_KEY] = A.host_port(*c.pool_addr(TRAINER_POOL, driver=drv))
        mark(d, f"trainer{attempt}_ready")
        while not d.get(STOP_KEY):
            if drv.proc.poll() is not None:
                mark(
                    d, f"trainer{attempt}_failed", reason=f"exit {drv.proc.returncode}", tail="".join(drv.lines[-30:])
                )
                return {"ok": False, "reason": "exited"}
            time.sleep(5)
        return {"ok": True}
    finally:
        if drv:
            drv.stop()
        c.stop()


def _prune_checkpoints(save_dir: str, keep: int = 2) -> list[str]:
    """Delete all but the newest ``keep`` iteration directories of this run's checkpoint dir."""
    import shutil
    from pathlib import Path

    iters = sorted(Path(save_dir).glob("iter_*"))
    for old in iters[:-keep]:
        shutil.rmtree(old)
    return [p.name for p in iters[:-keep]]


@trainer_app.function(gpu=f"H100:{SHAPE['trainer_gpus']}", cpu=16, memory=131072, **gpu_kw)
def step_trainer(cfg: dict, dict_name: str, step: int, attempt: int) -> dict:
    """Per-step: one trainer deployment serving one step; commits the volume when the driver has saved."""
    from miles_ext.disagg import ondemand as O
    from miles_ext.disagg import procs as P

    t0, tag = time.time(), f"{step}/{attempt}"
    cfg, d = A.RunConfig.of(cfg), modal.Dict.from_name(dict_name)
    ro, out, drv = d[ROLLOUT_KEY], {"step": step, "attempt": attempt, "container_up": t0}, None
    if O.superseded(d.get(O.WANT_KEY.format(step)), attempt):  # started after the driver gave up on this attempt
        out["superseded"] = True
        d[f"paid/{tag}"] = {**out, "paid_s": time.time() - t0}
        return out
    c = P.RayCluster("trainer", 0, P.i6pn_addr(), gpus=tuple(range(SHAPE["trainer_gpus"])))
    try:
        # Modal may reuse the previous step's container, whose view of the volume predates that step's commit
        state.reload()
        P.wait_port(ro["host"], P.MOONCAKE_PORT, timeout=20)
        c.start()
        drv = P.Driver(
            "trainer",
            ENTRY + A.build(cfg, A.TRAINER, mooncake_master=ro["mooncake"], trainer_segment=False),
            _env(c, dict_name, cfg),
        ).start()
        d[O.TRAINER_ADDR_KEY.format(step=step) + f"/{attempt}"] = A.host_port(*c.pool_addr(TRAINER_POOL, driver=drv))
        out["ready"] = time.time()
        while not (d.get(O.RELEASE_KEY.format(step)) or d.get(STOP_KEY)):
            if drv.proc.poll() is not None:
                raise RuntimeError(f"trainer exited {drv.proc.returncode}: {''.join(drv.lines[-20:])[-1500:]}")
            if d.get(O.COMMIT_KEY.format(step)) is not None and "committed" not in out:
                out["pruned"] = _prune_checkpoints(cfg.save_dir)
                state.commit()
                out["committed"] = time.time()
                d[O.COMMITTED_KEY.format(step)] = out["committed"]
            time.sleep(1)
        out["released"] = time.time()
    except Exception as e:  # recorded for the driver, which retries the step
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
        d[O.LOST_KEY.format(step) + f"/{attempt}"] = out["error"]
    finally:
        if drv:
            drv.stop()
        c.stop()
        out["paid_s"] = time.time() - t0
        d[f"paid/{tag}"] = out
    return out


@rollout_app.function(image=image, cpu=1, memory=2048, timeout=2 * HOURS + 600)
def conductor_per_step(cfg: dict, dict_name: str, trainer_app_id: str, kill: str = "") -> dict:
    """Per-step: start the rollout (on-demand driver), launch one trainer per step request, stop everything.

    ``kill`` tests trainer loss: ``<step>:<point>[,...]`` cancels that step's first trainer once it is ``train``-ing
    (before its save) or once its save is ``committed`` (before its weight push)."""
    from miles_ext.disagg import ondemand as O
    from miles_ext.disagg import procs as P

    d = modal.Dict.from_name(dict_name)
    t_fn = modal.Function.from_name(TRAINER_APP, "step_trainer")
    ro_call, calls, started = _spawn_once(d, "calls/rollout", rollout, cfg, dict_name, PER_STEP), {}, time.time()
    kills = {int(k): point for k, point in (x.split(":") for x in kill.split(",") if x)}
    cancelled: set = set()
    try:
        while not _done(ro_call) and time.time() - started < 2 * HOURS:
            for step, point in list(kills.items()):
                key = O.TRAINER_ADDR_KEY.format(step=step) + "/0" if point == "train" else O.COMMITTED_KEY.format(step)
                if (step, 0) in calls and d.get(key) is not None:
                    calls[(step, 0)].cancel()
                    mark(d, f"killed_step{step}_{point}")
                    del kills[step]
            for key, want in [(k, v) for k, v in d.items() if k.startswith("want/")]:
                step, attempt = int(key.split("/")[1]), want["attempt"]
                for old in [k for k in calls if k[0] == step and O.superseded(want, k[1]) and k not in cancelled]:
                    calls[old].cancel()  # a trainer still queued for a GPU would start late and idle
                    cancelled.add(old)
                if (step, attempt) not in calls:
                    calls[(step, attempt)] = _spawn_once(
                        d, f"calls/step/{step}/{attempt}", t_fn, cfg, dict_name, step, attempt
                    )
            for (step, attempt), call in calls.items():
                lost = O.LOST_KEY.format(step) + f"/{attempt}"
                if _done(call) and not d.get(O.RELEASE_KEY.format(step)) and d.get(lost) is None:
                    d[lost] = "trainer function ended before release"
            time.sleep(2)
    finally:
        d[STOP_KEY] = True
        for call in calls.values():
            call.cancel()
        ro_call.cancel()
        P.stop_app(trainer_app_id)
    return {k: v for k, v in d.items() if k.split("/")[0] in ("steps", "paid", "lost", EVENTS_KEY)}


@rollout_app.function(image=image, cpu=1, memory=2048, timeout=2 * HOURS + 600)
def conductor(cfg: dict, dict_name: str, trainer_app_id: str, ccfg: dict) -> dict:
    """Start the rollout, start the trainer just in time, retry it, and stop every app at the end."""
    from miles_ext.disagg import procs as P

    d, ccfg = modal.Dict.from_name(dict_name), C.Config(**ccfg)
    t_fn = modal.Function.from_name(TRAINER_APP, "trainer")
    ro_call = _spawn_once(d, "calls/rollout", rollout, cfg, dict_name)
    s, calls, seen = C.State(), [], set()
    s, _ = C.step(s, {"kind": "rollout_up", "app": modal.current_function_call_id() or ""}, time.time(), ccfg)
    started = time.time()
    while s.phase not in (C.Phase.DONE, C.Phase.FAILED):
        now = time.time()
        events = {k[len(EVENTS_KEY) + 1 :]: v for k, v in d.items() if k.startswith(f"{EVENTS_KEY}/")}
        for name in sorted(set(events) - seen, key=lambda n: events[n]["t"]):
            seen.add(name)
            kind = None
            if name.endswith("_ready") and name.startswith("trainer"):
                kind = "trainer_ready"
            elif name.endswith("_failed") and name.startswith("trainer"):
                kind = "trainer_failed"
            elif name in ("rollout_done", "rollout_failed"):
                kind = name
            if kind:
                s, actions = C.step(s, {"kind": kind, **events[name]}, now, ccfg)
                calls = _act(actions, t_fn, cfg, dict_name, s, calls, d)
        if "rollout0_started" in events and s.phase is C.Phase.WAIT_BATCH:
            # progress proxy: time since rollout 0 started against the expected batch time
            el = now - events["rollout0_started"]["t"]
            ev = {"kind": "progress", "ready": int(el), "wanted": max(1, int(cfg["expected_batch_s"])), "elapsed": el}
            s, actions = C.step(s, ev, now, ccfg)
            calls = _act(actions, t_fn, cfg, dict_name, s, calls, d)
        s, actions = C.step(s, {"kind": "tick"}, now, ccfg)
        calls = _act(actions, t_fn, cfg, dict_name, s, calls, d)
        if now - started > 2 * HOURS:
            s, actions = C.step(s, {"kind": "rollout_failed", "reason": "2 h hard stop"}, now, ccfg)
        if ro_call and _done(ro_call) and s.phase not in (C.Phase.DONE, C.Phase.FAILED):
            s, actions = C.step(s, {"kind": "rollout_failed", "reason": "rollout function ended"}, now, ccfg)
        time.sleep(3)
    d[STOP_KEY] = True
    for call in calls:
        call.cancel()
    ro_call.cancel()
    P.stop_app(trainer_app_id)
    return {
        "phase": s.phase.value,
        "reason": s.reason,
        "attempts": s.attempts,
        "log": list(s.log),
        "events": {k: v for k, v in d.items() if k.startswith(f"{EVENTS_KEY}/")},
    }


def _done(call) -> bool:
    try:
        call.get(timeout=0)
        return True
    except (TimeoutError, modal.exception.TimeoutError):  # pending (modal raises the builtin one) or not yet queued
        return False
    except Exception:  # a failed call is done too
        return True


def _spawn_once(d, key: str, fn, *args):
    """Spawn ``fn`` unless this run already did: Modal restarts a preempted conductor with the same input, and it
    must reattach to the rollout and trainers it started instead of starting a second set."""
    if (call_id := d.get(key)) is not None:
        return modal.FunctionCall.from_id(call_id)
    call = fn.spawn(*args)
    d[key] = call.object_id
    return call


def _act(actions, t_fn, cfg, dict_name, s, calls, d):
    for a in actions:
        if a.kind == "launch_trainer":
            calls.append(_spawn_once(d, f"calls/trainer/{s.attempts}", t_fn, cfg, dict_name, s.attempts))
        elif a.kind == "stop_app" and calls:
            calls[-1].cancel()
    return calls


@rollout_app.local_entrypoint()
def main(
    phase: str = LATE,
    num_rollout: int = 0,
    expected_batch_s: int = 0,
    trainer_startup_s: int = 180,
    max_attempts: int = 3,
    bench: bool = False,
    replay: str = "",
    kill: str = "",
):
    assert phase in (LATE, PER_STEP), f"--phase must be {LATE} or {PER_STEP}, got {phase!r}"
    kw = dict(expected_batch_s=expected_batch_s, save_dir=f"{STATE}/{RUN}/ckpt" if phase == PER_STEP else "")
    steps = {"num_rollout": num_rollout} if num_rollout else {}  # overrides the preset's run length only when given
    if replay:
        cfg = A.replay(RUN, uuid.uuid4().hex[:16], replay, **kw, **steps).as_dict()
    elif bench:
        cfg = A.bench(RUN, uuid.uuid4().hex[:16], **kw, **steps).as_dict()
    else:
        cfg = A.RunConfig(run_id=RUN, run_uuid=uuid.uuid4().hex[:16], num_rollout=num_rollout or 3, **kw).as_dict()
    dict_name = f"miles-disagg-{RUN}"
    modal.Dict.objects.create(dict_name, allow_existing=True)
    trainer_app.deploy(name=TRAINER_APP)
    trainer_app_id = trainer_app.app_id
    print(f"trainer app {TRAINER_APP} {trainer_app_id}", flush=True)
    ccfg = {"trainer_startup_s": trainer_startup_s, "max_attempts": max_attempts}
    try:
        if phase == PER_STEP:
            out = conductor_per_step.remote(cfg, dict_name, trainer_app_id, kill)
        else:
            out = conductor.remote(cfg, dict_name, trainer_app_id, ccfg)
        print(json.dumps(out, indent=1, default=str))
    finally:
        modal.Dict.objects.delete(dict_name)
