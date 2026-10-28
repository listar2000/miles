"""Harbor trials run in children of one forkserver, with Miles' subproc contract (ext-042)."""

import asyncio
import os

import pytest
from miles_ext.harbor import trial_procs
from miles_ext.recipes import recipe as R

from miles.rollout.agentic.agent_function import report_to_caller

PRELOAD = ("miles.rollout.agentic.agent_function",)


async def _whoami(tag: str) -> dict:
    report_to_caller(f"report {tag}")
    return {"pid": os.getpid(), "tag": tag, "env": os.environ.get("TRIAL_PROCS_TEST")}


async def _fails() -> None:
    raise ValueError("boom in the trial")


async def _sleeps() -> None:
    report_to_caller("ready")
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        report_to_caller("cancelled")  # the trial's own cleanup runs on cancel
        raise


def test_trials_run_in_forked_children_with_reports_and_the_callers_env(monkeypatch):
    async def main():
        monkeypatch.setenv("TRIAL_PROCS_TEST", "set after the server started")
        reports = []
        outs = await asyncio.gather(
            *(
                trial_procs.call(_whoami, preload=PRELOAD, on_child_reports=reports.extend, tag=str(i))
                for i in range(3)
            )
        )
        return outs, reports

    trial_procs.start(PRELOAD)
    outs, reports = asyncio.run(main())
    assert [o["tag"] for o in outs] == ["0", "1", "2"]
    assert len({o["pid"] for o in outs} | {os.getpid()}) == 4  # a process per trial
    assert all(o["env"] == "set after the server started" for o in outs)
    assert sorted(reports) == ["report 0", "report 1", "report 2"]


def test_a_trial_error_raises_in_the_caller():
    with pytest.raises(ValueError, match="boom in the trial") as info:
        asyncio.run(trial_procs.call(_fails, preload=PRELOAD))
    assert "_fails" in str(info.value.__cause__)  # the child's traceback


def test_cancelling_the_call_cancels_the_trial():
    async def main():
        ready, cancelled = asyncio.Event(), asyncio.Event()

        def reports(items):
            (ready if "ready" in items else cancelled).set()

        task = asyncio.create_task(trial_procs.call(_sleeps, preload=PRELOAD, on_child_reports=reports))
        await asyncio.wait_for(ready.wait(), timeout=30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=5)

    asyncio.run(main())


def test_harbor_harnesses_run_inline_and_single_turn_in_subprocesses():
    modes = {}
    for harness in R.HARNESSES:
        recipe = R.from_dict({"name": "t", "model": {"preset": "qwen3.5-9b"}, "taskset": {"harness": harness}})
        argv = R.miles_args(recipe, "t-1", resume=False)
        modes[harness] = argv[argv.index("--custom-agent-function-mode") + 1]
    assert modes == {
        R.HARNESS_SANDBOX: "inline",
        R.HARNESS_HOST: "inline",
        R.HARNESS_TERMINUS_SANDBOX: "inline",
        R.HARNESS_SINGLE_TURN: "subproc",
    }
