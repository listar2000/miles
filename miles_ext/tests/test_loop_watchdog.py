import asyncio
import logging
import threading
import time

from miles_ext.obs import loop_watchdog
from miles_ext.recipes import recipe as R


def _blocking_write():
    time.sleep(0.6)


def test_stall_logs_the_blocking_frame_and_its_length(caplog):
    async def main():
        stop = loop_watchdog.watch(
            asyncio.get_running_loop(), threading.get_ident(), stall_s=0.2, tick_s=0.05, resample_s=0.2
        )
        await asyncio.sleep(0.1)
        _blocking_write()
        await asyncio.sleep(0.2)
        stop.set()

    with caplog.at_level(logging.WARNING, logger=loop_watchdog.__name__):
        asyncio.run(main())
    stalled, *resampled, ended = [r.getMessage() for r in caplog.records]
    assert "_blocking_write" in stalled and "thread MainThread (loop)" in stalled
    assert resampled and all("still stalled" in m and "_blocking_write" in m for m in resampled)
    assert float(ended.split("after ")[1].rstrip("s")) >= 0.5


def test_recipe_runs_trials_in_subprocesses():
    argv = R.miles_args(R.from_dict({"name": "t", "model": {"preset": "qwen3.5-9b"}}), "t-1", resume=False)
    assert argv[argv.index("--custom-agent-function-mode") + 1] == "subproc"
