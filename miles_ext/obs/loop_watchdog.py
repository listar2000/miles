"""Log every thread's stack when the rollout event loop stalls, and the stall's length when it recovers."""

import asyncio
import logging
import sys
import threading
import time
import traceback

from miles.rollout.generate_hub import agentic_tool_call

logger = logging.getLogger(__name__)

_TICK_S = 0.5
_STALL_S = 5.0
_RESAMPLE_S = 30.0
_started: set[int] = set()  # ids of loops already watched


def _stacks(loop_thread_id: int, *, all_threads: bool) -> str:
    frames = sys._current_frames()
    names = {t.ident: t.name for t in threading.enumerate()}
    ids = [loop_thread_id] + (sorted(set(frames) - {loop_thread_id, threading.get_ident()}) if all_threads else [])
    out = []
    for tid in ids:
        frame = frames.get(tid)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else "(thread gone)\n"
        out.append(f"--- thread {names.get(tid, tid)}{' (loop)' if tid == loop_thread_id else ''}:\n{stack}")
    return "".join(out)


def watch(
    loop: asyncio.AbstractEventLoop,
    loop_thread_id: int,
    *,
    stall_s: float = _STALL_S,
    tick_s: float = _TICK_S,
    resample_s: float = _RESAMPLE_S,
) -> threading.Event:
    """Start the watcher for ``loop``, which runs on thread ``loop_thread_id``; set the returned event to stop it."""
    stop = threading.Event()
    last_beat = [time.monotonic()]

    def beat() -> None:
        last_beat[0] = time.monotonic()

    def run() -> None:
        stalled_since = last_sample = None
        while not stop.wait(tick_s):
            try:
                loop.call_soon_threadsafe(beat)
            except RuntimeError:  # the loop is closed
                return
            lag = time.monotonic() - last_beat[0]
            now = time.monotonic()
            if lag >= stall_s and stalled_since is None:
                stalled_since, last_sample = last_beat[0], now
                stacks = _stacks(loop_thread_id, all_threads=True)
                logger.warning(f"rollout event loop stalled for {lag:.1f}s; thread stacks:\n{stacks}")
            elif lag >= stall_s and now - last_sample >= resample_s:
                last_sample = now
                stacks = _stacks(loop_thread_id, all_threads=False)
                logger.warning(f"rollout event loop still stalled after {lag:.1f}s:\n{stacks}")
            elif lag < stall_s and stalled_since is not None:
                logger.warning(f"rollout event loop stall ended after {last_beat[0] - stalled_since:.1f}s")
                stalled_since = None

    threading.Thread(target=run, name="loop-watchdog", daemon=True).start()
    return stop


async def generate(input):
    loop = asyncio.get_running_loop()
    if id(loop) not in _started:
        _started.add(id(loop))
        watch(loop, threading.get_ident())
    return await agentic_tool_call.generate(input)


generate.add_arguments = agentic_tool_call.generate.add_arguments
