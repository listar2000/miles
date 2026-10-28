"""Run each Harbor trial in a child of a preloaded forkserver, with Miles' subproc protocol (ext-042)."""

import asyncio
import contextlib
import ctypes
import logging
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

from miles.rollout.agentic import agent_function as _af
from miles.rollout.agentic import rollout_limits
from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

# what this module uses from Miles' subproc agent calls; a rename upstream fails the import, not a rollout
_AF_API = (
    "_ResultPipe",
    "_run_and_report",
    "_read_results",
    "_unpack_outcome",
    "_send_outcome",
    "_exit",
    "_CANCEL_GRACE_S",
    "_LEFTOVER_CLEANUP_S",
    "_PR_SET_PDEATHSIG",
    "_result_pipe",
)
if missing := [n for n in _AF_API if not hasattr(_af, n)]:
    raise ImportError(f"miles.rollout.agentic.agent_function no longer defines {missing}; adapt {__name__}")
if not isinstance(getattr(multiprocessing.process, "_children", None), set):
    raise ImportError(f"multiprocessing.process._children is gone; adapt {__name__}._start")

_lock = threading.Lock()
_ctx: Any = None
_reapers: set[asyncio.Future] = set()
_starter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trial-procs-start")


def in_agent_process() -> bool:
    """True inside an agent process (this module's children, or Miles' own subproc children)."""
    return _af._result_pipe is not None


def start(preload: Iterable[str]) -> None:
    """Start the forkserver once and wait for its preload to finish."""
    global _ctx
    with _lock:
        if _ctx is None:
            modules = list(dict.fromkeys(preload))
            ctx = multiprocessing.get_context("forkserver")
            ctx.set_forkserver_preload(modules)
            from multiprocessing import forkserver

            forkserver.ensure_running()
            # the server imports ``preload`` after it starts; wait until it forks
            warmup = ctx.Process(target=_noop, daemon=True)
            warmup.start()
            warmup.join()
            _ctx = ctx
            logger.info(f"trial forkserver up, preloaded {modules}")


def _start(proc, write_end) -> None:
    try:
        proc.start()
    finally:
        write_end.close()
    # reaped by _watch on the loop; left in the table, every later start would poll it from this thread
    multiprocessing.process._children.discard(proc)


def _noop() -> None:
    pass


async def call(
    fn: Callable[..., Awaitable[Any]],
    *,
    preload: Iterable[str],
    on_child_reports: Callable[[list[Any]], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Await ``fn(**kwargs)`` in a child of the forkserver; same contract as Miles' subproc call."""
    if _ctx is None:
        await asyncio.to_thread(start, tuple(preload))
    # pickled here, so an unpicklable argument fails in the caller, as under subproc
    payload = pickle.dumps((fn, kwargs, _af._CANCEL_GRACE_S, _af._LEFTOVER_CLEANUP_S))
    env = {**os.environ, rollout_limits.SCOPE_ENV_VAR: rollout_limits.current_scope()}
    read_end, write_end = _ctx.Pipe(duplex=False)
    proc = _ctx.Process(target=_child_main, args=(write_end, payload, env), daemon=True)
    # Starting a process waits for the server to fork; keep that wait off the event loop.
    started = asyncio.get_running_loop().run_in_executor(_starter, _start, proc, write_end)
    try:
        await asyncio.shield(started)
    except BaseException:
        with contextlib.suppress(BaseException):
            await started  # a fork already asked for lands first, then is stopped
        read_end.close()
        if proc.pid is not None:
            _terminate(proc.pid, _watch(proc))
        raise
    pid, exited = proc.pid, _watch(proc)
    try:
        reader, transport = await _stream(read_end)
    except BaseException:
        _terminate(pid, exited)
        raise

    async def results() -> bytes | None:
        try:
            return await _af._read_results(SimpleNamespace(stdout=reader), on_child_reports)
        finally:
            transport.close()

    try:
        # not cancelled with the call, so reports from the agent's cancel cleanup still arrive
        output = await asyncio.shield(results())
    except asyncio.CancelledError:
        _terminate(pid, exited)
        raise
    if output is None:
        returncode = await exited
        raise RuntimeError(f"The agent process exited with code {returncode} before returning a result")
    return _af._unpack_outcome(output)


async def _stream(conn) -> tuple[asyncio.StreamReader, asyncio.BaseTransport]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    pipe = os.fdopen(_take_fd(conn), "rb", buffering=0)
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)
    return reader, transport


def _take_fd(conn) -> int:
    """The connection's descriptor as a plain fd of our own (Connection has no detach)."""
    fd = os.dup(conn.fileno())
    conn.close()
    return fd


def _watch(proc) -> asyncio.Future:
    """A future for the child's exit code; also reaps it (the outcome comes before the agent's leftover cleanup)."""
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    _reapers.add(done)
    done.add_done_callback(_reapers.discard)

    def on_exit() -> None:
        loop.remove_reader(proc.sentinel)
        proc.join(0)
        code = proc.exitcode
        proc.close()
        if not done.done():
            done.set_result(code)

    loop.add_reader(proc.sentinel, on_exit)
    return done


def _terminate(pid: int, exited: asyncio.Future) -> None:
    """SIGTERM now, SIGKILL after the grace periods; neither once the child is reaped (its pid may be reused)."""
    if exited.done():
        return
    _send_signal(pid, signal.SIGTERM)  # the agent process cancels the agent and enforces both deadlines itself
    # a backstop for an agent process that cannot run Python to enforce them, such as one stuck holding the GIL
    kill = asyncio.get_running_loop().call_later(
        _af._CANCEL_GRACE_S + _af._LEFTOVER_CLEANUP_S, _send_signal, pid, signal.SIGKILL
    )
    exited.add_done_callback(lambda _: kill.cancel())


def _send_signal(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def _child_main(conn, payload: bytes, env: dict[str, str]) -> None:
    """The forked child: Miles' subproc child (``_child_main``) with the call and pipe from the forkserver."""
    _die_with_parent(os.getppid())
    os.environ.clear()
    os.environ.update(env)  # the caller's environment now, not the server's at its start
    result_pipe = _af._ResultPipe(os.fdopen(_take_fd(conn), "wb"))
    _af._result_pipe = result_pipe  # report_to_caller sends on it
    configure_logger_raw(f"agent_function pid={os.getpid()}")
    try:
        fn, kwargs, cancel_grace_s, leftover_cleanup_s = pickle.loads(payload)
    except Exception as error:
        _af._send_outcome(result_pipe, ("error", error, traceback.format_exc()))
        _af._exit(0)
    threads_before = set(threading.enumerate())
    try:
        asyncio.run(_af._run_and_report(fn, kwargs, result_pipe, cancel_grace_s, leftover_cleanup_s))
    except BaseException:
        traceback.print_exc()
        _af._exit(1)
    # the deadline armed once the call was over bounds this wait
    for thread in set(threading.enumerate()) - threads_before:
        if not thread.daemon:
            thread.join()
    _af._exit(0)


def _die_with_parent(parent_pid: int) -> None:
    """SIGKILL this child when the forkserver exits, which it does when the rollout process dies."""
    if not sys.platform.startswith("linux"):
        return
    if ctypes.CDLL(None, use_errno=True).prctl(_af._PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        os._exit(1)  # the server died before the signal was armed
