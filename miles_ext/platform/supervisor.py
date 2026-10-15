"""R5/R6: rank 0's supervisor around the Miles process, inside the trainer container.

It runs Miles as a child process and, alongside it:

- reports progress to the progress Dict every ``HEARTBEAT_SECS`` (phase, step,
  deadline, the newest values of ``METRICS``), read from Miles' own event files
  (``{save}/events/*.jsonl``, written because ``--save`` is set);
- commits the results volume whenever a new checkpoint lands, so a save survives a
  container that is killed afterwards (a v2 volume keeps only committed writes);
- saves before the segment deadline and before the caller's lease runs out: it
  touches Miles' ``--save-trigger-sentinel``, which Miles checks after each training
  step and removes once that save is written (train_async.py), then stops Miles and
  exits cleanly, so the caller can relaunch and resume from that save;
- stops Miles on SIGTERM (app stop, cancel);
- writes the exit record (``progress.EXIT_FILE`` and the Dict) with a reason.

Nothing here changes Miles; it uses the sentinel and event files Miles already has.
"""

import collections
import datetime
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from miles_ext.platform import progress as P

SENTINEL = "/tmp/miles-save-now"
POLL_SECS = 5
STOP_GRACE_SECS = 120
DEFAULT_SAVE_SECS = 120  # until a save has been timed
DEFAULT_STEP_SECS = 300  # until a step has been timed
MAX_MARGIN_FRACTION = 0.25  # of the time left at launch; the function timeout adds GRACE beyond the deadline
TAIL_LINES = 60

# newest value of each, as reported in the heartbeat (keys as Miles logs them)
METRICS = (
    "rollout/raw_reward", "rollout/episode_raw_reward", "rollout/raw_reward_unfiltered", "rollout/truncated_ratio",
    "rollout/num_training_samples", "rollout/fully_async/avg_staleness", "rollout/fully_async/max_staleness",
    "rollout/fully_async/stale_groups_filtered", "rollout/fully_async/aborted_groups_filtered",
    "rollout/response_len/mean", "train/loss", "train/grad_norm", "train/step", "perf/step_time",
    "perf/rollout_time",
)
_EVENT_FILES = ("rollout_executor.jsonl", "actor_cell00000_rank00000.jsonl", "trainer_controller_actor.jsonl")
_OOM = ("CUDA out of memory", "OutOfMemoryError", "torch.OutOfMemoryError")
_NCCL = ("Watchdog caught collective operation timeout", "NCCL timeout", "ProcessGroupNCCL", "NCCL communicator was aborted")


def classify(returncode: int, tail: list[str]) -> str:
    """The exit reason of a Miles process that ended by itself."""
    if returncode == 0:
        return "success"
    text = "\n".join(tail)
    if any(s in text for s in _OOM):
        return "oom"
    if any(s in text for s in _NCCL) and ("timeout" in text.lower() or "timed out" in text.lower()):
        return "nccl_timeout"
    return "error"


def trigger_at(deadline: float | None, lease_expires: float | None, step_secs: float | None,
               save_secs: float | None, started: float) -> float | None:
    """When to ask for a save: the earlier of the deadline and the lease, less the time a save
    and one step take (Miles looks at the sentinel only after a step). The margin is at most
    ``MAX_MARGIN_FRACTION`` of the time the segment had, so short segments still train; the
    deadline is soft, and the function timeout (launch adds ``GRACE_SECS``) is the hard bound."""
    ends = [t for t in (deadline, lease_expires) if t is not None]
    if not ends:
        return None
    end = min(ends)
    margin = (save_secs or DEFAULT_SAVE_SECS) + (step_secs or DEFAULT_STEP_SECS)
    return end - min(margin, MAX_MARGIN_FRACTION * max(0.0, end - started))


class _EventTail:
    """Reads new lines from Miles' event files on each poll. Miles appends to them, but on a
    resume it replaces the events directory with the snapshot taken at the save it loads
    (event_logger/checkpoint.py), so a file is followed by inode and read again from the start
    when it changes; events from before ``since`` (earlier segments) are skipped. Only files on
    this container's mount are seen: on a multi-node run an actor on another node shows up
    when that node commits."""

    def __init__(self, events_dir: Path, since: float = 0.0):
        self.dir, self.since, self.offsets, self.inodes = events_dir, since, {}, {}
        self.metrics: dict = {}
        self.step: int | None = None
        self.step_ends: list[float] = []
        self.seen: set[str] = set()

    def poll(self):
        for name in _EVENT_FILES:
            path = self.dir / name
            try:
                st = path.stat()
                if self.inodes.get(name) != st.st_ino or st.st_size < self.offsets.get(name, 0):
                    self.inodes[name], self.offsets[name] = st.st_ino, 0
                with path.open("rb") as f:
                    f.seek(self.offsets.get(name, 0))
                    data = f.read()
            except OSError:
                continue
            self.seen.add(name)
            end = data.rfind(b"\n") + 1
            self.offsets[name] = self.offsets.get(name, 0) + end
            for line in data[:end].splitlines():
                self._event(line)

    def _event(self, line: bytes):
        try:
            e = json.loads(line)
            ts = datetime.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
        except (ValueError, KeyError, AttributeError):
            return
        if ts < self.since:
            return
        if e.get("type") == "metric":
            m = e.get("metrics") or {}
            for k in METRICS:
                if k in m and isinstance(m[k], (int, float)):
                    self.metrics[k] = m[k]
            if isinstance(m.get("rollout/step"), int):
                self.step = max(self.step or 0, m["rollout/step"])
        elif e.get("type") == "train_group_step_end":
            self.step_ends.append(ts)

    def step_secs(self, since: float) -> float | None:
        """Median time between training steps in this segment (steps ending after ``since``)."""
        ends = sorted(t for t in self.step_ends if t >= since)
        gaps = sorted(b - a for a, b in zip(ends, ends[1:]))
        return gaps[len(gaps) // 2] if gaps else None


class Supervisor:
    def __init__(self, run_id: str, recipe: dict, run_dir: Path, *, progress_dict, segment: int,
                 deadline: float | None, commit, image: dict, backend: str, call_id: str | None = None):
        self.run_id, self.recipe, self.run_dir = run_id, recipe, Path(run_dir)
        self.dict, self.keys = progress_dict, P.keys(run_id)
        self.segment, self.deadline, self.commit, self.image = segment, deadline, commit, image
        self.call_id = call_id
        self.backend, self.num_rollout = backend, recipe["algorithm"]["num_rollout"]
        self.tail = collections.deque(maxlen=TAIL_LINES)
        self.started = time.time()
        self.events = _EventTail(self.run_dir / "checkpoints" / "events", since=self.started)
        self.phase = "starting"
        self.sigterm = threading.Event()
        self.save_requested: tuple[float, str] | None = None  # (when, why) while a requested save is pending
        self.final_saved = False  # a requested save landed on the last step: Miles finishes by itself
        self.save_secs: float | None = None
        self.tracker = self._tracker()

    # ---- helpers ----
    def _tracker(self) -> int | None:
        try:
            return int((self.run_dir / "checkpoints" / P.TRACKER).read_text().strip())
        except (OSError, ValueError):
            return None

    def _lease_expires(self) -> float | None:
        try:
            lease = self.dict.get(self.keys["lease"])
        except Exception as e:  # a Dict outage must not stop training
            print(f"supervisor: lease read failed: {type(e).__name__}: {e}", flush=True)
            return None
        return (lease or {}).get("expires_at")

    def _put(self, key: str, value: dict):
        try:
            self.dict[key] = value
        except Exception as e:
            print(f"supervisor: progress write failed: {type(e).__name__}: {e}", flush=True)

    def heartbeat(self):
        self._put(self.keys["run"], {
            "run_id": self.run_id, "call_id": self.call_id, "phase": self.phase, "segment": self.segment,
            "step": self.events.step,
            "next_rollout_id": P.next_rollout_id(self.backend, self.tracker), "num_rollout": self.num_rollout,
            "deadline": self.deadline, "save_requested": self.save_requested, "started_at": self.started,
            "heartbeat_at": time.time(), "metrics": dict(self.events.metrics), "event_files": sorted(self.events.seen),
            "image": self.image,
        })

    def _pipe(self, proc: subprocess.Popen):
        for raw in proc.stdout:
            line = raw.decode(errors="replace")
            sys.stdout.write(line)
            self.tail.append(line.rstrip("\n"))
        sys.stdout.flush()

    def _stop(self, proc: subprocess.Popen):
        self.phase = "stopping"
        self.heartbeat()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(STOP_GRACE_SECS)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    # ---- the run ----
    def run(self, cmd: list[str], cwd: str, env: dict) -> dict:
        try:
            signal.signal(signal.SIGTERM, lambda *_: self.sigterm.set())
        except ValueError:  # not the main thread: app stop then reaches Miles directly
            print("supervisor: SIGTERM handler not installed (not the main thread)", flush=True)
        if os.path.exists(SENTINEL):
            os.remove(SENTINEL)
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        piper = threading.Thread(target=self._pipe, args=(proc,), daemon=True)
        piper.start()
        reason, last_beat = None, 0.0
        while proc.poll() is None:
            time.sleep(POLL_SECS)
            self.events.poll()
            if self.events.step is not None and self.phase == "starting":
                self.phase = "training"
            tracker = self._tracker()
            if tracker != self.tracker:  # a save landed: make it durable now
                self.tracker = tracker
                self.commit()
                print(f"supervisor: save {tracker} committed", flush=True)
            if self.sigterm.is_set():
                reason = "stopped"
                break
            if self.save_requested is None and not self.final_saved:
                expires = self._lease_expires()
                when = trigger_at(self.deadline, expires, self.events.step_secs(self.started), self.save_secs,
                                  self.started)
                if when is not None and time.time() >= when:
                    why = "lease_expired" if expires is not None and (self.deadline is None or expires < self.deadline) \
                        else "segment_end"
                    Path(SENTINEL).touch()
                    self.save_requested = (time.time(), why)
                    self.phase = "checkpointing"
                    print(f"supervisor: asked Miles to save ({why})", flush=True)
            elif self.save_requested is not None and not os.path.exists(SENTINEL):  # Miles removes it once saved
                self.save_secs = time.time() - self.save_requested[0]
                self.tracker = self._tracker()
                self.commit()
                if P.is_complete(self.backend, self.tracker, self.num_rollout):
                    print("supervisor: that save was the final step; letting Miles finish", flush=True)
                    self.final_saved, self.save_requested, self.phase = True, None, "training"
                else:
                    reason = self.save_requested[1]
                    print(f"supervisor: save {self.tracker} written; stopping Miles ({reason})", flush=True)
                    break
            if time.time() - last_beat >= P.HEARTBEAT_SECS:
                self.heartbeat()
                last_beat = time.time()
        if reason is not None:
            self._stop(proc)
        else:
            proc.wait()
        piper.join(timeout=10)
        self.events.poll()
        self.tracker = self._tracker()
        if reason is None:
            reason = classify(proc.returncode, list(self.tail))
        return self.finish(reason, proc.returncode)

    def finish(self, reason: str, returncode: int | None, note: str | None = None) -> dict:
        """Write the exit record (volume and Dict) and the final heartbeat."""
        record = {
            "run_id": self.run_id, "call_id": self.call_id, "reason": reason, "retryable": reason in P.RETRYABLE,
            "returncode": returncode,
            "segment": self.segment, "started_at": self.started, "ended_at": time.time(),
            "step": self.events.step, "tracker": self.tracker,
            "next_rollout_id": P.next_rollout_id(self.backend, self.tracker), "num_rollout": self.num_rollout,
            "complete": P.is_complete(self.backend, self.tracker, self.num_rollout),
            "metrics": dict(self.events.metrics), "image": self.image, "note": note,
            "tail": list(self.tail)[-TAIL_LINES:] if reason not in ("success",) else [],
        }
        write_exit_record(self.run_dir, record)
        self._put(self.keys["exit"], record)
        self.phase = "ended"
        self.heartbeat()
        try:  # this segment no longer holds the run (launch may start the next one)
            if (self.dict.get(self.keys["active"]) or {}).get("call_id") in (self.call_id, None):
                self.dict.pop(self.keys["active"], None)
        except Exception as e:
            print(f"supervisor: releasing the run failed: {type(e).__name__}: {e}", flush=True)
        return record


def write_exit_record(run_dir: Path, record: dict):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (P.EXIT_FILE + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    tmp.replace(run_dir / P.EXIT_FILE)
    with (run_dir / P.EXITS_LOG).open("a") as f:
        f.write(json.dumps({k: v for k, v in record.items() if k != "tail"}, default=str) + "\n")
