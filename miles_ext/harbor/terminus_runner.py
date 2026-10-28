"""Harbor's Terminus-2 run inside the task sandbox, against local processes (ext-045, see its patch page).

    python terminus_runner.py setup          # as root: install tmux with Terminus-2's own installer
    python terminus_runner.py run CONFIG     # as the agent: one Terminus-2 run; writes CONFIG["result_path"]
"""

import asyncio
import json
import os
import signal
import sys
import traceback
from pathlib import Path


class LocalEnvironment:
    """The part of Harbor's ``BaseEnvironment`` that Terminus-2 and its ``TmuxSession`` call, as local processes.

    ``user`` is ignored: the runner already runs as the user Harbor runs the agent as (root for ``setup``)."""

    default_user = None

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from harbor.environments.base import ExecResult

        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        except asyncio.TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            return ExecResult(stdout=None, stderr=f"command timed out after {timeout_sec} seconds", return_code=124)
        return ExecResult(
            stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), return_code=proc.returncode
        )

    async def is_dir(self, path, user=None) -> bool:
        return Path(path).is_dir()

    async def is_file(self, path, user=None) -> bool:
        return Path(path).is_file()


async def setup() -> int:
    from harbor.agents.terminus_2.tmux_session import TmuxSession

    env = LocalEnvironment("setup")
    session = TmuxSession(
        "setup",
        env,
        logging_path=Path("/dev/null"),
        local_asciinema_recording_path=None,
        remote_asciinema_recording_path=None,
    )
    await session._attempt_tmux_installation()
    probe = await env.exec("tmux -V")
    print(probe.stdout or probe.stderr, end="")
    return probe.return_code


async def run(config_path: str) -> int:
    from harbor.agents.terminus_2 import Terminus2
    from harbor.models.agent.context import AgentContext

    config = json.loads(Path(config_path).read_text())
    agent = Terminus2(logs_dir=Path(config["logs_dir"]), model_name=config["model_name"], **config["kwargs"])
    env = LocalEnvironment(config["session_id"])
    context = AgentContext()
    result = {}
    try:
        await agent.setup(env)
        await agent.run(config["instruction"], env, context)
    except Exception as e:
        traceback.print_exc()
        result["exception"] = {"type": type(e).__name__, "message": str(e)[-4000:]}
    result["context"] = context.model_dump(mode="json")
    path = Path(config["result_path"])
    path.with_suffix(".tmp").write_text(json.dumps(result))
    path.with_suffix(".tmp").replace(path)
    return 1 if "exception" in result else 0


if __name__ == "__main__":
    mode = sys.argv[1]
    sys.exit(asyncio.run(setup() if mode == "setup" else run(sys.argv[2])))
