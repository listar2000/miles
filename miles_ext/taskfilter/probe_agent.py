"""Harbor's oracle agent plus a mini-swe-agent install probe in its setup.

The probe runs the training harness's own setup (``miles_ext.harbor.sandbox_agent``
``SandboxMiniSweAgent.setup``: Harbor's install, then a version check) with
public network for setup only, as training does since ext-009, and restores
the task's policy before the oracle runs (failing to restore aborts the trial,
so the solution never runs online). A ``mini-swe-agent --help`` smoke then
runs as the agent user under the task's own policy. The record goes to
``<agent logs>/install_probe.json``; setup never raises for an install
failure, so the oracle and verifier always run.

Harbor passes ``task_dir``/``trial_paths``/``agent_timeout_sec`` only to an
agent named ``oracle``; this import-path agent gets ``task_dir`` and
``agent_timeout_sec`` through ``AgentConfig.kwargs`` and derives the trial
paths from its logs dir (``<trial_dir>/agent``).
"""

import asyncio
import json
import time
from pathlib import Path

from harbor.agents.oracle import OracleAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from miles_ext.harbor.sandbox_agent import _UV_PATH, SandboxMiniSweAgent
from miles_ext.stack import MINI_SWE_AGENT_VERSION  # the version training installs

PROBE_FILE = "install_probe.json"


def _tail(text, n: int = 1200) -> str:
    text = str(text or "")
    return text if len(text) <= n else "..." + text[-n:]


class ProbeOracleAgent(OracleAgent):
    def __init__(self, logs_dir: Path, task_dir: str | Path, install_timeout_sec: float = 1500,
                 harness_version: str = MINI_SWE_AGENT_VERSION, **kwargs):
        logs_dir = Path(logs_dir)
        super().__init__(logs_dir=logs_dir, task_dir=Path(task_dir), trial_paths=TrialPaths(logs_dir.parent), **kwargs)
        self._install_timeout_sec = install_timeout_sec
        self._harness = SandboxMiniSweAgent(logs_dir=logs_dir / "probe", version=harness_version)

    @staticmethod
    def name() -> str:
        return "oracle-install-probe"

    async def setup(self, environment: BaseEnvironment) -> None:
        baseline = environment.network_policy
        public = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
        record = {"task_network": baseline.network_mode.value, "install_ok": False, "install_s": None,
                  "smoke_ok": False, "error": None}
        start = time.monotonic()
        try:
            # SandboxMiniSweAgent.setup opens the network too, but the wait_for below can cancel it
            # before its finally restores the policy; this outer toggle restores it either way
            if baseline != public:
                await environment.set_network_policy(public)
            await asyncio.wait_for(self._harness.setup(environment), timeout=self._install_timeout_sec)
            record["install_ok"] = True
        except asyncio.TimeoutError:
            record["error"] = f"install timed out after {self._install_timeout_sec}s"
        except Exception as e:  # recorded, never raised: the oracle must still run
            record["error"] = f"{type(e).__name__}: {_tail(e)}"
        finally:
            record["install_s"] = round(time.monotonic() - start, 1)
            if environment.network_policy != baseline:
                await environment.set_network_policy(baseline)  # raising here aborts the trial on purpose
        if record["install_ok"]:
            try:
                smoke = await self._harness.exec_as_agent(
                    environment, command=_UV_PATH + "mini-swe-agent --help >/dev/null && echo SMOKE_OK")
                record["smoke_ok"] = "SMOKE_OK" in (smoke.stdout or "")
            except Exception as e:
                record["error"] = f"smoke: {type(e).__name__}: {_tail(e)}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / PROBE_FILE).write_text(json.dumps(record))
