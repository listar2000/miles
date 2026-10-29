"""`taskset.harness: terminus-2-sandbox`: Harbor's Terminus-2 inside the task sandbox (ext-045, see its patch page)."""

import asyncio
import functools
import importlib.metadata
import io
import json
import shlex
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from miles_ext.harnesses import common
from miles_ext.harnesses.terminus import runner as terminus_runner

NAME = "terminus-2-sandbox"
PREFIX = "/opt/miles-terminus"
PYTHON = f"{PREFIX}/venv/bin/python"
RUNNER = f"{PREFIX}/src/{Path(terminus_runner.__file__).name}"
BUNDLE = "/tmp/miles-terminus-bundle.tar.gz"
RUN_CONFIG = "/tmp/miles-terminus-run.json"
RESULT = (EnvironmentPaths.agent_dir / "terminus-2-sandbox.result.json").as_posix()
OUTPUT = (EnvironmentPaths.agent_dir / "terminus-2-sandbox.txt").as_posix()
# distribution -> module: what harbor.agents.terminus_2 imports outside the standard library and Harbor itself
# (pyyaml: its task-skills listing); a Harbor bump that adds one fails setup at the import check below
RUNTIME_DEPS = {
    "litellm": "litellm",
    "pydantic": "pydantic",
    "tenacity": "tenacity",
    "shortuuid": "shortuuid",
    "toml": "toml",
    "pyyaml": "yaml",
}
UV_VERSION = "0.7.13"  # the uv Harbor's mini-swe-agent install uses


@functools.cache
def bundle() -> bytes:
    """The image's Harbor (package and dist-info: ``harbor.__version__`` reads the metadata) and the runner, as a
    tar.gz to unpack in ``PREFIX/src``."""
    dist = importlib.metadata.distribution("harbor")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in dist.files or ():
            if f.parts[0] == "harbor" or f.parts[0].endswith(".dist-info"):
                if f.suffix != ".pyc":
                    tar.add(dist.locate_file(f), arcname=str(f))
        tar.add(terminus_runner.__file__, arcname=Path(RUNNER).name)
    return buf.getvalue()


def install_script() -> str:
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_DEPS}
    pins = " ".join(f"{name}=={v}" for name, v in versions.items())
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    # a venv already in the image is used only when it has exactly the image's Python minor and pins
    probe = (
        f"import sys, importlib.metadata as m; assert sys.version_info[:2] == {sys.version_info[:2]}; "
        f"assert all(m.version(d) == v for d, v in {versions!r}.items())"
    )
    return f"""set -euo pipefail
mkdir -p {PREFIX}/src && tar -xzf {BUNDLE} -C {PREFIX}/src
if ! {PYTHON} -c {shlex.quote(probe)} 2>/dev/null; then
  if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl;
    elif command -v apk >/dev/null 2>&1; then apk add --no-cache curl bash;
    elif command -v dnf >/dev/null 2>&1; then dnf install -y curl;
    elif command -v yum >/dev/null 2>&1; then yum install -y curl; fi
  fi
  export PATH="{PREFIX}/bin:$PATH" UV_PYTHON_INSTALL_DIR={PREFIX}/python UV_CACHE_DIR={PREFIX}/uv-cache
  command -v uv >/dev/null 2>&1 ||
    curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | env UV_UNMANAGED_INSTALL={PREFIX}/bin sh
  uv venv --allow-existing --python {python} {PREFIX}/venv
  uv pip install --python {PYTHON} {pins}
  rm -rf {PREFIX}/uv-cache
fi
site="$({PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo {PREFIX}/src > "$site/miles-terminus.pth"
{PYTHON} -c "import harbor.agents.terminus_2"
{PYTHON} {RUNNER} setup
chmod -R a+rX {PREFIX}
rm -f {BUNDLE}
"""


def raised(kind: str, message: str) -> Exception:
    """The sandbox's exception under its own class name, which is what Harbor records and the agent function maps
    to an exit status (e.g. ``SingleTurnMaxSeqLenExceededError`` -> SequenceLengthLimitExceeded)."""
    from harbor.llms import base as llm_base

    cls = getattr(llm_base, kind, None)
    if not (isinstance(cls, type) and issubclass(cls, Exception)):
        cls = type(kind, (RuntimeError,), {})
    return cls(message)


class SandboxTerminus2(common.PublicSetup, BaseInstalledAgent):
    SUPPORTS_ATIF = True

    def __init__(
        self,
        *args,
        terminus: dict[str, Any],
        model_info: dict[str, Any] | None = None,
        max_seq_len: int | None = None,
        compact_at_tokens: int | None = None,
        **kwargs,
    ):
        """``terminus``: Terminus2's constructor kwargs; ``model_info`` joins them (its ``max_input_tokens`` is
        Terminus' context limit). ``max_seq_len`` is what Miles passes every harness; Terminus2 does not use it."""
        super().__init__(*args, **kwargs)
        self._terminus = {**terminus, "record_terminal_session": False}
        if model_info:
            self._terminus["model_info"] = model_info
        if compact_at_tokens:
            limit = int((model_info or {}).get("max_input_tokens") or 0)
            if not 0 < compact_at_tokens < limit:
                raise ValueError(f"compact_at_tokens {compact_at_tokens} must be below the context limit {limit}")
            self._terminus.update(enable_summarize=True, proactive_summarization_threshold=limit - compact_at_tokens)

    @staticmethod
    def name() -> str:
        return NAME

    async def install(self, environment: BaseEnvironment) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as f:
            f.write(bundle())
            f.flush()
            await environment.upload_file(f.name, BUNDLE)
        # safe to repeat: each step skips what is already in place (as mini-swe-agent's install retries)
        for attempt in range(1, common.INSTALL_ATTEMPTS + 1):
            try:
                return await self.exec_as_root(environment, command=install_script())
            except NonZeroAgentExitCodeError as e:
                if attempt == common.INSTALL_ATTEMPTS:
                    raise
                self.logger.warning(f"harness install attempt {attempt} failed, retrying: {str(e)[:200]}")
                await asyncio.sleep(common.INSTALL_BACKOFF_SECS * attempt)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        config = {
            "model_name": self.model_name,
            "kwargs": self._terminus,
            "instruction": self.render_instruction(instruction),
            "session_id": environment.session_id,
            "logs_dir": EnvironmentPaths.agent_dir.as_posix(),
            "result_path": RESULT,
        }
        # a file, not argv: an agent command like `pkill -f <text from the task>` must not match the runner
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(config, f)
            f.flush()
            await environment.upload_file(f.name, RUN_CONFIG)
        await self.exec_as_root(environment, command=f"chmod a+r {RUN_CONFIG}")
        run = await environment.exec(
            command=f"set -o pipefail; {PYTHON} {RUNNER} run {RUN_CONFIG} 2>&1 | tee {OUTPUT}"
        )
        read = await environment.exec(command=f"cat {shlex.quote(RESULT)}")
        try:
            result = json.loads(read.stdout or "")
        except ValueError:
            raise NonZeroAgentExitCodeError(
                f"Terminus-2 runner exited {run.return_code} without a result: {self._truncate_output(run.stdout)}"
            ) from None
        for field, value in result.get("context", {}).items():
            setattr(context, field, value)
        if exc := result.get("exception"):
            raise raised(exc["type"], exc["message"])
