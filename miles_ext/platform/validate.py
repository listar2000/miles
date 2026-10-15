"""R2: validate a recipe the way a run would meet it, and return structured results.

``validate(recipe, run_id, target)`` is the dry run, and the first step of every
real run:

1. instant checks, here: recipe keys, ranges and layouts (``recipe.validate``),
   names against the target, the secret keys the run needs;
2. the deep check, on Modal in the fork image: fork markers, the Megatron model
   imports, the Ray head lookup, a real Harbor trial config with the sandbox
   provider's preflight and resource policies, and Miles' ``parse_args`` on the
   exact argv. FSDP runs it on CPU; Megatron on one GPU, because its argument
   parser loads ``libcuda.so.1`` (``libcuda_stub=True`` tries the CUDA toolkit's
   stub on CPU instead).

Errors come back as ``{path, message}``: ``path`` is the dotted recipe key when
known, ``argv`` for Miles' parser, otherwise the check's name. Inputs are read
where a run reads them: ``<results_volume>/<run_id>/inputs/{tasks/,prompts.jsonl}``
(the platform copies the taskset there first), or ``/inputs`` when the caller
mounts them itself (the test launcher).
"""

import json
import os
import re
import subprocess
import sys
import traceback

from miles_ext.platform.schema import reserved_flag_errors
from miles_ext.platform.target import Target
from miles_ext.recipes import recipe as R

GPU_FOR_MEGATRON = "H100"
LIBCUDA_STUB = "/usr/local/cuda/lib64/stubs/libcuda.so"
PARSED_KEYS = ["train_backend", "fully_async", "rollout_num_gpus", "actor_num_nodes", "actor_num_gpus_per_node",
               "global_batch_size", "num_rollout", "save_interval", "save", "save_hf", "tito_model",
               "sglang_tool_call_parser", "wandb_project", "wandb_team"]


def _path_of(message: str, default: str) -> str:
    m = re.match(r"([a-z_]+(?:\.[a-z_]+)+)\b", message)
    return m.group(1) if m else default


def instant_errors(recipe: dict, run_id: str, target: Target | dict | None) -> list[dict]:
    """Pure checks, no Modal and no image: what a form or CLI can run as you type."""
    errors = []
    try:
        target = Target.of(target)
        recipe = target.apply(recipe)
    except (ValueError, KeyError) as e:
        return [{"path": "target", "message": str(e)}]
    try:
        R.validate(recipe, local_inputs=False)
    except (ValueError, KeyError) as e:
        errors.append({"path": _path_of(str(e), "recipe"), "message": str(e)})
    try:
        R.sandbox_app_names_for_cleanup(recipe, run_id)
    except ValueError as e:
        errors.append({"path": "run_id", "message": str(e)})
    errors += reserved_flag_errors(recipe)
    return errors


def validate(recipe: dict, run_id: str, target: Target | dict | None = None, *, libcuda_stub: bool = False,
             deep: bool = True, secret_name: str | None = None) -> dict:
    """``{ok, errors, argv, resolved, trial_config, parsed, ran_on}``. ``deep=False`` runs only the instant checks.

    The deep check gets the run's secret (``secret_name``, else ``target.secret_name``) so the
    sandbox provider's preflight sees its key (Sail: ``SAIL_API_KEY``)."""
    target = Target.of(target) if not isinstance(target, Target) else target
    errors = instant_errors(recipe, run_id, target)
    out = {"ok": False, "errors": errors, "argv": None, "resolved": None, "trial_config": None, "parsed": None,
           "ran_on": None}
    if errors:
        return out
    resolved = target.apply(recipe)
    out["resolved"] = resolved
    out["argv"] = [R.entrypoint(resolved), *R.miles_args(resolved, run_id, resume=False)]
    if not deep:
        out["ok"] = True
        return out

    import modal

    from miles_ext.platform import functions

    megatron = R.backend(resolved) == "megatron"
    gpu = GPU_FOR_MEGATRON if megatron and not libcuda_stub else None
    volumes = {
        R.MODELS: modal.Volume.from_name(target.weights_volume, environment_name=target.modal_environment),
        R.RESULTS: modal.Volume.from_name(target.results_volume, environment_name=target.modal_environment),
    }
    secret_name = secret_name or target.secret_name
    secrets = [modal.Secret.from_name(secret_name, environment_name=target.modal_environment)] if secret_name else []
    fn = functions.deep_check.with_options(gpu=gpu, volumes=volumes, secrets=secrets)
    with functions.app.run(environment_name=target.modal_environment):
        remote = fn.remote(resolved, run_id, libcuda_stub)
    out.update({k: remote[k] for k in ("trial_config", "parsed")}, ran_on=gpu or "cpu")
    out["errors"] = remote["errors"]
    out["ok"] = not remote["errors"]
    return out


def _inputs(run_id: str) -> None:
    """Make /inputs the run's inputs on the results volume, unless the caller mounted them."""
    inputs = R.TASKS.parent
    if inputs.exists():
        return
    src = R.RESULTS / run_id / "inputs"
    if not (src / R.TASKS.name).is_dir() or not (src / R.PROMPTS.name).is_file():
        raise FileNotFoundError(f"no taskset at {src}/{{{R.TASKS.name}/,{R.PROMPTS.name}}} (copy it there first)")
    inputs.symlink_to(src)


def _parse_args(recipe: dict, run_id: str, libcuda_stub: bool) -> dict:
    """Miles' parse_args on the exact argv, in a subprocess (argparse exits; asserts raise)."""
    from miles_ext import stack

    env = {**os.environ, **R.trainer_env(recipe)}
    if libcuda_stub:
        stub_dir = "/tmp/libcuda-stub"
        os.makedirs(stub_dir, exist_ok=True)
        link = os.path.join(stub_dir, "libcuda.so.1")
        if not os.path.exists(link):
            os.symlink(LIBCUDA_STUB, link)
        env["LD_LIBRARY_PATH"] = f"{stub_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    code = ("import json, sys; sys.argv = json.loads(sys.stdin.read()); "
            "from miles.utils.arguments import parse_args; a = parse_args(); "
            f"print('PARSED ' + json.dumps({{k: getattr(a, k, None) for k in {PARSED_KEYS!r}}}, default=str))")
    argv = [R.entrypoint(recipe), *R.miles_args(recipe, run_id, resume=False)]
    proc = subprocess.run([sys.executable, "-c", code], input=json.dumps(argv), capture_output=True, text=True,
                          cwd=stack.MILES_SRC, env=env, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("PARSED "):
            return json.loads(line[7:])
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
    raise RuntimeError(f"Miles parse_args failed (exit {proc.returncode}):\n{tail}")


def run_deep_check(recipe: dict, run_id: str, *, libcuda_stub: bool = False) -> dict:
    """Container side (fork image). Each step records its error and the independent ones still run."""
    from miles_ext import stack

    errors: list[dict] = []
    result = {"errors": errors, "trial_config": None, "parsed": None,
              "image_commit": os.environ.get("MILES_FORK_COMMIT"), "stack": os.environ.get("MILES_FORK_RELEASE")}

    def step(name: str, fn, path: str | None = None):
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001 (SystemExit from argparse included)
            message = str(e) or type(e).__name__
            if not isinstance(e, (RuntimeError, ValueError, FileNotFoundError)):
                message += "\n" + "".join(traceback.format_exception_only(type(e), e)).strip()
            errors.append({"path": path or name, "message": message})
            return None

    os.environ.update({**R.harbor_env(recipe, run_id), **R.trainer_env(recipe)})
    os.environ.setdefault("MILES_SESSION_PUBLIC_URL", "https://tunnel.invalid/token")
    step("inputs", lambda: _inputs(run_id), "taskset")
    step("fork_markers", lambda: subprocess.run(
        [sys.executable, str(stack.MILES_SRC / "fork/tools/ledger.py"), "markers"], cwd=stack.MILES_SRC, check=True,
        capture_output=True, text=True))
    if not (R.model_dir(recipe) / "config.json").exists():
        errors.append({"path": "model", "message": f"no weights at {R.model_dir(recipe)} (run prepare_weights)"})
    if R.backend(recipe) == "megatron":
        step("megatron_imports", lambda: subprocess.run(
            [sys.executable, "-c", "import megatron.training.training, megatron.post_training.checkpointing, "
             "miles_plugins.mbridge"], cwd=stack.MILES_SRC, check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:/root/Megatron-LM"}),
            "model.backend")

    def trial():
        from harbor.environments.factory import EnvironmentFactory

        from miles_ext.harbor import agent_function

        tasks = sorted(os.listdir(R.TASKS))
        if not tasks:
            raise ValueError(f"taskset: no tasks in {R.TASKS}")
        config = agent_function.build_trial_config(
            {"instance_id": tasks[0]}, "http://10.0.0.1:30003/sessions/abc/v1", {"temperature": 1.0})
        env_cfg = config.environment
        EnvironmentFactory.run_preflight(env_cfg.type, env_cfg.import_path)
        EnvironmentFactory.validate_resource_policies(env_cfg)
        rows = len(R.PROMPTS.read_text().splitlines())
        return {"agent": config.agent.import_path or str(config.agent.name), "environment": str(env_cfg.type),
                "environment_import_path": env_cfg.import_path, "environment_kwargs": env_cfg.kwargs,
                "tasks": len(tasks), "prompt_rows": rows}

    result["trial_config"] = step("trial_config", trial, "sandbox")
    result["parsed"] = step("parse_args", lambda: _parse_args(recipe, run_id, libcuda_stub), "argv")
    return result
