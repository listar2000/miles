"""miles_ext.platform without Modal calls: guards, names, retention, schema, instant validation.

    python -m pytest miles_ext/tests/test_platform.py
"""

import importlib
import inspect
from pathlib import Path

import pytest

# modules by import_module: the package re-exports functions named like its modules (validate, teardown, schema)
C, S, SEC, T, V, W = (importlib.import_module(f"miles_ext.platform.{m}")
                      for m in ("checkpoints", "schema", "secrets", "teardown", "validate", "weights"))
from miles_ext.platform.target import Target, run_secret_name
from miles_ext.recipes import recipe as R

ROOT = Path(inspect.getfile(R)).parents[2]
DEV = {"modal_environment": "miles-dev", "app_name": "mdev-run1", "sandbox_app_prefix": "miles-mobius-",
       "results_volume": "mdev-results", "weights_volume": "miles-swe-models"}
SHA = "a" * 40


def _recipe(name: str = "qwen35-9b-swegym8-sail.yaml", **over) -> dict:
    spec = __import__("yaml").safe_load((ROOT / "miles_ext/recipes/configs" / name).read_text())
    spec["taskset"] = {k: v for k, v in spec["taskset"].items() if k not in ("tasks_dir", "prompts")}
    return R.from_dict(spec, over)


# target

def test_dev_target_accepted():
    t = Target.of(DEV)
    assert t.apply(_recipe())["sandbox"]["app_prefix"] == "miles-mobius-"
    assert Target.of(None).app_name == "miles-fork"


@pytest.mark.parametrize("field,value", [
    ("app_name", "mobius-run1"), ("app_name", "mdev-*"), ("sandbox_app_prefix", "mobius-"),
    ("sandbox_app_prefix", "miles-mobius"), ("results_volume", "mobius-results"), ("weights_volume", "x"),
    ("secret_name", "mobius-secret"), ("modal_environment", "Main env"), ("bogus", "x")])
def test_target_guard(field, value):
    with pytest.raises((ValueError, KeyError)):
        Target.of({**DEV, field: value})


def test_secret_names():
    assert run_secret_name("run1", Target.of(DEV)) == "mdev-secret-run1"
    assert run_secret_name("run1", Target.of(None)) == "miles-secret-run1"
    with pytest.raises(ValueError):
        run_secret_name("../x", Target.of(DEV))


def test_secret_values():
    sail = _recipe()
    with pytest.raises(ValueError, match="missing"):
        SEC.secret_values(sail, {"WANDB_API_KEY": "w"})
    with pytest.raises(ValueError, match="not allowed"):
        SEC.secret_values(sail, {"WANDB_API_KEY": "w", "SAIL_API_KEY": "s", "MODAL_TOKEN_SECRET": "m"})
    assert SEC.secret_values(sail, {"WANDB_API_KEY": "w", "SAIL_API_KEY": "s", "HF_TOKEN": ""}) == {
        "WANDB_API_KEY": "w", "SAIL_API_KEY": "s"}


# teardown

def test_resource_names_dev():
    names = T.resource_names(_recipe(), "run1", DEV)
    assert names["sandbox_apps"] == ["miles-mobius-run1", "miles-fork-sbx-run1"]
    assert names["gpu_app"] == "mdev-run1" and names["gpu_app_per_run"]
    assert names["secret"] == "mdev-secret-run1"
    assert not T.resource_names(_recipe(), "run1", None)["gpu_app_per_run"]  # shared miles-fork: by call id only


def test_teardown_is_idempotent(monkeypatch):
    live = {"miles-mobius-run1": 3}
    secret = {"exists": True}

    class Adapter:
        def __init__(self, env):
            assert env == "miles-dev"

        def terminate(self, name):
            R.check_own_app_name(name)
            return live.pop(name, 0)

        def live(self, name):
            return live.get(name, 0)

    stopped = []
    monkeypatch.setitem(T.ADAPTERS, "sail", Adapter)
    monkeypatch.setattr(T, "_stop_app", lambda ident, env: stopped.append((ident, env)) or "ap-1")
    monkeypatch.setattr(T, "run_secret_exists", lambda run_id, target: secret["exists"])
    monkeypatch.setattr(T, "delete_run_secret", lambda run_id, target: secret.update(exists=False))
    first = T.teardown("run1", _recipe(), DEV)
    assert first["remaining"] == [] and "sandboxes:miles-mobius-run1:3" in first["released"]
    assert stopped == [("mdev-run1", "miles-dev")]
    second = T.teardown("run1", _recipe(), DEV)
    assert second["remaining"] == [] and not any(r.startswith(("sandboxes", "secret")) for r in second["released"])


def test_unknown_provider_has_no_adapter():
    with pytest.raises(ValueError, match="no teardown adapter"):
        T.sandbox_adapter("daytona", None)


# checkpoints

@pytest.mark.parametrize("run_id,step,kind", [
    ("../x", 1, "full"), ("run1/..", 1, "full"), ("run1", -1, "full"), ("run1", 1.5, "full"),
    ("run1", True, "full"), ("run1", 1, "trials"), ("", 1, "hf"), ("run1", 10**7, "hf")])
def test_checkpoint_path_guard(run_id, step, kind):
    with pytest.raises(ValueError):
        C.checkpoint_path(run_id, step, kind)


def test_checkpoint_paths():
    assert C.checkpoint_path("run1", 20, "full") == "run1/checkpoints/iter_0000020"
    assert C.checkpoint_path("run1", 20, "hf") == "run1/hf/iter_0000020"
    assert C.hf_export_template("run1") == "/results/run1/hf/iter_{rollout_id:07d}"


def _saves(full, hf):
    return [{"step": s, "kind": "full"} for s in full] + [{"step": s, "kind": "hf"} for s in hf]


def _steps(plan):
    return sorted((p["kind"], p["step"]) for p in plan)


def test_retention_while_training_keeps_newest_and_resume_point():
    plan = C.retention_plan(_saves([10, 20, 30], [10, 20, 30]), keep=1, keep_final_full=True, final=False, latest=30)
    assert _steps(plan) == [("full", 10), ("full", 20)]
    plan = C.retention_plan(_saves([10, 20, 30], []), keep=2, keep_final_full=True, final=False, latest=10)
    assert _steps(plan) == []  # 20, 30 newest; 10 is the resume point


def test_retention_final_knob_and_pins():
    saves = _saves([10, 20], [10, 20])
    assert _steps(C.retention_plan(saves, keep=1, keep_final_full=True, final=True, latest=20)) == [
        ("full", 10), ("hf", 10)]
    assert _steps(C.retention_plan(saves, keep=1, keep_final_full=False, final=True, latest=20)) == [
        ("full", 10), ("full", 20), ("hf", 10)]
    assert _steps(C.retention_plan(saves, keep=1, keep_final_full=False, final=True, latest=20, pinned=(10,))) == [
        ("full", 20)]
    # FSDP before finalize_hf: no HF export yet, so the final full save stays whatever the knob says
    assert _steps(C.retention_plan(_saves([10, 20], []), keep=1, keep_final_full=False, final=True, latest=20)) == [
        ("full", 10)]


def test_retention_local(tmp_path):
    run = tmp_path / "run1"
    for sub, steps in (("checkpoints", (10, 20)), ("hf", (10, 20))):
        for s in steps:
            (run / sub / C.iter_dir(s)).mkdir(parents=True)
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("20")
    (run / "checkpoints" / "rollout").mkdir()
    (run / "trials").mkdir()
    C.apply_retention_local(run, keep=1, keep_final_full=False, final=True)
    left = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_dir())
    assert left == ["checkpoints", "checkpoints/rollout", "hf", "hf/iter_0000020", "trials"]


def test_delete_refuses_resume_point(monkeypatch):
    removed = []
    monkeypatch.setattr(C, "_volume", lambda target: object())
    monkeypatch.setattr(C, "_latest", lambda volume, run_id: 20)
    monkeypatch.setattr(C, "_remove", lambda volume, path: removed.append(path) or {"path": path, "deleted": True})
    with pytest.raises(RuntimeError, match="resumes from"):
        C.delete_checkpoint("run1", 20, "full", DEV)
    C.delete_checkpoint("run1", 20, "hf", DEV)
    C.delete_checkpoint("run1", 10, "full", DEV)
    assert removed == ["run1/hf/iter_0000020", "run1/checkpoints/iter_0000010"]


# weights

def test_weights_key():
    assert W.weights_key("Qwen/Qwen3.5-9B", SHA) == f"Qwen--Qwen3.5-9B/{SHA}"
    for repo, rev in (("Qwen", SHA), ("Qwen/../x", SHA), ("Qwen/Qwen3.5-9B", "main"), ("a/b", SHA[:-1])):
        with pytest.raises(ValueError):
            W.weights_key(repo, rev)


@pytest.mark.parametrize("key", ["Qwen3.5-9B", "../x", f"Qwen--Qwen3.5-9B/{SHA}/x", f"Qwen--Qwen3.5-9B/../{SHA}",
                                 "", f"/Qwen--a/{SHA}"])
def test_delete_weights_refuses(key, monkeypatch):
    monkeypatch.setattr(W, "_locks", lambda target: {})
    monkeypatch.setattr(W, "_volume", lambda target: pytest.fail("reached the volume"))
    with pytest.raises(ValueError):
        W.delete_weights(key, DEV)


def test_delete_weights_refuses_locked(monkeypatch):
    monkeypatch.setattr(W, "_locks", lambda target: {f"Qwen--Qwen3.5-9B/{SHA}": {"owner": "x"}})
    with pytest.raises(RuntimeError, match="being prepared"):
        W.delete_weights(f"Qwen--Qwen3.5-9B/{SHA}", DEV)


def test_model_dir_follows_revision():
    assert str(R.model_dir(_recipe())) == "/models/Qwen3.5-9B"
    assert str(R.model_dir(_recipe(model={"revision": SHA}))) == f"/models/Qwen--Qwen3.5-9B/{SHA}"


# schema and instant validation

def test_schema_shape():
    s = S.schema()
    assert s["api_version"] == "miles-rl/v1"
    props = s["json_schema"]["properties"]
    assert props["checkpoint"]["properties"]["keep_final_full"] == {"type": "boolean"}
    assert "qwen3.5-9b" in s["presets"] and "--save" in s["reserved_flags"]


def test_instant_errors():
    assert V.instant_errors(_recipe(), "run1", DEV) == []
    errs = V.instant_errors(_recipe(extra_args=["--save", "/x", "--wandb-project=p"]), "run1", DEV)
    assert [e["path"] for e in errs] == ["extra_args", "extra_args"]
    errs = V.instant_errors(_recipe(checkpoint={"save_hf": True}), "run1", DEV)
    assert errs[0]["path"] == "checkpoint.save_hf"
    errs = V.instant_errors(_recipe(), "bad id", DEV)
    assert errs[0]["path"] == "run_id"
    assert V.instant_errors(_recipe(), "run1", {**DEV, "app_name": "x"})[0]["path"] == "target"


def test_validate_shallow_resolves_argv():
    out = V.validate(_recipe("qwen35-9b-swegym8-megatron-2node.yaml", checkpoint={"save_hf": True}), "run1", DEV,
                     deep=False)
    assert out["ok"] and out["argv"][0] == "train_async.py"
    assert out["argv"][out["argv"].index("--save-hf") + 1] == "/results/run1/hf/iter_{rollout_id:07d}"
    assert out["resolved"]["sandbox"]["app_prefix"] == "miles-mobius-"


def test_launcher_final_retention_switch():
    src = (ROOT / "miles_ext/launch/modal_app.py").read_text()
    assert '"0" if TARGET.secret_name else "1"' in src  # platform runs (per-run secret) retain themselves
    assert "if FINAL_RETENTION:" in src
