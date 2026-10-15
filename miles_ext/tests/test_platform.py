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
    assert names["progress_dict"] == "mdev-progress"
    assert names["progress_keys"] == ["active:run1", "commit:run1", "exit:run1", "lease:run1", "run:run1"]
    assert not T.resource_names(_recipe(), "run1", None)["gpu_app_per_run"]  # shared miles-fork: by call id only



RUN = "0bebf81d-61f8-5339-a02c-cfa4cc4ec536"
OPTION_A = {"modal_environment": "rl-dev", "name_id": "0bebf81d", "app_name": "rl-dev-0bebf81d",
            "results_volume": "rl-dev-results", "weights_volume": "rl-dev-weights"}


def test_resource_names_option_a():
    names = T.resource_names(_recipe(), RUN, OPTION_A)
    assert names["gpu_app"] == "rl-dev-0bebf81d" and names["gpu_app_per_run"]
    assert names["sandbox_apps"] == ["rl-dev-0bebf81d-sbx"]
    assert names["secret"] == "rl-dev-0bebf81d-secret"
    assert names["progress_dict"] == "rl-dev-progress"
    assert names["results_dir"] == f"{RUN}/" and f"run:{RUN}" in names["progress_keys"]
    recipe = Target.of(OPTION_A).apply(_recipe())
    R.validate(recipe, local_inputs=False)
    assert R.harbor_env(recipe, RUN)  # the sandbox env builds with the new app name


@pytest.mark.parametrize("change", [
    {"name_id": None},                       # rl- names need a name id
    {"name_id": "0BEBF81D"},
    {"app_name": "rl-dev-0bebf81e"},         # app name must end in the name id
    {"app_name": "rl-0bebf81d"},             # and carry the environment
    {"app_name": "mdev-run1"},               # a name id belongs to rl- names only
])
def test_option_a_guard(change):
    with pytest.raises(ValueError):
        Target.of({**OPTION_A, **change})


def test_option_a_helper_and_queue_names():
    from miles_ext.platform.target import completion_queue_name, helper_app_name
    t = Target.of(OPTION_A)
    assert helper_app_name(t) == "rl-dev-helpers" and completion_queue_name(t) == "rl-dev-completion"
    assert helper_app_name(Target.of(DEV)) == "miles-platform"  # dev keeps its names until cleanup
    assert completion_queue_name(Target.of(DEV)) == "miles-fork-completion"


def test_wandb_project_and_name():
    recipe = R.from_dict({**_recipe(), "tracking": {**_recipe()["tracking"],
                          "wandb_project": "miles-mobius-dev-test-project", "wandb_name": "0bebf81d · swe-gym overfit"}})
    R.validate(recipe, local_inputs=False)
    assert R.wandb_project(recipe, RUN) == "miles-mobius-dev-test-project"
    bad = {**recipe, "tracking": {**recipe["tracking"], "wandb_name": 7}}
    assert any(e["path"] == "tracking.wandb_name" for e in R.field_errors(bad))


def test_staging_legacy_prefix_accepted():
    t = Target.of({**DEV, "modal_environment": "miles-staging", "app_name": "mstg-run1",
                   "sandbox_app_prefix": "mstg-", "results_volume": "mstg-results"})
    assert run_secret_name("run1", t) == "mstg-secret-run1"

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
    progress = {"run:run1": {}, "lease:run1": {}, "active:run1": {"call_id": "fc-live"}, "run:run2": {}}
    monkeypatch.setattr(T, "progress_dict", lambda target: progress)
    monkeypatch.setattr(T, "delete_keys", lambda run_id, target: [
        k for k in (f"run:{run_id}", f"lease:{run_id}", f"active:{run_id}") if progress.pop(k, None) is not None])
    cancelled, running, app_state = [], {"fc-live"}, {"ap-eph": "APP_STATE_EPHEMERAL"}

    class Call:
        def __init__(self, call_id):
            self.call_id = call_id

        def cancel(self, terminate_containers):
            cancelled.append(self.call_id)

    monkeypatch.setattr(T.modal.FunctionCall, "from_id", Call)
    monkeypatch.setattr(T, "call_running", lambda call_id: call_id in running)
    monkeypatch.setattr(T, "_app_live", lambda ident, env: app_state.get(ident))
    # the trainer's app is ephemeral: found by listing the environment's apps by exact name
    listed = []
    monkeypatch.setattr(T, "_live_app_ids", lambda name, env: listed.append((name, env)) or list(app_state))
    first = T.teardown("run1", _recipe(), DEV)
    assert cancelled == ["fc-live"]  # from active:run1, though the caller passed no call id
    assert "progress:run:run1" in first["released"] and progress == {"run:run2": {}}
    assert "sandboxes:miles-mobius-run1:3" in first["released"]
    assert first["remaining"] == ["call:fc-live", "app:ap-eph:APP_STATE_EPHEMERAL"]  # not stopped yet
    assert stopped == [("ap-eph", "miles-dev")] and listed[0] == ("mdev-run1", "miles-dev")
    running.clear(), app_state.clear()
    second = T.teardown("run1", _recipe(), DEV, call_id="fc-live")
    assert second["remaining"] == [] and not any(r.startswith(("sandboxes", "secret")) for r in second["released"])


def test_modal_internals_teardown_uses_exist():
    """teardown stops apps through Modal internals (the SDK has no public app stop): an SDK
    upgrade that moves them fails here, not in a run's teardown."""
    from modal._utils.async_utils import synchronizer
    from modal.cli.app import resolve_app_identifier
    from modal.client import _Client
    from modal_proto import api_pb2

    assert callable(resolve_app_identifier) and hasattr(_Client, "from_env") and synchronizer
    assert api_pb2.AppStopRequest and api_pb2.APP_STOP_SOURCE_CLI is not None
    assert {api_pb2.APP_STATE_STOPPED, api_pb2.APP_STATE_STOPPING} and api_pb2.AppState.Name(api_pb2.APP_STATE_STOPPED)
    assert api_pb2.AppListRequest(environment_name="e") and {"app_id", "description", "state"} <= {
        f.name for f in api_pb2.AppListResponse.AppListItem.DESCRIPTOR.fields}


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
            (run / sub / C.iter_dir(s) / "model").mkdir(parents=True)
    (run / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("20")
    (run / "checkpoints" / "rollout").mkdir()
    (run / "trials").mkdir()
    C.apply_retention_local(run, keep=1, keep_final_full=False, final=True)
    left = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_dir())
    assert left == ["checkpoints", "checkpoints/rollout", "hf", "hf/iter_0000020", "hf/iter_0000020/model", "trials"]


def test_delete_refuses_resume_point(monkeypatch):
    removed = []
    monkeypatch.setattr(C, "_volume", lambda target: object())
    monkeypatch.setattr(C, "read_tracker_volume", lambda volume, run_id: 20)
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


def test_weights_lock_refresh_keeps_owner_fresh():
    import threading
    import time

    locks = {"k": {"owner": "me", "ts": 0}}
    stop = threading.Event()
    t = threading.Thread(target=W._refresh_lock, args=(locks, "k", "me", stop, 0.05))
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(1)
    assert time.time() - locks["k"]["ts"] < 1
    locks["k"] = {"owner": "other", "ts": 0}  # taken over: the old owner stops writing
    W._refresh_lock(locks, "k", "me", threading.Event(), 0.01)
    assert locks["k"] == {"owner": "other", "ts": 0}


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
    assert 'if FINAL_RETENTION and record["reason"] == "success":' in src


# r6: named training knobs, field checks, reserved flags

def _argv(recipe: dict, run_id: str = "run1") -> list[str]:
    return R.miles_args(recipe, run_id, resume=False)


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_named_knobs_defaults_match_r5_argv():
    argv = _argv(_recipe())
    assert _value(argv, "--advantage-estimator") == "grpo"
    assert _value(argv, "--pause-generation-mode") == "in_place"
    assert _value(argv, "--async-unused-samples-handler") == "drop"
    assert _value(argv, "--entropy-coef") == "0.0" and _value(argv, "--kl-coef") == "0"
    assert int(_value(argv, "--async-max-concurrent-samples")) > 0
    for flag in ("--use-kl-loss", "--seed", "--over-sampling-batch-size", "--ref-load"):
        assert flag not in argv


def test_named_knobs_emit_flags():
    r = _recipe(algorithm={"advantage_estimator": "gspo", "kl_loss_coef": 0.01, "entropy_coef": 0.001, "seed": 7},
                **{"async": {"over_sampling_batch_size": 64, "max_concurrent_samples": 48, "unused_samples": "retry",
                             "pause_generation": "retract"}})
    assert V.instant_errors(r, "run1", DEV) == []
    argv = _argv(r)
    assert _value(argv, "--advantage-estimator") == "gspo"
    assert "--use-kl-loss" in argv and _value(argv, "--kl-loss-coef") == "0.01"
    assert _value(argv, "--ref-load") == str(R.model_dir(r))  # FSDP loads the reference copy
    assert _value(argv, "--seed") == "7" and _value(argv, "--entropy-coef") == "0.001"
    assert _value(argv, "--over-sampling-batch-size") == "64"
    assert _value(argv, "--async-max-concurrent-samples") == "48"
    assert _value(argv, "--async-unused-samples-handler") == "retry"
    assert _value(argv, "--pause-generation-mode") == "retract"


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reinforce_pp_normalizes_advantages(estimator):
    """Miles asserts --normalize-advantages for these estimators (arguments.py, miles_validate_args)."""
    assert "--normalize-advantages" in _argv(_recipe(algorithm={"advantage_estimator": estimator}))
    assert "--normalize-advantages" not in _argv(_recipe())


def test_megatron_kl_loss_keeps_one_ref_load():
    argv = _argv(_recipe("qwen35-9b-swegym8-megatron-2node.yaml", algorithm={"kl_loss_coef": 0.01}))
    assert argv.count("--ref-load") == 1 and "--use-kl-loss" in argv


@pytest.mark.parametrize("over, path", [
    ({"algorithm": {"advantage_estimator": "ppo"}}, "algorithm.advantage_estimator"),
    ({"algorithm": {"kl_loss_coef": -1}}, "algorithm.kl_loss_coef"),
    ({"algorithm": {"lr": 0}}, "algorithm.lr"),
    ({"algorithm": {"num_rollout": "30"}}, "algorithm.num_rollout"),
    ({"algorithm": {"seed": 1.5}}, "algorithm.seed"),
    ({"algorithm": {"n_samples": True}}, "algorithm.n_samples"),
    ({"async": {"pause_generation": "abort"}}, "async.pause_generation"),
    ({"async": {"over_sampling_batch_size": 0}}, "async.over_sampling_batch_size"),
    ({"cluster": {"gpu": "A10"}}, "cluster.gpu"),
    ({"cluster": {"nodes": 0}}, "cluster.nodes"),
    ({"cluster": {"timeout_hours": 25}}, "cluster.timeout_hours"),
    ({"checkpoint": {"keep_final_full": "yes"}}, "checkpoint.keep_final_full"),
])
def test_field_errors(over, path):
    errs = V.instant_errors(_recipe(**over), "run1", DEV)
    assert [e["path"] for e in errs] == [path], errs


def test_async_bounds_follow_miles():
    errs = V.instant_errors(_recipe(algorithm={"rollout_batch_size": 8}, **{"async": {"over_sampling_batch_size": 7}}),
                            "run1", DEV)
    assert errs[0]["path"] == "async.over_sampling_batch_size"
    errs = V.instant_errors(_recipe(**{"async": {"max_concurrent_samples": 1}}), "run1", DEV)
    assert errs[0]["path"] == "async.max_concurrent_samples"


def test_field_errors_report_every_field():
    errs = V.instant_errors(_recipe(algorithm={"lr": "1e-6", "num_rollout": 0}), "run1", DEV)
    assert {e["path"] for e in errs} == {"algorithm.lr", "algorithm.num_rollout"}
    with pytest.raises(ValueError, match="algorithm.lr"):
        R.validate(_recipe(algorithm={"lr": "1e-6"}), local_inputs=False)


def test_reserved_flags_cover_everything_emitted():
    errs = V.instant_errors(_recipe(extra_args=["--advantage-estimator", "gspo", "--lr=1e-5", "--seed", "3"]),
                            "run1", DEV)
    assert [e["message"].split()[0] for e in errs] == ["--advantage-estimator", "--lr", "--seed"]
    assert V.instant_errors(_recipe(extra_args=["--log-probs-chunk-size", "512"]), "run1", DEV) == []
    assert S.emitted_flags(_recipe()) >= {"--save", "--load", "--lr", "--advantage-estimator"}


def test_schema_carries_rules():
    props = S.schema()["json_schema"]["properties"]
    assert props["algorithm"]["properties"]["advantage_estimator"]["enum"] == list(R.ADVANTAGE_ESTIMATORS)
    assert props["algorithm"]["properties"]["lr"] == {"type": "number", "exclusiveMinimum": 0}
    assert props["async"]["properties"]["over_sampling_batch_size"] == {
        "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}
    assert props["cluster"]["properties"]["gpu"]["enum"] == list(R.GPU_TYPES)
    assert set(R.FIELD_RULES) <= {p for p, _ in R._leaves(R.DEFAULTS)}  # no rule for a key that does not exist


def _r5_resolved(recipe: dict) -> dict:
    """A recipe as r5 resolved and stored it: no async section, no r6 algorithm keys."""
    old = {k: v for k, v in recipe.items() if k != "async"}
    old["algorithm"] = {k: v for k, v in recipe["algorithm"].items()
                        if k not in ("advantage_estimator", "kl_loss_coef", "entropy_coef", "seed")}
    return old


def test_recipe_resolved_by_r5_still_validates_and_launches():
    old = _r5_resolved(_recipe())
    assert V.instant_errors(old, "run1", DEV) == []
    resolved = Target.of(DEV).apply(old)
    assert _argv(resolved) == _argv(Target.of(DEV).apply(_recipe()))


def test_unknown_recipe_key_is_a_recipe_error():
    errs = V.instant_errors({**_recipe(), "algorithm": {**_recipe()["algorithm"], "lr_typo": 1}}, "run1", DEV)
    assert errs[0]["path"] == "recipe" and "lr_typo" in errs[0]["message"]
