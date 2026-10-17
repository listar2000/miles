"""R4: release everything a run created, idempotently, on every ending.

``teardown`` stops the run's GPU work, terminates the sandboxes of its sandbox
app(s) through the provider's adapter, and deletes its secret. It returns what
it released and what still remains; call it again until ``remaining`` is empty.

Guards (the Modal workspace and the Sail org are shared with other teams):

- every name comes from ``Target`` and the run id, and must be in our
  namespaces (``check_own_app_name``); an adapter refuses any other name;
- apps are addressed by exact name only: Modal ``App.lookup`` in the target's
  environment (never created) for sandbox apps, the environment's app list
  filtered by equality for the per-run GPU app (an ephemeral app has no lookup),
  Sail ``App.list`` filtered by equality (never minted). Nothing here matches by
  prefix or pattern;
- Modal calls carry ``environment_name=target.modal_environment``; Sail is
  scoped by the org of ``SAIL_API_KEY`` (the SDK has no project or environment
  parameter), so the key is the Sail guard.
"""

import modal

from miles_ext.platform.progress import call_running, delete_keys, progress_dict, progress_dict_name
from miles_ext.platform.progress import keys as progress_keys
from miles_ext.platform.secrets import delete_run_secret, run_secret_exists
from miles_ext.platform.target import Target, compile_cache_volume_name, gpu_app_is_per_run, run_secret_name
from miles_ext.recipes import recipe as R


class ModalSandboxes:
    """Harbor's Modal sandboxes of one app, in one Modal environment."""

    def __init__(self, environment: str | None):
        self.environment = environment

    def _app(self, name: str):
        R.check_own_app_name(name)
        try:
            return modal.App.lookup(name, environment_name=self.environment)  # exact name; never creates
        except modal.exception.NotFoundError:
            return None

    def live(self, name: str) -> int:
        app = self._app(name)
        return 0 if app is None else len(list(modal.Sandbox.list(app_id=app.app_id)))

    def terminate(self, name: str) -> int:
        app = self._app(name)
        if app is None:
            return 0
        sandboxes = list(modal.Sandbox.list(app_id=app.app_id))
        for sb in sandboxes:
            sb.terminate()
        return len(sandboxes)


class SailSandboxes:
    """Sailboxes of one Sail app, in the org of SAIL_API_KEY."""

    def __init__(self, environment: str | None = None):
        self.environment = environment  # unused: Sail has no environments; the key is the scope

    @staticmethod
    def _boxes(name: str):
        from sail import App, Sailbox

        R.check_own_app_name(name)
        for app in (a for a in App.list() if a.name == name):  # list, not find: never mint an app here
            for box in Sailbox.list(app_id=app):
                if box.status != "terminated":
                    yield box

    def live(self, name: str) -> int:
        return sum(1 for _ in self._boxes(name))

    def terminate(self, name: str) -> int:
        count = 0
        for box in self._boxes(name):
            box.terminate()
            count += 1
        return count


ADAPTERS = {"modal": ModalSandboxes, "sail": SailSandboxes}


def sandbox_adapter(provider: str, environment: str | None):
    """A new provider is a Harbor environment plus an adapter here with ``live`` and ``terminate``."""
    if provider not in ADAPTERS:
        raise ValueError(f"no teardown adapter for sandbox provider {provider!r} (have {sorted(ADAPTERS)})")
    return ADAPTERS[provider](environment)


def resource_names(recipe: dict, run_id: str, target: Target | dict | None) -> dict:
    """Every name the run creates, known before anything exists (the platform stores it first)."""
    target = Target.of(target)
    recipe = target.apply(recipe)
    return {
        "modal_environment": target.modal_environment,
        "gpu_app": target.app_name,
        "gpu_app_per_run": gpu_app_is_per_run(run_id, target),
        "sandbox_provider": recipe["sandbox"]["provider"],
        "sandbox_apps": R.sandbox_app_names_for_cleanup(recipe, run_id),
        "secret": run_secret_name(run_id, target),
        "results_volume": target.results_volume,
        "results_dir": f"{run_id}/",
        "weights_volume": target.weights_volume,
        "compile_cache_volume": compile_cache_volume_name(target),  # shared by the environment's runs
        "progress_dict": progress_dict_name(target),
        "progress_keys": sorted(progress_keys(run_id).values()),
    }


def _gpu_idents(run_id: str, target: Target, app_id: str | None) -> list[str]:
    """The app by id (from launch) and, when the name is this run's alone, every live app of
    that exact name in the target's environment. The trainer runs as an ephemeral app, which
    a name lookup does not find, so the names come from the environment's app list; a launch
    interrupted before it reported its ids still leaves the per-run app findable."""
    idents = [app_id] if app_id else []
    if gpu_app_is_per_run(run_id, target):
        idents += [a for a in _live_app_ids(target.app_name, target.modal_environment) if a not in idents]
    return idents


def _stop_gpu(run_id: str, target: Target, call_ids: list[str], app_id: str | None) -> list[str]:
    released = []
    for call_id in call_ids:
        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            released.append(f"call:{call_id}")
        except modal.exception.NotFoundError:
            pass
    for ident in _gpu_idents(run_id, target, app_id):
        if stopped := _stop_app(ident, target.modal_environment):
            released.append(f"app:{target.app_name}:{stopped}")
    return released


def _gpu_remaining(run_id: str, target: Target, call_ids: list[str], app_id: str | None) -> list[str]:
    """What still runs after a stop: cancelling is asynchronous, so a call or app can take a while."""
    remaining = [f"call:{c}" for c in call_ids if call_running(c)]
    for ident in _gpu_idents(run_id, target, app_id):
        if state := _app_live(ident, target.modal_environment):
            remaining.append(f"app:{ident}:{state}")
    return remaining


async def _stop_app_async(ident: str, environment: str | None) -> str | None:
    """What ``modal app stop`` does (the SDK has no public app stop), refusing another environment."""
    from modal.cli.app import resolve_app_identifier
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    try:
        app_id, env, lifecycle = await resolve_app_identifier(ident, environment, client)
    except modal.exception.NotFoundError:
        return None
    if env and environment and env != environment:
        raise RuntimeError(f"app {ident} is in environment {env!r}, not {environment!r}; refusing")
    if lifecycle.app_state == api_pb2.APP_STATE_STOPPED:
        return None
    await client._stub.AppStop(api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI))
    return app_id


def _stop_app(ident: str, environment: str | None) -> str | None:
    from modal._utils.async_utils import synchronizer

    return synchronizer.create_blocking(_stop_app_async)(ident, environment)


async def _app_live_async(ident: str, environment: str | None) -> str | None:
    from modal.cli.app import resolve_app_identifier
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    try:
        _, env, lifecycle = await resolve_app_identifier(ident, environment, client)
    except modal.exception.NotFoundError:
        return None
    if env and environment and env != environment:
        return None  # not ours to report (or stop)
    if lifecycle.app_state in (api_pb2.APP_STATE_STOPPED, api_pb2.APP_STATE_STOPPING):
        return None
    return api_pb2.AppState.Name(lifecycle.app_state)


async def _live_app_ids_async(name: str, environment: str | None) -> list[str]:
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    resp = await client._stub.AppList(api_pb2.AppListRequest(environment_name=environment or ""))
    return [a.app_id for a in resp.apps if a.description == name  # exact name, never a pattern
            and a.state not in (api_pb2.APP_STATE_STOPPED, api_pb2.APP_STATE_STOPPING)]


def _live_app_ids(name: str, environment: str | None) -> list[str]:
    """Ids of the apps named exactly ``name`` (deployed or ephemeral) that are not stopped,
    as ``modal app list`` shows them."""
    from modal._utils.async_utils import synchronizer

    R.check_own_app_name(name)
    return synchronizer.create_blocking(_live_app_ids_async)(name, environment)


def _app_live(ident: str, environment: str | None) -> str | None:
    """The app's state name while it is not stopped (or stopping), else None."""
    from modal._utils.async_utils import synchronizer

    return synchronizer.create_blocking(_app_live_async)(ident, environment)


def teardown(run_id: str, recipe: dict, target: Target | dict | None, call_id: str | None = None,
             app_id: str | None = None) -> dict:
    """Idempotent. ``call_id`` and ``app_id`` are what launch returned (either may be None);
    the live segment's call id in ``active:<run_id>`` is cancelled too, so a launch whose
    caller never saw the ids is still found. ``remaining`` lists GPU calls and apps that
    have not stopped yet, sandboxes and the secret."""
    target = Target.of(target)
    names = resource_names(recipe, run_id, target)
    active = (progress_dict(target).get(progress_keys(run_id)["active"]) or {}).get("call_id")
    call_ids = list(dict.fromkeys(c for c in (call_id, active) if c))
    released: list[str] = []
    released += _stop_gpu(run_id, target, call_ids, app_id)
    adapter = sandbox_adapter(names["sandbox_provider"], target.modal_environment)
    for app in names["sandbox_apps"]:
        if n := adapter.terminate(app):
            released.append(f"sandboxes:{app}:{n}")
    if run_secret_exists(run_id, target):
        delete_run_secret(run_id, target)
        released.append(f"secret:{names['secret']}")
    released += [f"progress:{key}" for key in delete_keys(run_id, target)]
    remaining = _gpu_remaining(run_id, target, call_ids, app_id)
    remaining += [f"sandboxes:{app}:{n}" for app in names["sandbox_apps"] if (n := adapter.live(app))]
    if run_secret_exists(run_id, target):
        remaining.append(f"secret:{names['secret']}")
    return {"released": released, "remaining": remaining, "names": names}
