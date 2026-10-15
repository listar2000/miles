"""R4: release everything a run created, idempotently, on every ending.

``teardown`` stops the run's GPU work, terminates the sandboxes of its sandbox
app(s) through the provider's adapter, and deletes its secret. It returns what
it released and what still remains; call it again until ``remaining`` is empty.

Guards (the Modal workspace and the Sail org are shared with other teams):

- every name comes from ``Target`` and the run id, and must be in our
  namespaces (``check_own_app_name``); an adapter refuses any other name;
- apps are addressed by exact name only: Modal ``App.lookup`` in the target's
  environment (never created), Sail ``App.list`` filtered by equality (never
  minted). Nothing here lists by prefix or pattern;
- Modal calls carry ``environment_name=target.modal_environment``; Sail is
  scoped by the org of ``SAIL_API_KEY`` (the SDK has no project or environment
  parameter), so the key is the Sail guard.
"""

import modal

from miles_ext.platform.progress import delete_keys, progress_dict_name
from miles_ext.platform.progress import keys as progress_keys
from miles_ext.platform.secrets import delete_run_secret, run_secret_exists
from miles_ext.platform.target import Target, gpu_app_is_per_run, run_secret_name
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
        "progress_dict": progress_dict_name(target),
        "progress_keys": sorted(progress_keys(run_id).values()),
    }


def _stop_gpu(run_id: str, target: Target, call_id: str | None, app_id: str | None) -> list[str]:
    released = []
    if call_id:
        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            released.append(f"call:{call_id}")
        except modal.exception.NotFoundError:
            pass
    # the app itself: by id (from launch) or, when the name is this run's alone, by exact name
    ident = app_id or (target.app_name if gpu_app_is_per_run(run_id, target) else None)
    if ident and (stopped := _stop_app(ident, target.modal_environment)):
        released.append(f"app:{target.app_name}:{stopped}")
    return released


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


def teardown(run_id: str, recipe: dict, target: Target | dict | None, call_id: str | None = None,
             app_id: str | None = None) -> dict:
    """Idempotent. ``call_id`` and ``app_id`` are what launch returned (either may be None)."""
    target = Target.of(target)
    names = resource_names(recipe, run_id, target)
    released: list[str] = []
    released += _stop_gpu(run_id, target, call_id, app_id)
    adapter = sandbox_adapter(names["sandbox_provider"], target.modal_environment)
    for app in names["sandbox_apps"]:
        if n := adapter.terminate(app):
            released.append(f"sandboxes:{app}:{n}")
    if run_secret_exists(run_id, target):
        delete_run_secret(run_id, target)
        released.append(f"secret:{names['secret']}")
    released += [f"progress:{key}" for key in delete_keys(run_id, target)]
    remaining = [f"sandboxes:{app}:{n}" for app in names["sandbox_apps"] if (n := adapter.live(app))]
    if run_secret_exists(run_id, target):
        remaining.append(f"secret:{names['secret']}")
    return {"released": released, "remaining": remaining, "names": names}
