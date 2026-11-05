"""Terminate what a run left in its task-sandbox app, through the provider's adapter (a recipe names one provider)."""

from miles_ext.recipes import recipe as R


class ModalSandboxes:
    """Harbor's Modal sandboxes of one app, in one Modal environment."""

    def __init__(self, environment: str | None):
        self.environment = environment

    def _app(self, name: str):
        import modal

        R.check_own_app_name(name)
        try:
            return modal.App.lookup(name, environment_name=self.environment)  # exact name; never creates
        except modal.exception.NotFoundError:
            return None

    def live(self, name: str) -> int:
        import modal

        app = self._app(name)
        return 0 if app is None else len(list(modal.Sandbox.list(app_id=app.app_id)))

    def terminate(self, name: str) -> int:
        import modal

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
