"""Where a run's resources live: the ``target`` every platform verb takes last.

The Modal workspace and the Sail org are shared with other teams, so every
name a run creates or deletes is checked against our namespaces
(``recipe.OWN_APP_NAMESPACES``) here, before any call reaches a provider.
Dev and prod are told apart by the Modal environment; on Sail they share the
org of the API key, so run ids must be unique across environments.
"""

import re
from dataclasses import asdict, dataclass, replace

from miles_ext.recipes import recipe as R

VOLUME_NAME_RE = r"[a-z0-9][a-z0-9-]*"


def check_own_name(name: str, what: str) -> str:
    """Raise unless ``name`` is a single name in our namespaces (no patterns, no paths)."""
    if not isinstance(name, str) or not name.startswith(R.OWN_APP_NAMESPACES) or not re.fullmatch(R.RUN_ID_RE, name):
        raise ValueError(f"{what} {name!r} must start with one of {R.OWN_APP_NAMESPACES} "
                         "and hold only letters, digits, '.', '_', '-'")
    if len(name) > R.APP_NAME_MAX:
        raise ValueError(f"{what} {name!r} is longer than {R.APP_NAME_MAX}")
    return name


@dataclass(frozen=True)
class Target:
    """Defaults are the hand-launched test setup (``miles_ext.launch.modal_app``).

    The platform's dev values: ``modal_environment="miles-dev"``,
    ``app_name="mdev-<run_id>"``, ``sandbox_app_prefix="miles-mobius-"``,
    ``results_volume="mdev-results"``."""

    modal_environment: str | None = None  # None: the Modal profile's default environment
    app_name: str = "miles-fork"  # the GPU app
    sandbox_app_prefix: str = R.DEFAULT_APP_PREFIX
    results_volume: str = "miles-fork-results"
    weights_volume: str = "miles-swe-models"
    secret_name: str | None = None  # None: run_secret_name()

    @classmethod
    def of(cls, target: "Target | dict | None") -> "Target":
        if target is None:
            target = cls()
        elif isinstance(target, dict):
            unknown = set(target) - set(cls.__dataclass_fields__)
            if unknown:
                raise KeyError(f"unknown target keys {sorted(unknown)}")
            target = cls(**target)
        target.check()
        return target

    def check(self) -> None:
        check_own_name(self.app_name, "target.app_name")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*-", self.sandbox_app_prefix or "") or not self.sandbox_app_prefix.startswith(
                R.OWN_APP_NAMESPACES):
            raise ValueError(f"target.sandbox_app_prefix {self.sandbox_app_prefix!r} must start with one of "
                             f"{R.OWN_APP_NAMESPACES} and end in '-'")
        for field in ("results_volume", "weights_volume"):
            value = getattr(self, field)
            if not re.fullmatch(VOLUME_NAME_RE, value or "") or not value.startswith(R.OWN_APP_NAMESPACES):
                raise ValueError(f"target.{field} {value!r} must be a volume name in {R.OWN_APP_NAMESPACES}")
        if self.secret_name is not None:
            check_own_name(self.secret_name, "target.secret_name")
        if self.modal_environment is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.modal_environment):
            raise ValueError(f"target.modal_environment {self.modal_environment!r}")

    def as_dict(self) -> dict:
        return asdict(self)

    def apply(self, recipe: dict) -> dict:
        """The recipe with this target's sandbox prefix (the only recipe key a target owns), and
        defaults for keys it lacks (a recipe resolved by an older release has no ``async``)."""
        recipe = R.from_dict(recipe)
        return {**recipe, "sandbox": {**recipe["sandbox"], "app_prefix": self.sandbox_app_prefix}}

    def with_(self, **changes) -> "Target":
        return replace(self, **changes)


def namespace_of(name: str) -> str:
    return next(p for p in R.OWN_APP_NAMESPACES if name.startswith(p))


def run_secret_name(run_id: str, target: Target) -> str:
    """The per-run secret: ``target.secret_name``, or ``<namespace of app_name>secret-<run_id>``."""
    if target.secret_name:
        return target.secret_name
    if not re.fullmatch(R.RUN_ID_RE, run_id):
        raise ValueError(f"run id {run_id!r}")
    return check_own_name(f"{namespace_of(target.app_name)}secret-{run_id}", "secret name")


def gpu_app_is_per_run(run_id: str, target: Target) -> bool:
    """Teardown stops the GPU app by name only when the name is this run's alone
    (``mdev-<run_id>``); the shared test app ``miles-fork`` is stopped by call id only."""
    return target.app_name == f"{namespace_of(target.app_name)}{run_id}" and target.app_name != Target().app_name
