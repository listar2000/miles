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
SCOPED_NAMESPACE = "rl-"  # names rl-<env>-<name_id>; they need a name id
NAME_ID_RE = r"[a-z0-9]{6,16}"


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

    The platform's names (option A, 2026-10-15) use a short name id next to the run id:
    ``modal_environment="rl-dev"``, ``name_id="0bebf81d"``, ``app_name="rl-dev-0bebf81d"``,
    ``results_volume="rl-dev-results"``, ``weights_volume="rl-dev-weights"``; then the
    sandbox app is ``<app_name>-sbx``, the secret ``<app_name>-secret`` and the progress
    Dict ``rl-dev-progress``. The run id still keys files, W&B and progress keys.
    Before that (still accepted): ``app_name="mdev-<run_id>"``, ``sandbox_app_prefix``,
    ``results_volume="mdev-results"``."""

    modal_environment: str | None = None  # None: the Modal profile's default environment
    app_name: str = "miles-fork"  # the GPU app
    sandbox_app_prefix: str = R.DEFAULT_APP_PREFIX
    results_volume: str = "miles-fork-results"
    weights_volume: str = "miles-swe-models"
    secret_name: str | None = None  # None: run_secret_name()
    name_id: str | None = None  # short id in the run's names (option A); the platform keeps it unique

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
        if self.name_id is not None or self.app_name.startswith(SCOPED_NAMESPACE):
            if not re.fullmatch(NAME_ID_RE, self.name_id or ""):
                raise ValueError(f"target.name_id {self.name_id!r} must be {NAME_ID_RE} (needed for {SCOPED_NAMESPACE} names)")
            if not re.fullmatch(rf"{SCOPED_NAMESPACE}[a-z0-9]+-{self.name_id}", self.app_name):
                raise ValueError(f"target.app_name {self.app_name!r} must be {SCOPED_NAMESPACE}<env>-{self.name_id}")
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
        app_name = f"{self.app_name}-sbx" if self.name_id else None
        return {**recipe, "sandbox": {**recipe["sandbox"], "app_prefix": self.sandbox_app_prefix, "app_name": app_name}}

    def with_(self, **changes) -> "Target":
        return replace(self, **changes)


def namespace_of(name: str) -> str:
    return next(p for p in R.OWN_APP_NAMESPACES if name.startswith(p))


def scope_of(target: Target) -> str:
    """The prefix of the target's shared names: ``rl-<env>-`` with a name id, else the namespace."""
    return target.app_name[:-len(target.name_id)] if target.name_id else namespace_of(target.app_name)


def helper_app_name(target: Target) -> str:
    """The app the platform's helper functions (deep check, HF export, weights download) run in:
    ``rl-<env>-helpers`` with a name id, else the shared ``miles-platform``."""
    return check_own_name(f"{scope_of(target)}helpers", "helper app") if target.name_id else "miles-platform"


def completion_queue_name(target: Target) -> str:
    """The trainer's completion queue (partitions per run id): ``rl-<env>-completion`` with a name id."""
    return check_own_name(f"{scope_of(target)}completion", "queue") if target.name_id else "miles-fork-completion"


def run_secret_name(run_id: str, target: Target) -> str:
    """The per-run secret: ``target.secret_name``, or ``<app_name>-secret`` with a name id,
    else ``<namespace of app_name>secret-<run_id>``."""
    if target.secret_name:
        return target.secret_name
    if target.name_id:
        return check_own_name(f"{target.app_name}-secret", "secret name")
    if not re.fullmatch(R.RUN_ID_RE, run_id):
        raise ValueError(f"run id {run_id!r}")
    return check_own_name(f"{namespace_of(target.app_name)}secret-{run_id}", "secret name")


def gpu_app_is_per_run(run_id: str, target: Target) -> bool:
    """Teardown stops the GPU app by name only when the name is this run's alone
    (``mdev-<run_id>``); the shared test app ``miles-fork`` is stopped by call id only."""
    if target.name_id:
        return True
    return target.app_name == f"{namespace_of(target.app_name)}{run_id}" and target.app_name != Target().app_name
