"""R7: the per-run Modal secret, holding only the keys the run needs.

The caller (the platform's controller, from the project's credential pool)
passes the values; nothing is read from a .env here. The Sail key is required
when the run's sandboxes are on Sail: Harbor's Sail provider reads
``SAIL_API_KEY`` in the rollout workers, which inherit the trainer's env.
"""

import modal

from miles_ext.platform.target import Target, run_secret_name

ALLOWED_KEYS = ("WANDB_API_KEY", "HF_TOKEN", "SAIL_API_KEY")


def required_keys(recipe: dict) -> list[str]:
    keys = ["WANDB_API_KEY"]
    if recipe["sandbox"]["provider"] == "sail":
        keys.append("SAIL_API_KEY")
    return keys


def secret_values(recipe: dict, values: dict[str, str]) -> dict[str, str]:
    """The secret's contents: refuses unknown keys and missing required ones."""
    unknown = sorted(set(values) - set(ALLOWED_KEYS))
    if unknown:
        raise ValueError(f"per-run secret: keys {unknown} are not allowed (allowed: {ALLOWED_KEYS})")
    missing = [k for k in required_keys(recipe) if not values.get(k)]
    if missing:
        raise ValueError(f"per-run secret: missing {missing}")
    return {k: str(v) for k, v in values.items() if v}


def create_run_secret(recipe: dict, run_id: str, target: Target | dict | None, values: dict[str, str]) -> str:
    """Create (or replace) the run's secret; returns its name."""
    target = Target.of(target)
    name = run_secret_name(run_id, target)
    modal.Secret.objects.create(name, secret_values(recipe, values), allow_existing=True,
                                environment_name=target.modal_environment)
    return name


def delete_run_secret(run_id: str, target: Target | dict | None) -> str:
    """Idempotent; returns the name it deleted (or that was already gone)."""
    target = Target.of(target)
    name = run_secret_name(run_id, target)
    modal.Secret.objects.delete(name, allow_missing=True, environment_name=target.modal_environment)
    return name


def run_secret_exists(run_id: str, target: Target | dict | None) -> bool:
    target = Target.of(target)
    name = run_secret_name(run_id, target)
    try:
        modal.Secret.from_name(name, environment_name=target.modal_environment).hydrate()
        return True
    except modal.exception.NotFoundError:
        return False
