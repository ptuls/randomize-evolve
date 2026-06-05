"""Operative evaluator configuration for the prefix KV-cache problem."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from randomize_evolve.evaluators.prefix_kv_cache import EvaluatorConfig

PREFIX_KV_CONFIG_ENV = "RANDOMIZE_EVOLVE_PREFIX_KV_CONFIG"
PREFIX_KV_QUICK_ENV = "RANDOMIZE_EVOLVE_PREFIX_KV_QUICK"
DEFAULT_CONFIG_PATH = Path("configs/prefix_kv_cache.yaml")

_TUPLE_FIELDS = {
    "capacity_sweep_blocks",
    "seeds",
    "train_families",
    "validation_families",
    "hidden_families",
}
_MAPPING_FIELDS = {"family_request_multipliers"}
_SCORING_FIELDS = {
    "w_avg_tok",
    "w_avg_blk",
    "min_workload_weight",
    "min_seed_weight",
    "request_tail_weight",
    "worst_window_weight",
    "priority_hit_weight",
    "wasted_admission_weight",
    "admission_utility_weight",
    "avoidable_eviction_weight",
    "latency_weight",
    "latency_cap",
    "churn_weight",
    "churn_cap",
    "fairness_weight",
    "fairness_cap",
    "k_complex",
    "complexity_exponent",
    "v_min",
    "invalid_surcharge",
}
_DIRECT_FIELDS = (
    {field.name for field in fields(EvaluatorConfig)}
    - _TUPLE_FIELDS
    - _MAPPING_FIELDS
    - _SCORING_FIELDS
)


def load_evaluator_config(path: Path = DEFAULT_CONFIG_PATH) -> EvaluatorConfig:
    """Load the evaluator settings that are operative for reports and evolution."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data = _mapping(data, label="root")
    problem = _mapping(data.get("problem"), label="problem")
    settings = _mapping(problem.get("settings"), label="problem.settings")
    config = evaluator_config_from_settings(settings)
    evaluator = _mapping(data.get("evaluator"), label="evaluator")
    if evaluator.get("timeout") is not None:
        config = replace(config, timeout_s=float(evaluator["timeout"]))
    return config


def evaluator_config_from_settings(
    settings: Mapping[str, Any],
    *,
    base: EvaluatorConfig | None = None,
) -> EvaluatorConfig:
    """Overlay YAML problem settings onto an evaluator configuration."""

    supported_settings = _DIRECT_FIELDS | _TUPLE_FIELDS | _MAPPING_FIELDS | {"scoring"}
    unknown_settings = sorted(set(settings) - supported_settings)
    if unknown_settings:
        raise ValueError(
            "unsupported prefix KV-cache evaluator settings: "
            + ", ".join(unknown_settings)
        )
    kwargs: dict[str, Any] = {}
    for name in _DIRECT_FIELDS:
        if name in settings:
            kwargs[name] = settings[name]
    for name in _TUPLE_FIELDS:
        if name in settings:
            kwargs[name] = tuple(settings[name])
    for name in _MAPPING_FIELDS:
        if name in settings:
            kwargs[name] = dict(settings[name])
    scoring = _mapping(settings.get("scoring"), label="problem.settings.scoring")
    unknown_scoring = sorted(set(scoring) - _SCORING_FIELDS)
    if unknown_scoring:
        raise ValueError(
            "unsupported prefix KV-cache scoring settings: "
            + ", ".join(unknown_scoring)
        )
    for name in _SCORING_FIELDS:
        if name in scoring:
            kwargs[name] = scoring[name]
    return replace(base or EvaluatorConfig(), **kwargs)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def active_evaluator_config(default: EvaluatorConfig) -> EvaluatorConfig:
    """Return an environment-selected config or the supplied test/default config."""

    path = os.environ.get(PREFIX_KV_CONFIG_ENV)
    config = load_evaluator_config(Path(path)) if path else default
    if os.environ.get(PREFIX_KV_QUICK_ENV) == "1":
        config = replace(
            config,
            request_count=36,
            seeds=(3,),
            family_request_multipliers={},
        )
    return config


@contextmanager
def prefix_kv_config_environment(path: Path, *, quick: bool = False) -> Iterator[None]:
    """Expose the operative config path to Levi evaluator worker processes."""

    previous = os.environ.get(PREFIX_KV_CONFIG_ENV)
    previous_quick = os.environ.get(PREFIX_KV_QUICK_ENV)
    os.environ[PREFIX_KV_CONFIG_ENV] = str(path.resolve())
    if quick:
        os.environ[PREFIX_KV_QUICK_ENV] = "1"
    else:
        os.environ.pop(PREFIX_KV_QUICK_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PREFIX_KV_CONFIG_ENV, None)
        else:
            os.environ[PREFIX_KV_CONFIG_ENV] = previous
        if previous_quick is None:
            os.environ.pop(PREFIX_KV_QUICK_ENV, None)
        else:
            os.environ[PREFIX_KV_QUICK_ENV] = previous_quick
