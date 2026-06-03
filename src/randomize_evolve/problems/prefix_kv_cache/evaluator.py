"""Levi evaluation entry point for prefix KV-cache scoring policies."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

from randomize_evolve.evaluator_entry import (
    EvaluatorResult,
    load_candidate_factory,
    load_candidate_factory_from_source,
    run_with_timeout,
)
from randomize_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult as PrefixEvaluationResult,
)
from randomize_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
    scoring_fn_complexity,
)

EVALUATION_TIMEOUT_S = 60

DEFAULT_CONFIG = EvaluatorConfig(capacity_sweep_blocks=(24, 48))


def evaluate(program_path: str) -> EvaluatorResult:
    """Evaluate a candidate module using train and validation splits only."""

    try:
        source = Path(program_path).read_text(encoding="utf-8")
    except Exception as exc:
        return _error_result(
            "failed to load candidate factory",
            _load_error_artifacts(exc),
        )
    return _evaluate_isolated(
        _evaluate_program_path,
        program_path,
        scoring_fn_complexity(source),
    )


def evaluate_factory(factory: Callable) -> EvaluatorResult:
    """Evaluate an already-loaded candidate factory.

    Complexity is unavailable for opaque callables and is therefore set to 0.
    """

    return _evaluate_isolated(_evaluate_factory, factory, 0)


def evaluate_source(source: str) -> EvaluatorResult:
    """Evaluate candidate source and apply the formula-complexity penalty."""

    return _evaluate_isolated(
        _evaluate_source,
        source,
        scoring_fn_complexity(source),
    )


def evaluate_hidden(factory: Callable) -> EvaluatorResult:
    """Evaluate the quarantined hidden split for final reporting only."""

    return _evaluate_isolated(
        _evaluate_factory,
        factory,
        0,
        splits=("hidden",),
        include_hidden=True,
    )


def _evaluate_isolated(
    worker: Callable[
        [object, int, tuple[str, ...]],
        tuple[PrefixEvaluationResult | None, dict | None],
    ],
    candidate: object,
    complexity: int,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
    include_hidden: bool = False,
) -> EvaluatorResult:
    try:
        result, load_error = run_with_timeout(
            worker,
            candidate,
            complexity,
            splits,
            timeout_seconds=min(EVALUATION_TIMEOUT_S, DEFAULT_CONFIG.timeout_s),
        )
    except TimeoutError as exc:
        return _error_result(
            "evaluation timed out",
            {
                "error_type": "TimeoutError",
                "error_message": str(exc),
                "suggestion": "Inspect candidate scoring methods for long-running logic.",
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _error_result(
            "evaluation failed",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Unexpected evaluator failure; inspect the traceback.",
            },
        )
    if load_error is not None:
        return _error_result("failed to load candidate factory", load_error)
    if result is None:  # pragma: no cover - defensive
        return _error_result(
            "evaluation failed",
            {
                "error_type": "RuntimeError",
                "error_message": "evaluation worker returned no result",
                "suggestion": "Unexpected evaluator failure; inspect the traceback.",
            },
        )
    return _success_result(result, include_hidden=include_hidden)


def _evaluate_program_path(
    program_path: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult | None, dict | None]:
    try:
        factory = load_candidate_factory(str(program_path))
    except Exception as exc:
        return None, _load_error_artifacts(exc)
    return _evaluate_factory(factory, complexity, splits)


def _evaluate_source(
    source: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult | None, dict | None]:
    try:
        factory = load_candidate_factory_from_source(str(source))
    except Exception as exc:
        return None, _load_error_artifacts(exc)
    return _evaluate_factory(factory, complexity, splits)


def _evaluate_factory(
    factory: object,
    complexity: int,
    splits: tuple[str, ...],
) -> tuple[PrefixEvaluationResult, None]:
    evaluator = PrefixKVCacheEvaluator(DEFAULT_CONFIG, splits=splits)
    return evaluator(factory, scoring_fn_complexity=complexity), None  # type: ignore[arg-type]


def _load_error_artifacts(exc: Exception) -> dict:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "full_traceback": traceback.format_exc(),
        "suggestion": _load_suggestion(),
    }


def _success_result(
    prefix_result: PrefixEvaluationResult,
    *,
    include_hidden: bool = False,
) -> EvaluatorResult:
    metrics = {
        "combined_score": prefix_result.combined_score,
        "success": prefix_result.success,
        "invalid_fraction": prefix_result.invalid_fraction,
    }
    for split, split_metrics in prefix_result.split_metrics.items():
        if split == "hidden" and not include_hidden:
            continue
        for key, value in split_metrics.items():
            if isinstance(value, (int, float, bool)):
                metrics[f"{split}_{key}"] = value

    artifacts = {
        "split_metrics": {
            key: value
            for key, value in prefix_result.split_metrics.items()
            if include_hidden or key != "hidden"
        },
        "workload_metrics": {
            key: value
            for key, value in prefix_result.workload_metrics.items()
            if include_hidden or not key.startswith("hidden/")
        },
        "capacity_metrics": prefix_result.capacity_metrics,
        "candidate_metadata": prefix_result.candidate_metadata,
    }
    return EvaluatorResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluatorResult:
    return EvaluatorResult(
        metrics={
            "combined_score": DEFAULT_CONFIG.v_min - 1.0 - DEFAULT_CONFIG.invalid_surcharge,
            "success": False,
            "invalid_fraction": 1.0,
            "error": message,
        },
        artifacts=artifacts,
    )


def _load_suggestion() -> str:
    return (
        "Ensure the module defines `candidate_factory(capacity_blocks, "
        "block_size_tokens, seed=None)` or `build_candidate(...)` and returns an "
        "object implementing the prefix KV-cache scoring interface."
    )
