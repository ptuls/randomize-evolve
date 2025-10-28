"""OpenEvolve evaluation entry point for approximate heavy hitter algorithms."""

import concurrent.futures
import importlib.util
import math
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluators.heavy_hitters import (
    EvaluationResult as HeavyEvaluationResult,
)
from randomize_evolve.evaluators.heavy_hitters import Evaluator, EvaluatorConfig

EVALUATION_TIMEOUT_S = 75

DEFAULT_CONFIG = EvaluatorConfig(
    key_bits=18,
    stream_length=25000,
    queries=4096,
    top_k=12,
    num_true_heavy_hitters=16,
    heavy_hitters_fraction=0.7,
    max_update_weight=6,
    seeds=(5, 11, 29, 47),
    build_timeout_s=2.0,
    query_timeout_s=1.5,
    max_memory_bytes=80 * 1024 * 1024,
)

CandidateFactory = Callable[[int, int], object]


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the heavy hitter evaluator."""
    try:
        factory = _load_candidate_factory(program_path)
    except Exception as exc:  # pragma: no cover - defensive
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": (
                "Ensure the module defines `candidate_factory(key_bits, capacity)` "
                "or `build_candidate(key_bits, capacity)` and returns an object implementing "
                "observe(), estimate(), and top_k()."
            ),
        }
        return _error_result("failed to load candidate factory", artifacts)

    evaluator = Evaluator(DEFAULT_CONFIG)

    try:
        heavy_result = _run_with_timeout(evaluator, factory, timeout_seconds=EVALUATION_TIMEOUT_S)
    except TimeoutError as exc:
        artifacts = {
            "error_type": "TimeoutError",
            "error_message": str(exc),
            "suggestion": "Inspect the candidate implementation for long-running operations.",
        }
        return _error_result("evaluation timed out", artifacts)
    except Exception as exc:  # pragma: no cover - defensive
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Unexpected evaluator failure; inspect the traceback.",
        }
        return _error_result("evaluation failed", artifacts)

    return _success_result(heavy_result)


def _run_with_timeout(
    func: Callable[..., HeavyEvaluationResult],
    *args,
    timeout_seconds: float,
    **kwargs,
) -> HeavyEvaluationResult:
    """Execute ``func`` with a wall-clock timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:  # pragma: no cover - best effort
            future.cancel()
            raise TimeoutError(f"evaluation exceeded {timeout_seconds}s wall-clock limit") from exc


def _load_candidate_factory(program_path: str) -> CandidateFactory:
    path = Path(program_path)
    if not path.exists():
        raise FileNotFoundError(f"program path {path} does not exist")

    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]

    factory = _extract_factory(module)
    if not callable(factory):
        raise TypeError("candidate_factory must be callable")
    return factory


def _extract_factory(module: ModuleType) -> CandidateFactory:
    if hasattr(module, "candidate_factory"):
        return getattr(module, "candidate_factory")  # type: ignore[return-value]
    if hasattr(module, "build_candidate"):
        return getattr(module, "build_candidate")  # type: ignore[return-value]
    raise AttributeError("candidate module must expose `candidate_factory` or `build_candidate`")


def _success_result(heavy_result: HeavyEvaluationResult) -> EvaluationResult:
    total_trials = len(heavy_result.trials)
    reliability = total_trials / len(DEFAULT_CONFIG.seeds)

    combined_score = _score_to_reward(heavy_result.score)
    if not heavy_result.success:
        combined_score *= 0.7

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "heavy_precision": heavy_result.heavy_precision,
        "heavy_recall": heavy_result.heavy_recall,
        "mean_absolute_error": heavy_result.mean_absolute_error,
        "mean_relative_error": heavy_result.mean_relative_error,
        "zero_frequency_error": heavy_result.zero_frequency_error,
        "bits_per_observation": heavy_result.bits_per_observation,
        "mean_build_time_ms": heavy_result.mean_build_time_ms,
        "mean_query_time_ms": heavy_result.mean_query_time_ms,
        "mean_peak_memory_bytes": heavy_result.mean_peak_memory_bytes,
    }

    artifacts = {
        "errors": heavy_result.error or "",
        "score_breakdown": (
            f"precision={heavy_result.heavy_precision:.3f}, "
            f"recall={heavy_result.heavy_recall:.3f}, "
            f"rel_err={heavy_result.mean_relative_error:.4f}, "
            f"abs_err={heavy_result.mean_absolute_error:.2f}, "
            f"zero_err={heavy_result.zero_frequency_error:.2f}, "
            f"memory={heavy_result.mean_peak_memory_bytes:.0f}B, "
            f"build={heavy_result.mean_build_time_ms:.2f}ms, "
            f"query={heavy_result.mean_query_time_ms:.2f}ms, "
            f"raw_score={heavy_result.score:.2f}"
        ),
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluationResult:
    metrics = {
        "combined_score": 0.0,
        "reliability": 0.0,
        "heavy_precision": 0.0,
        "heavy_recall": 0.0,
        "mean_absolute_error": math.inf,
        "mean_relative_error": math.inf,
        "zero_frequency_error": math.inf,
        "bits_per_observation": math.inf,
        "mean_build_time_ms": math.inf,
        "mean_query_time_ms": math.inf,
        "mean_peak_memory_bytes": math.inf,
        "error": message,
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _score_to_reward(score: float) -> float:
    return 0.0 if not math.isfinite(score) else 1.0 / (1.0 + max(score, 0.0))
