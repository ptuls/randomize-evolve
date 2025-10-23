"""OpenEvolve evaluation entry point for Bloom filter alternatives.

This module exposes a single ``evaluate`` function that OpenEvolve can import
when running in direct (non-cascade) mode. It wraps the
``BloomAlternativeEvaluator`` and adapts the internal metrics into the
``EvaluationResult`` structure required by the platform.
"""

import concurrent.futures
import importlib.util
import math
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluators.bloom_alternatives import Distribution
from randomize_evolve.evaluators.bloom_alternatives import EvaluationResult as BloomEvaluationResult
from randomize_evolve.evaluators.bloom_alternatives import Evaluator, EvaluatorConfig

EVALUATION_TIMEOUT_S = 60

DEFAULT_CONFIG = EvaluatorConfig(
    key_bits=32,
    positives=5000,
    queries=10000,
    negative_fraction=0.5,
    seeds=(17, 23, 71, 89, 131),
    build_timeout_s=1.0,
    query_timeout_s=1.0,
    false_negative_penalty=1_000_000.0,
    false_positive_weight=180.0,
    memory_weight=0.05,
    latency_weight=8.0,
    max_memory_bytes=100 * 1024 * 1024,
)

CandidateFactory = Callable[[int, int], object]


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the Bloom filter evaluator."""
    try:
        factory = _load_candidate_factory(program_path)
    except Exception as exc:  # pragma: no cover - defensive
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": (
                "Ensure the module defines `candidate_factory(key_bits, capacity)` "
                "or `build_candidate(key_bits, capacity)`."
            ),
        }
        return _error_result("failed to load candidate factory", artifacts)

    evaluator = Evaluator(DEFAULT_CONFIG)

    try:
        bloom_result = _run_with_timeout(evaluator, factory, timeout_seconds=EVALUATION_TIMEOUT_S)
    except TimeoutError as exc:
        artifacts = {
            "error_type": "TimeoutError",
            "error_message": str(exc),
            "suggestion": "Review the candidate implementation for long-running operations.",
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

    return _success_result(bloom_result)


def _run_with_timeout(
    func: Callable[..., BloomEvaluationResult],
    *args,
    timeout_seconds: float,
    **kwargs,
) -> BloomEvaluationResult:
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


def _success_result(bloom_result: BloomEvaluationResult) -> EvaluationResult:
    total_trials = len(bloom_result.trials)
    reliability = total_trials / len(DEFAULT_CONFIG.seeds)

    combined_score = _score_to_reward(bloom_result.score)
    if not bloom_result.success:
        combined_score *= 0.7

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "bits_per_item": bloom_result.bits_per_item,
        "false_positive_rate": bloom_result.false_positive_rate,
        "false_negative_rate": bloom_result.false_negative_rate,
        "mean_build_time_ms": bloom_result.mean_build_time_ms,
        "mean_query_time_ms": bloom_result.mean_query_time_ms,
        "mean_peak_memory_bytes": bloom_result.mean_peak_memory_bytes,
    }

    artifacts = {
        "errors": bloom_result.error or "",
        "score_breakdown": (
            f"fp_rate={bloom_result.false_positive_rate:.4f}, "
            f"fn_rate={bloom_result.false_negative_rate:.4f}, "
            f"memory={bloom_result.mean_peak_memory_bytes:.0f}B ({bloom_result.bits_per_item:.1f} bits/item), "
            f"build={bloom_result.mean_build_time_ms:.2f}ms, "
            f"query={bloom_result.mean_query_time_ms:.2f}ms, "
            f"raw_score={bloom_result.score:.2f}"
        ),
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluationResult:
    metrics = {
        "combined_score": 0.0,
        "reliability": 0.0,
        "bits_per_item": math.inf,
        "false_positive_rate": 1.0,
        "false_negative_rate": 1.0,
        "mean_build_time_ms": math.inf,
        "mean_query_time_ms": math.inf,
        "mean_peak_memory_bytes": math.inf,
        "error": message,
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _score_to_reward(score: float) -> float:
    return 0.0 if not math.isfinite(score) else 1.0 / (1.0 + max(score, 0.0))
