"""Levi evaluation entry point for cache eviction/admission policies."""

import math

from randomize_evolve.evaluator_entry import (
    EvaluationEntryPoint,
    EvaluatorResult,
    score_to_reward,
)
from randomize_evolve.evaluators.cache_eviction import (
    EvaluationResult as CacheEvaluationResult,
)
from randomize_evolve.evaluators.cache_eviction import Evaluator, EvaluatorConfig

__all__ = ["DEFAULT_CONFIG", "Evaluator", "evaluate", "evaluate_factory", "evaluate_source"]

EVALUATION_TIMEOUT_S = 90

DEFAULT_CONFIG = EvaluatorConfig(
    key_bits=24,
    capacity=512,
    trace_length=50000,
    working_set_size=8192,
    zipf_exponent=1.15,
    scan_burst_every=1200,
    scan_burst_length=384,
    drift_interval=10000,
    drift_stride=2048,
    seeds=(7, 19, 43, 89, 131),
    trace_timeout_s=3.0,
    max_memory_bytes=80 * 1024 * 1024,
)


def evaluate(program_path: str) -> EvaluatorResult:
    """Evaluate a candidate module using the cache eviction evaluator."""
    return _ENTRY_POINT.evaluate(program_path)


def evaluate_factory(factory) -> EvaluatorResult:
    """Evaluate a loaded candidate factory using the cache eviction evaluator."""
    return _ENTRY_POINT.evaluate_factory(factory)


def evaluate_source(source: str) -> EvaluatorResult:
    """Evaluate candidate source using the cache eviction evaluator."""
    return _ENTRY_POINT.evaluate_source(source)


def _success_result(cache_result: CacheEvaluationResult) -> EvaluatorResult:
    total_trials = len(cache_result.trials)
    reliability = total_trials / len(DEFAULT_CONFIG.seeds)

    combined_score = score_to_reward(cache_result.score)
    if not cache_result.success:
        combined_score *= 0.7

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "hit_rate": cache_result.hit_rate,
        "miss_rate": cache_result.miss_rate,
        "admission_rate": cache_result.admission_rate,
        "eviction_rate": cache_result.eviction_rate,
        "metadata_bytes_per_cached_item": cache_result.metadata_bytes_per_cached_item,
        "mean_peak_memory_bytes": cache_result.mean_peak_memory_bytes,
        "mean_trace_time_ms": cache_result.mean_trace_time_ms,
        "mean_access_time_us": cache_result.mean_access_time_us,
        "mean_operation_count": cache_result.mean_operation_count,
    }

    artifacts = {
        "errors": cache_result.error or "",
        "score_breakdown": (
            f"hit_rate={cache_result.hit_rate:.4f}, "
            f"miss_rate={cache_result.miss_rate:.4f}, "
            f"metadata={cache_result.metadata_bytes_per_cached_item:.1f}B/item, "
            f"access={cache_result.mean_access_time_us:.3f}us, "
            f"ops={cache_result.mean_operation_count:.0f}, "
            f"raw_score={cache_result.score:.2f}"
        ),
    }
    return EvaluatorResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluatorResult:
    metrics = {
        "combined_score": 0.0,
        "reliability": 0.0,
        "hit_rate": 0.0,
        "miss_rate": 1.0,
        "admission_rate": 0.0,
        "eviction_rate": 0.0,
        "metadata_bytes_per_cached_item": math.inf,
        "mean_peak_memory_bytes": math.inf,
        "mean_trace_time_ms": math.inf,
        "mean_access_time_us": math.inf,
        "mean_operation_count": math.inf,
        "error": message,
    }
    return EvaluatorResult(metrics=metrics, artifacts=artifacts)


_ENTRY_POINT = EvaluationEntryPoint(
    evaluator_factory=lambda: Evaluator(DEFAULT_CONFIG),
    timeout_seconds=EVALUATION_TIMEOUT_S,
    load_error_suggestion=(
        "Ensure the module defines `candidate_factory(key_bits, capacity)` "
        "or `build_candidate(key_bits, capacity)` and returns an object implementing "
        "on_access(key, hit), should_admit(key), and pick_victim()."
    ),
    timeout_suggestion=("Inspect the candidate for full-cache scans or unbounded metadata growth."),
    success_result_builder=_success_result,
    error_result_builder=_error_result,
)
