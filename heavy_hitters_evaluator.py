"""OpenEvolve evaluation entry point for approximate heavy hitter algorithms."""

import math

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluator_entry import EvaluationEntryPoint, score_to_reward
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


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the heavy hitter evaluator."""
    return _ENTRY_POINT.evaluate(program_path)


def _success_result(heavy_result: HeavyEvaluationResult) -> EvaluationResult:
    total_trials = len(heavy_result.trials)
    reliability = total_trials / len(DEFAULT_CONFIG.seeds)

    combined_score = score_to_reward(heavy_result.score)
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


_ENTRY_POINT = EvaluationEntryPoint(
    evaluator_factory=lambda: Evaluator(DEFAULT_CONFIG),
    timeout_seconds=EVALUATION_TIMEOUT_S,
    load_error_suggestion=(
        "Ensure the module defines `candidate_factory(key_bits, capacity)` "
        "or `build_candidate(key_bits, capacity)` and returns an object implementing "
        "observe(), estimate(), and top_k()."
    ),
    timeout_suggestion=(
        "Inspect the candidate implementation for long-running operations."
    ),
    success_result_builder=_success_result,
    error_result_builder=_error_result,
)
