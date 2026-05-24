"""OpenEvolve evaluation entry point for Bloom filter alternatives."""

import math

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluator_entry import EvaluationEntryPoint, score_to_reward
from randomize_evolve.evaluators.bloom_alternatives import (
    EvaluationResult as BloomEvaluationResult,
)
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
    false_positive_weight=25000.0,
    memory_weight=0.01,
    latency_weight=2.0,
    bloom_regret_weight=200.0,
    max_memory_bytes=100 * 1024 * 1024,
)


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the Bloom filter evaluator."""
    return _ENTRY_POINT.evaluate(program_path)


def _success_result(bloom_result: BloomEvaluationResult) -> EvaluationResult:
    total_trials = len(bloom_result.trials)
    reliability = total_trials / len(DEFAULT_CONFIG.seeds)

    combined_score = score_to_reward(bloom_result.score)
    if not bloom_result.success:
        combined_score *= 0.7

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "bits_per_item": bloom_result.bits_per_item,
        "bloom_optimal_bits_per_item": bloom_result.bloom_optimal_bits_per_item,
        "excess_bits_per_item_vs_bloom": bloom_result.excess_bits_per_item_vs_bloom,
        "false_positive_rate": bloom_result.false_positive_rate,
        "bloom_optimal_false_positive_rate": bloom_result.bloom_optimal_false_positive_rate,
        "false_positive_ratio_vs_bloom": bloom_result.false_positive_ratio_vs_bloom,
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
            f"bloom_opt={bloom_result.bloom_optimal_bits_per_item:.1f} bits/item, "
            f"excess_bits={bloom_result.excess_bits_per_item_vs_bloom:.2f}, "
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
        "bloom_optimal_bits_per_item": 0.0,
        "excess_bits_per_item_vs_bloom": math.inf,
        "false_positive_rate": 1.0,
        "bloom_optimal_false_positive_rate": 0.0,
        "false_positive_ratio_vs_bloom": math.inf,
        "false_negative_rate": 1.0,
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
        "or `build_candidate(key_bits, capacity)`."
    ),
    timeout_suggestion=("Review the candidate implementation for long-running operations."),
    success_result_builder=_success_result,
    error_result_builder=_error_result,
)
