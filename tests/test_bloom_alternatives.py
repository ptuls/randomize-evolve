"""Tests for Bloom-relative set-membership evaluation."""

import math

from randomize_evolve.evaluators.bloom_alternatives import (
    Evaluator,
    EvaluatorConfig,
    baseline_bloom_filter,
)


def test_bloom_optimal_formula_matches_classic_one_percent_rule() -> None:
    bits_per_item = Evaluator._optimal_bloom_bits_per_item(0.01)

    assert math.isclose(bits_per_item, 9.585, rel_tol=0.0, abs_tol=0.01)
    assert math.isclose(
        Evaluator._optimal_bloom_false_positive_rate(bits_per_item),
        0.01,
        rel_tol=0.0,
        abs_tol=1e-4,
    )


def test_bloom_regret_increases_score_for_same_candidate_quality() -> None:
    evaluator = Evaluator()

    no_regret = evaluator._score(
        false_positive_rate=0.01,
        false_negative_rate=0.0,
        mean_peak_memory_bytes=8_000.0,
        mean_build_time_ms=50.0,
        mean_query_time_ms=10.0,
        excess_bits_per_item_vs_bloom=0.0,
    )
    with_regret = evaluator._score(
        false_positive_rate=0.01,
        false_negative_rate=0.0,
        mean_peak_memory_bytes=8_000.0,
        mean_build_time_ms=50.0,
        mean_query_time_ms=10.0,
        excess_bits_per_item_vs_bloom=2.0,
    )

    assert with_regret > no_regret


def test_evaluator_reports_bloom_relative_metrics() -> None:
    config = EvaluatorConfig(
        positives=128,
        queries=256,
        negative_fraction=0.5,
        seeds=(17,),
        build_timeout_s=1.0,
        query_timeout_s=1.0,
    )

    result = Evaluator(config)(baseline_bloom_filter(bits_per_item=10))

    assert result.success
    assert result.bloom_optimal_bits_per_item > 0.0
    assert result.excess_bits_per_item_vs_bloom >= 0.0
    assert result.bloom_optimal_false_positive_rate > 0.0
    assert result.false_positive_ratio_vs_bloom > 0.0
