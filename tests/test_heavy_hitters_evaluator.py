"""Smoke tests for the heavy hitter evaluator and baseline."""

from initial_program_heavy_hitters import candidate_factory
from randomize_evolve.evaluators.heavy_hitters import (
    Evaluator,
    EvaluatorConfig,
    baseline_count_min_sketch,
)


def test_baseline_count_min_sketch_runs() -> None:
    config = EvaluatorConfig(
        key_bits=14,
        stream_length=4000,
        queries=512,
        top_k=6,
        num_true_heavy_hitters=6,
        seeds=(3, 7),
    )
    evaluator = Evaluator(config)
    result = evaluator(baseline_count_min_sketch())

    assert result.success
    assert 0.0 <= result.heavy_recall <= 1.0
    assert 0.0 <= result.heavy_precision <= 1.0
    assert result.mean_peak_memory_bytes > 0


def test_initial_candidate_factory_stays_within_timeout_budget() -> None:
    config = EvaluatorConfig(
        key_bits=18,
        stream_length=25000,
        queries=4096,
        top_k=12,
        num_true_heavy_hitters=16,
        heavy_hitters_fraction=0.7,
        max_update_weight=6,
        seeds=(19,),
        build_timeout_s=2.0,
        query_timeout_s=1.5,
        max_memory_bytes=80 * 1024 * 1024,
    )
    evaluator = Evaluator(config)
    result = evaluator(candidate_factory)

    assert result.success
    assert result.error is None
