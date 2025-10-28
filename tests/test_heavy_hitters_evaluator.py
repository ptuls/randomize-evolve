"""Smoke tests for the heavy hitter evaluator and baseline."""

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
