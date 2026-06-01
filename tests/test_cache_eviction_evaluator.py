"""Tests for the cache eviction/admission evaluator and baselines."""

import pytest

from randomize_evolve.evaluators.cache_eviction import (
    Evaluator,
    EvaluatorConfig,
    lfu_policy,
    lru_policy,
    random_replacement_policy,
    window_tinylfu_policy,
)
from randomize_evolve.problems.cache_eviction.evaluator import evaluate_factory
from randomize_evolve.problems.cache_eviction.initial_program import candidate_factory
from randomize_evolve.problems.cache_eviction.run import (
    INITIAL_PROGRAM_SOURCE,
    build_arg_parser,
)


def _small_config(**overrides) -> EvaluatorConfig:
    values = {
        "key_bits": 16,
        "capacity": 64,
        "trace_length": 6000,
        "working_set_size": 1024,
        "scan_burst_every": 500,
        "scan_burst_length": 96,
        "drift_interval": 1500,
        "drift_stride": 256,
        "seeds": (3, 11),
        "trace_timeout_s": 2.0,
    }
    values.update(overrides)
    return EvaluatorConfig(**values)


def test_trace_mixes_zipf_scans_and_drift() -> None:
    config = _small_config(trace_length=2200, seeds=(5,))
    evaluator = Evaluator(config)

    trace = evaluator._generate_trace(5)

    assert len(trace) == 2200
    assert len(set(trace[500:596])) == 96
    assert max(trace[:400]) < config.working_set_size
    assert max(trace[config.drift_interval + 1 : config.drift_interval + 100]) >= (
        config.drift_stride
    )


def test_baseline_policies_run_on_mixed_workload() -> None:
    evaluator = Evaluator(_small_config())

    results = {
        "random": evaluator(random_replacement_policy()),
        "lru": evaluator(lru_policy()),
        "lfu": evaluator(lfu_policy()),
        "wtinylfu": evaluator(window_tinylfu_policy()),
    }

    assert all(result.success for result in results.values())
    assert all(0.0 <= result.hit_rate <= 1.0 for result in results.values())
    assert results["lru"].hit_rate > results["random"].hit_rate
    assert results["wtinylfu"].metadata_bytes_per_cached_item > 0


def test_initial_candidate_uses_contract_and_reports_metadata() -> None:
    evaluator = Evaluator(_small_config())

    result = evaluator(candidate_factory)

    assert result.success
    assert result.hit_rate > 0.0
    assert result.metadata_bytes_per_cached_item > 0
    assert result.mean_operation_count > 0


def test_framework_rejects_nonresident_victim() -> None:
    class BadPolicy:
        def on_access(self, key: int, hit: bool) -> None:
            del key, hit

        def should_admit(self, key: int) -> bool:
            del key
            return True

        def pick_victim(self) -> int:
            return -1

    evaluator = Evaluator(_small_config(capacity=2, trace_length=20, scan_burst_length=0))
    result = evaluator(lambda key_bits, capacity: BadPolicy())

    assert not result.success
    assert "non-resident" in (result.error or "")


def test_problem_entry_point_adapts_cache_metrics() -> None:
    result = evaluate_factory(candidate_factory)

    assert result.metrics["combined_score"] > 0.0
    assert 0.0 <= result.metrics["hit_rate"] <= 1.0
    assert "metadata=" in result.artifacts["score_breakdown"]


def test_cache_run_defaults_and_seed_source() -> None:
    args = build_arg_parser().parse_args([])

    assert args.iterations == 25
    assert args.config == "configs/cache_eviction_workload.yaml"
    assert "def candidate_factory" in INITIAL_PROGRAM_SOURCE.text()


def test_capacity_must_stay_below_working_set() -> None:
    with pytest.raises(ValueError, match="capacity"):
        EvaluatorConfig(capacity=128, working_set_size=128)
