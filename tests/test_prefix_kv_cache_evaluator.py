"""Tests for the prefix KV-cache evaluator."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import math
import os
from pathlib import Path
from types import SimpleNamespace
import textwrap
import time

import pytest

from randomize_evolve.evaluators.prefix_kv_cache import (
    BASELINES,
    REPORTING_BASELINES,
    EvaluatorConfig,
    PrefixBlockInfo,
    PrefixKVCacheEvaluator,
    PrefixKVCacheSimulator,
    RequestInfo,
    TrialMetrics,
    WorkloadRequest,
    _aggregate_trials,
    baseline_depth_prefer_shallow,
    baseline_future_reuse_heuristic,
    baseline_lfu_blocks,
    baseline_lru_blocks,
    baseline_no_cache,
    baseline_oracle_future_reuse,
    baseline_prefix_anchor,
    baseline_prefix_fanout,
    baseline_tenant_fair_lru,
    baseline_tinylfu_lru,
    build_workload,
    scoring_fn_complexity,
)
from randomize_evolve.problems.prefix_kv_cache import evaluator as levi_evaluator
from randomize_evolve.problems.prefix_kv_cache import runner as prefix_runner
from randomize_evolve.problems.prefix_kv_cache.runner import (
    _artifact_report_config,
    _baseline_report_headline,
    _config_from_args,
    _evaluate_candidate_program,
    _score_weight_sensitivity_rows,
    compare_baselines,
    save_run_artifacts,
    write_baseline_plots,
)
from randomize_evolve.problems.prefix_kv_cache.configuration import (
    PREFIX_KV_CONFIG_ENV,
    PREFIX_KV_QUICK_ENV,
    active_evaluator_config,
    evaluator_config_from_settings,
    load_evaluator_config,
)


class AdmitAllLRU:
    def on_request_start(self, request, now: int) -> None:
        return None

    def score_admission(self, block, now: int) -> float:
        return 1.0

    def score_eviction(self, block, now: int) -> float:
        return float(now - block.last_accessed_at)

    def on_cache_hit(self, block, request, now: int) -> None:
        return None

    def on_cache_miss(self, block, request, now: int) -> None:
        return None


def _block_info(**overrides) -> PrefixBlockInfo:
    values = {
        "block_id": 1,
        "prefix_hash": 1,
        "parent_hash": None,
        "depth": 2,
        "start_token": 0,
        "end_token": 8,
        "token_count": 8,
        "tenant_id": 0,
        "created_at": 0,
        "last_accessed_at": 3,
        "hit_count": 0,
        "descendant_count": 5,
        "active_ref_count": 0,
        "estimated_recompute_cost": 8.0,
    }
    values.update(overrides)
    return PrefixBlockInfo(**values)


def test_shared_system_prompt_lru_has_hits() -> None:
    config = EvaluatorConfig(
        request_count=36,
        seeds=(3,),
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(baseline_lru_blocks)

    metrics = result.workload_metrics["train/shared_system_prompt"]
    assert metrics["token_hit_rate"] > 0.25
    assert result.invalid_fraction == 0.0


def test_no_cache_zero_hits() -> None:
    config = EvaluatorConfig(request_count=24, seeds=(3,))
    result = PrefixKVCacheEvaluator(config)(baseline_no_cache)

    assert result.split_metrics["train"]["token_hit_rate"] == 0.0
    assert result.split_metrics["validation"]["token_hit_rate"] == 0.0
    assert result.invalid_fraction == 0.0


def test_block_recurrence_timestamps_use_only_prior_accesses() -> None:
    class CaptureRecurrence(AdmitAllLRU):
        def __init__(self) -> None:
            self.observations = []

        def on_cache_hit(self, block, request, now: int) -> None:
            self.observations.append(
                (now, block.prev_last_accessed_at, block.last_access_gap)
            )

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observations.append(
                (now, block.prev_last_accessed_at, block.last_access_gap)
            )

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=1,
            prompt_tokens=(1, 2, 3, 4),
            arrival_step=arrival_step,
        )
        for request_id, arrival_step in enumerate((2, 7, 11))
    )
    policy = CaptureRecurrence()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observations == [
        (2, None, None),
        (7, 2, 5),
        (11, 7, 4),
    ]


def test_block_access_gap_summary_is_bounded_and_deterministic() -> None:
    class CaptureGapSummary(AdmitAllLRU):
        def __init__(self) -> None:
            self.observations = []

        def on_cache_hit(self, block, request, now: int) -> None:
            self.observations.append((now, block.access_gap_mean, block.access_gap_var))

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observations.append((now, block.access_gap_mean, block.access_gap_var))

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=1,
            prompt_tokens=(1, 2, 3, 4),
            arrival_step=arrival_step,
        )
        for request_id, arrival_step in enumerate((0, 4, 10, 12))
    )
    policy = CaptureGapSummary()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observations[:2] == [(0, None, None), (4, None, None)]
    assert policy.observations[2] == pytest.approx((10, 4.5, 0.75))
    assert policy.observations[3] == pytest.approx((12, 3.875, 1.734375))
    block = next(iter(simulator.blocks.values()))
    for now in range(13, 10_013):
        simulator._record_access(block, now)
    assert block.access_gap_sample_count == 2
    assert isinstance(block.access_gap_mean, float)
    assert isinstance(block.access_gap_mean_square, float)


def test_subtree_aggregates_include_known_descendants() -> None:
    class CaptureSubtree(AdmitAllLRU):
        def __init__(self) -> None:
            self.hit_observations = []

        def on_cache_hit(self, block, request, now: int) -> None:
            self.hit_observations.append(
                (
                    request.request_id,
                    block.depth,
                    block.subtree_hit_rate,
                    block.active_ref_count,
                    block.subtree_active_ref_count,
                )
            )

    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=8,
                priority=0,
                request_type="unit",
                prompt_tokens=(),
            ),
            true_output_length=128,
            prompt_tokens=tuple(range(8)),
            arrival_step=request_id,
        )
        for request_id in range(2)
    )
    policy = CaptureSubtree()
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.hit_observations == [
        (1, 1, 0.25, 2, 3),
        (1, 2, 0.5, 2, 2),
    ]
    root = min(simulator.blocks.values(), key=lambda block: block.depth)
    assert simulator._subtree_hit_counts[root.prefix_hash] >= root.hit_count
    assert (
        simulator._subtree_active_ref_counts[root.prefix_hash] >= root.active_ref_count
    )


def test_request_regime_context_is_bounded_and_independent_of_request_type() -> None:
    class CaptureRegime(AdmitAllLRU):
        def __init__(self) -> None:
            self.observations = []

        def on_request_start(self, request, now: int) -> None:
            self.observations.append(
                (
                    request.request_type,
                    request.recent_admission_pressure,
                    request.recent_miss_rate,
                )
            )

    def run(request_types):
        prompts = ((1, 1, 1, 1), (2, 2, 2, 2)) + ((2, 2, 2, 2),) * 38
        requests = tuple(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=0,
                    session_id=0,
                    prompt_length=4,
                    priority=0,
                    request_type=request_type,
                    prompt_tokens=(),
                ),
                true_output_length=1,
                prompt_tokens=prompt,
                arrival_step=request_id,
            )
            for request_id, (request_type, prompt) in enumerate(
                zip(request_types, prompts, strict=True)
            )
        )
        policy = CaptureRegime()
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=1,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
        )
        simulator.run(policy, requests, split="train", workload="unit", seed=1)
        return policy.observations, simulator

    first, simulator = run(tuple(f"type_{index}" for index in range(40)))
    second, _ = run(tuple("different" for _ in range(40)))

    assert first[0][1:] == (0.0, 0.0)
    assert first[1][1:] == (1.0, 1.0)
    assert first[2][1:] == (1.0, 1.0)
    assert first[3][1:] == pytest.approx((1.0, 2.0 / 3.0))
    assert [observation[1:] for observation in first] == [
        observation[1:] for observation in second
    ]
    assert simulator._recent_admission_pressure.maxlen == 32
    assert simulator._recent_miss_rates.maxlen == 32
    assert len(simulator._recent_admission_pressure) == 32
    assert len(simulator._recent_miss_rates) == 32


def test_discrete_baselines_break_equal_priority_ties_with_lru() -> None:
    older = _block_info(last_accessed_at=1)
    newer = _block_info(last_accessed_at=9)

    for factory in (
        baseline_lfu_blocks,
        baseline_depth_prefer_shallow,
        baseline_prefix_fanout,
    ):
        policy = factory(8, 4)
        assert policy.score_eviction(older, now=10) > policy.score_eviction(
            newer, now=10
        )


def test_lfu_still_prefers_to_evict_a_less_frequent_block() -> None:
    unused = _block_info(last_accessed_at=9, hit_count=0)
    frequent = _block_info(last_accessed_at=1, hit_count=1)
    policy = baseline_lfu_blocks(8, 4)

    assert policy.score_eviction(unused, now=10) > policy.score_eviction(
        frequent, now=10
    )


def test_oracle_evicts_furthest_next_reuse_even_if_it_is_more_frequent() -> None:
    sooner_once = _block_info(
        estimated_future_reuse=1.0,
        estimated_next_reuse_distance=2.0,
    )
    later_often = _block_info(
        block_id=2,
        prefix_hash=2,
        estimated_future_reuse=10.0,
        estimated_next_reuse_distance=10.0,
    )
    heuristic = baseline_future_reuse_heuristic(8, 4)
    oracle = baseline_oracle_future_reuse(8, 4)

    assert heuristic.score_eviction(sooner_once, now=0) > heuristic.score_eviction(
        later_often, now=0
    )
    assert oracle.score_eviction(later_often, now=0) > oracle.score_eviction(
        sooner_once, now=0
    )


def test_tenant_fair_lru_prefers_eviction_from_better_served_tenant() -> None:
    served = _block_info(tenant_id=0)
    underserved = _block_info(block_id=2, prefix_hash=2, tenant_id=1)
    request = RequestInfo(
        request_id=0,
        tenant_id=0,
        session_id=0,
        prompt_length=8,
        priority=0,
        request_type="unit",
        prompt_tokens=(),
    )
    policy = baseline_tenant_fair_lru(8, 4)

    policy.on_cache_hit(served, request, now=0)
    policy.on_cache_miss(underserved, request, now=0)

    assert policy.score_eviction(served, now=10) > policy.score_eviction(
        underserved, now=10
    )


def test_tenant_fair_lru_reduces_multi_tenant_fairness_gap() -> None:
    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("multi_tenant_skew",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))

    lru = evaluator(baseline_lru_blocks)
    tenant_fair = evaluator(baseline_tenant_fair_lru)

    lru_gap = lru.workload_metrics["validation/multi_tenant_skew"][
        "tenant_fairness_penalty"
    ]
    tenant_fair_gap = tenant_fair.workload_metrics["validation/multi_tenant_skew"][
        "tenant_fairness_penalty"
    ]
    assert tenant_fair_gap < lru_gap


def test_prefix_fanout_does_not_regress_lru_on_branching() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("agent_trace_branching",),
    )
    lru = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    fanout = PrefixKVCacheEvaluator(config, splits=("validation",))(
        baseline_prefix_fanout
    )

    lru_hit_rate = lru.workload_metrics["validation/agent_trace_branching"][
        "token_hit_rate"
    ]
    fanout_hit_rate = fanout.workload_metrics["validation/agent_trace_branching"][
        "token_hit_rate"
    ]
    assert fanout_hit_rate >= lru_hit_rate


def test_adversarial_over_admission_high_churn() -> None:
    config = EvaluatorConfig(
        request_count=36,
        seeds=(3,),
        capacity_blocks=8,
        hidden_families=("adversarial_unique_prompts",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("hidden",))(
        lambda *_: AdmitAllLRU()
    )
    metrics = result.workload_metrics["hidden/adversarial_unique_prompts"]

    assert metrics["token_hit_rate"] == 0.0
    assert metrics["cache_churn_per_1k"] > 2500.0
    assert result.invalid_fraction == 0.0


def test_invalid_candidate_penalized() -> None:
    class BadPolicy(AdmitAllLRU):
        def score_admission(self, block, now: int) -> float:
            return float("nan")

    config = EvaluatorConfig(request_count=12, seeds=(3,))
    invalid = PrefixKVCacheEvaluator(config)(lambda *_: BadPolicy())
    valid_scores = [
        PrefixKVCacheEvaluator(config)(factory).combined_score
        for factory in BASELINES.values()
    ]

    assert invalid.invalid_fraction > 0.0
    assert invalid.combined_score < min(valid_scores)
    assert invalid.success is False


def test_factory_internal_type_error_is_not_retried() -> None:
    calls = []

    def factory(capacity_blocks, block_size_tokens, seed=None):
        calls.append(seed)
        raise TypeError("internal construction failure")

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(factory)

    assert result.invalid_fraction == 1.0
    assert calls == [1003]


def test_missing_policy_hooks_are_structured_invalid_results() -> None:
    class MissingHooks:
        def score_admission(self, block, now: int) -> float:
            return -1.0

        def score_eviction(self, block, now: int) -> float:
            return 0.0

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(
        lambda *_: MissingHooks()
    )

    assert result.invalid_fraction == 1.0
    assert (
        result.workload_metrics["train/shared_system_prompt"]["invalid_reason"]
        == "policy must implement on_request_start()"
    )


def test_candidate_memory_limit_is_enforced() -> None:
    class MemoryHeavyPolicy(AdmitAllLRU):
        def __init__(self) -> None:
            self.payload = bytearray(16 * 1024)

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
        max_memory_bytes=1024,
    )

    result = PrefixKVCacheEvaluator(config, splits=("train",))(
        lambda *_: MemoryHeavyPolicy()
    )

    assert result.invalid_fraction == 1.0
    assert (
        "candidate used"
        in result.workload_metrics["train/shared_system_prompt"]["invalid_reason"]
    )


def test_evaluate_source_minimal_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(request_count=12, seeds=(3,)),
    )
    source = _minimal_policy_source("-1.0", "0.0")

    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["success"] is True
    assert result.metrics["combined_score"] < 0.0
    assert result.artifacts["candidate_metadata"]["scoring_fn_complexity"] > 0
    assert (
        result.artifacts["score_breakdown"]["combined_score"]
        == result.metrics["combined_score"]
    )
    assert result.artifacts["score_breakdown"]["complexity_cost"] > 0.0


def test_evaluate_factory_uses_configured_timeout(monkeypatch) -> None:
    captured = {}
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        timeout_s=0.25,
    )
    monkeypatch.setattr(levi_evaluator, "DEFAULT_CONFIG", config)

    def fake_run_with_timeout(func, *args, timeout_seconds, **kwargs):
        captured["timeout_seconds"] = timeout_seconds
        return func(*args, **kwargs)

    monkeypatch.setattr(levi_evaluator, "run_with_timeout", fake_run_with_timeout)

    result = levi_evaluator.evaluate_factory(baseline_no_cache)

    assert result.metrics["success"] is True
    assert captured["timeout_seconds"] == 0.25


def test_evaluate_source_times_out_during_module_loading(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(timeout_s=0.01),
    )
    source = """
import time
time.sleep(0.5)
"""

    started = time.perf_counter()
    result = levi_evaluator.evaluate_source(source)

    assert result.metrics["error"] == "evaluation timed out"
    assert time.perf_counter() - started < 0.3


def test_root_anchored_match() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    request = WorkloadRequest(
        info=RequestInfo(
            request_id=0,
            tenant_id=0,
            session_id=0,
            prompt_length=8,
            priority=0,
            request_type="unit",
            prompt_tokens=tuple(range(8)),
        ),
        true_output_length=64,
    )
    blocks = simulator._materialize_chain(request, now=0)
    blocks[1].resident = True

    assert simulator.match_resident_prefix(blocks) == 0


def test_forced_bypass_not_invalid() -> None:
    class AdmitEverything(AdmitAllLRU):
        pass

    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        capacity_blocks=1,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(
        lambda *_: AdmitEverything()
    )
    metrics = result.workload_metrics["train/shared_system_prompt"]

    assert result.invalid_fraction == 0.0
    assert metrics["forced_bypass_count"] > 0


def test_pinned_blocks_are_released_after_generation_finishes() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        active_tokens_per_step=64,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple(range(request_id * 4, request_id * 4 + 4)),
            ),
            true_output_length=128 if request_id == 0 else 64,
        )
        for request_id in range(3)
    )

    metrics = simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.forced_bypass_count == 1
    assert metrics.forced_bypass_tokens == 4
    assert metrics.admission_count == 2
    assert metrics.eviction_count == 1


def test_re_admitted_block_becomes_most_recently_used() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=2,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        active_tokens_per_step=64,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple([token] * 4),
            ),
            true_output_length=1,
        )
        for request_id, token in enumerate((1, 2, 3, 1, 4))
    )

    simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    resident_token_sets = {
        request.info.prompt_tokens
        for request in requests
        if simulator.blocks[
            simulator._materialize_chain(request, now=5)[0].prefix_hash
        ].resident
    }
    assert resident_token_sets == {(1, 1, 1, 1), (4, 4, 4, 4)}


def test_cache_miss_charges_failed_lookup_probe() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=2.0,
        eviction_cost_per_block=0.0,
    )
    request = WorkloadRequest(
        info=RequestInfo(
            request_id=0,
            tenant_id=0,
            session_id=0,
            prompt_length=4,
            priority=0,
            request_type="unit",
            prompt_tokens=(1, 2, 3, 4),
        ),
        true_output_length=1,
    )

    metrics = simulator.run(
        baseline_no_cache(1, 4),
        (request,),
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.lookup_block_count == 1
    assert metrics.lookup_blocks_per_request == 1.0
    assert metrics.admission_score_count == 1
    assert metrics.admission_rejection_count == 1
    assert metrics.admission_rate == 0.0
    assert metrics.policy_bypass_tokens == 4
    assert metrics.forced_bypass_tokens == 0
    assert metrics.p95_latency_proxy == 6.0


def test_admission_lifecycle_and_short_reuse_regret_are_reported() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple([token] * 4),
            ),
            true_output_length=1,
        )
        for request_id, token in enumerate((1, 1, 2, 1))
    )

    metrics = simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.admission_count == 3
    assert metrics.useful_admission_count == 1
    assert metrics.wasted_admission_count == 2
    assert metrics.useful_admission_rate == 1 / 3
    assert metrics.wasted_admission_rate == 2 / 3
    assert metrics.admitted_token_count == 12
    assert metrics.useful_admission_token_count == 4
    assert metrics.wasted_admission_token_count == 8
    assert metrics.useful_admission_token_rate == 1 / 3
    assert metrics.wasted_admission_token_rate == 2 / 3
    assert metrics.admission_saved_tokens == 4
    assert metrics.admission_saved_tokens_per_admission == 4 / 3
    assert metrics.admission_token_utility == 1 / 3
    assert metrics.evicted_without_hit_count == 1
    assert metrics.short_reuse_after_eviction_missed_tokens == 4
    assert metrics.short_reuse_after_eviction_missed_token_rate > 0.0
    assert metrics.eviction_reuse_distance_p50 == 1.0


def test_admission_token_utility_distinguishes_full_and_partial_blocks() -> None:
    def run(tokens: tuple[int, ...]) -> TrialMetrics:
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=1,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
        )
        requests = tuple(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=0,
                    session_id=0,
                    prompt_length=len(tokens),
                    priority=0,
                    request_type="unit",
                    prompt_tokens=tokens,
                ),
                true_output_length=1,
            )
            for request_id in range(2)
        )
        return simulator.run(
            AdmitAllLRU(),
            requests,
            split="train",
            workload="unit",
            seed=1,
        )

    full = run((1, 1, 1, 1))
    partial = run((2,))

    assert full.useful_admission_rate == partial.useful_admission_rate == 1.0
    assert full.admission_saved_tokens_per_admission == 4.0
    assert partial.admission_saved_tokens_per_admission == 1.0
    assert full.admission_token_utility == 1.0
    assert partial.admission_token_utility == 0.25


def test_token_weighted_admission_waste_reflects_block_size() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=2,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    request_tokens = ((1,), (1,), (2, 2, 2, 2))
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=len(tokens),
                priority=0,
                request_type="unit",
                prompt_tokens=tokens,
            ),
            true_output_length=1,
        )
        for request_id, tokens in enumerate(request_tokens)
    )

    metrics = simulator.run(
        AdmitAllLRU(),
        requests,
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.useful_admission_count == 1
    assert metrics.wasted_admission_count == 1
    assert metrics.wasted_admission_rate == 0.5
    assert metrics.admitted_token_count == 5
    assert metrics.useful_admission_token_count == 1
    assert metrics.wasted_admission_token_count == 4
    assert metrics.wasted_admission_token_rate == 0.8


def test_avoidable_eviction_audit_distinguishes_lru_from_oracle() -> None:
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=4,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple([token] * 4),
            ),
            true_output_length=1,
        )
        for request_id, token in enumerate((1, 2, 1, 3, 2))
    )

    def run(factory, *, expose_future_reuse: bool) -> TrialMetrics:
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=2,
            block_size_tokens=4,
            prefill_cost_per_token=1.0,
            lookup_cost_per_block=0.0,
            eviction_cost_per_block=0.0,
            expose_future_reuse=expose_future_reuse,
        )
        return simulator.run(
            factory(2, 4),
            requests,
            split="train",
            workload="unit",
            seed=1,
        )

    lru = run(baseline_lru_blocks, expose_future_reuse=False)
    oracle = run(baseline_oracle_future_reuse, expose_future_reuse=True)

    assert lru.avoidable_eviction_count == 1
    assert lru.avoidable_short_reuse_eviction_count == 1
    assert oracle.avoidable_eviction_count == 0
    assert oracle.token_hit_rate > lru.token_hit_rate


def test_temporal_and_tenant_tail_metrics_expose_service_collapse() -> None:
    simulator = PrefixKVCacheSimulator(
        capacity_blocks=1,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
    )
    requests = []
    for request_id in range(8):
        token = 1 if request_id < 4 else request_id
        tenant_id = 0 if request_id < 6 else 1
        requests.append(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    session_id=0,
                    prompt_length=4,
                    priority=0,
                    request_type="unit",
                    prompt_tokens=tuple([token] * 4),
                ),
                true_output_length=1,
            )
        )

    metrics = simulator.run(
        AdmitAllLRU(),
        tuple(requests),
        split="train",
        workload="unit",
        seed=1,
    )

    assert metrics.token_hit_rate > metrics.worst_quarter_token_hit_rate
    assert metrics.final_quarter_token_hit_rate == 0.0
    assert metrics.quarter_token_hit_rate_stddev > 0.0
    assert metrics.tenant_count == 2
    assert metrics.tenant_token_hit_rate_p10 == 0.0
    assert metrics.tenant_jain_fairness < 1.0


def test_hidden_not_in_combined_score(monkeypatch) -> None:
    config_a = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        hidden_families=("adversarial_unique_prompts",),
    )
    config_b = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        hidden_families=("cross_family_mixture",),
    )
    monkeypatch.setattr(levi_evaluator, "DEFAULT_CONFIG", config_a)
    first = levi_evaluator.evaluate_factory(baseline_lru_blocks)
    monkeypatch.setattr(levi_evaluator, "DEFAULT_CONFIG", config_b)
    second = levi_evaluator.evaluate_factory(baseline_lru_blocks)

    assert first.metrics["combined_score"] == second.metrics["combined_score"]
    assert "hidden" not in first.artifacts["split_metrics"]
    assert all(
        not key.startswith("hidden/") for key in first.artifacts["workload_metrics"]
    )


def test_structure_probe_not_in_combined_selection_score() -> None:
    config_a = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        validation_families=("hotset_cold_scan",),
        probe_families=("agent_trace_branching",),
    )
    config_b = replace(
        config_a,
        probe_families=("cyclic_working_set_pressure",),
    )

    first = PrefixKVCacheEvaluator(config_a, splits=("validation", "probe"))(
        baseline_lru_blocks
    )
    second = PrefixKVCacheEvaluator(config_b, splits=("validation", "probe"))(
        baseline_lru_blocks
    )

    assert first.combined_score == second.combined_score
    assert "probe/agent_trace_branching" in first.workload_metrics
    assert "probe/cyclic_working_set_pressure" in second.workload_metrics
    assert first.candidate_metadata["selection_invalid_fraction"] == 0.0


def test_structure_probe_invalidity_is_quarantined_from_selection() -> None:
    config = EvaluatorConfig(
        request_count=12,
        seeds=(3,),
        validation_families=("hotset_cold_scan",),
        probe_families=("agent_trace_branching",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation", "probe"))
    valid = evaluator(baseline_lru_blocks)
    trials = [
        replace(
            trial,
            invalid=True,
            invalid_reason="probe-only failure",
        )
        if trial.split == "probe"
        else trial
        for trial in valid.trials
    ]

    rescored = evaluator.rescore_trials(trials)

    assert rescored.combined_score == valid.combined_score
    assert rescored.success is True
    assert rescored.invalid_fraction == 0.0
    assert rescored.candidate_metadata["reporting_invalid_fraction"] > 0.0
    assert rescored.split_metrics["probe"]["invalid_fraction"] == 1.0


def test_baselines_separate_on_validation() -> None:
    config = EvaluatorConfig(request_count=48, seeds=(3,), capacity_blocks=12)
    scores = {
        name: PrefixKVCacheEvaluator(config, splits=("validation",))(
            factory
        ).combined_score
        for name, factory in BASELINES.items()
    }

    assert len({round(score, 6) for score in scores.values()}) >= 5
    assert max(scores.values()) - min(scores.values()) > 50.0


def test_reporting_baseline_suite_includes_credibility_baselines() -> None:
    assert {
        "lru",
        "lfu",
        "cost_aware_lru",
        "prefix_anchor",
        "tinylfu_lru",
        "oracle_future_reuse",
    }.issubset(REPORTING_BASELINES)


def test_candidate_program_can_be_compared_against_baselines(tmp_path, capsys) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text(
        textwrap.dedent(
            """
            class NoCachePolicy:
                def on_request_start(self, request, now):
                    pass

                def score_admission(self, block, now):
                    return -1.0

                def score_eviction(self, block, now):
                    return 0.0

                def on_cache_hit(self, block, request, now):
                    pass

                def on_cache_miss(self, block, request, now):
                    pass


            def build_candidate(capacity_blocks, block_size_tokens, seed=None):
                return NoCachePolicy()
            """
        ),
        encoding="utf-8",
    )

    compare_baselines(
        quick=True,
        capacity_sweep_blocks=(8, 16),
        candidate_program=tmp_path,
    )

    output = capsys.readouterr().out
    assert "SMOKE-ONLY" in output
    assert "candidate: combined_score=" in output
    assert "capacity_8:" in output
    assert "capacity_16:" in output
    assert "lru: combined_score=" in output
    assert "[deployable]" in output
    assert "future_reuse_heuristic: combined_score=" in output
    assert "oracle_future_reuse: combined_score=" in output
    assert "[reporting-only/future-knowledge]" in output
    report = (tmp_path / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "Candidate `scoring_fn_complexity`" in report
    assert (
        "Smoke-only output; run the full panel before comparing policy rank." in report
    )


def test_candidate_program_comparison_applies_complexity_penalty(tmp_path) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text(
        textwrap.dedent(
            """
            class VerbosePolicy:
                def on_request_start(self, request, now):
                    pass

                def score_admission(self, block, now):
                    return -1.0

                def score_eviction(self, block, now):
                    return 0.0

                def on_cache_hit(self, block, request, now):
                    pass

                def on_cache_miss(self, block, request, now):
                    pass


            def build_candidate(capacity_blocks, block_size_tokens, seed=None):
                return VerbosePolicy()
            """
        ),
        encoding="utf-8",
    )

    result = _evaluate_candidate_program(
        EvaluatorConfig(request_count=4, seeds=(1,), capacity_sweep_blocks=(8,)),
        candidate_path,
    )

    assert result.candidate_metadata["scoring_fn_complexity"] > 0


def test_baseline_report_headline_does_not_overstate_candidate() -> None:
    def result(score: float) -> SimpleNamespace:
        return SimpleNamespace(combined_score=score)

    headline = _baseline_report_headline(
        [
            ("oracle_future_reuse", result(90.0)),
            ("tinylfu_lru", result(70.0)),
            ("candidate", result(60.0)),
            ("lru", result(50.0)),
        ]
    )

    assert headline == (
        "The candidate ranking is shown against deployable and reporting-only baselines."
    )


def test_baseline_report_headline_states_mixed_future_knowledge_ordering() -> None:
    def result(score: float) -> SimpleNamespace:
        return SimpleNamespace(combined_score=score)

    headline = _baseline_report_headline(
        [
            ("oracle_future_reuse", result(90.0)),
            ("candidate", result(80.0)),
            ("future_reuse_heuristic", result(70.0)),
            ("tinylfu_lru", result(60.0)),
        ]
    )

    assert headline == (
        "The candidate clears the deployable credibility baselines in this capacity "
        "sweep. It trails `oracle_future_reuse`. It beats `future_reuse_heuristic`."
    )


def test_complexity_counts_candidate_helper_methods() -> None:
    compact = """
class Policy:
    def score_admission(self, block, now):
        return 1.0

    def score_eviction(self, block, now):
        return 0.0


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""
    helper_heavy = """
class Policy:
    def score_admission(self, block, now):
        return self._helper(block)

    def score_eviction(self, block, now):
        return 0.0

    def _helper(self, block):
        total = 0.0
        for index in range(8):
            total += index * 0.25
        return total


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return Policy()
"""

    assert scoring_fn_complexity(helper_heavy) > scoring_fn_complexity(compact)


def test_complexity_counts_nested_factory_policy_methods() -> None:
    nested_policy = """
from types import SimpleNamespace


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    def score_admission(block, now):
        total = 0.0
        for index in range(8):
            total += index * block.depth
        return total

    def score_eviction(block, now):
        return 0.0

    return SimpleNamespace(
        score_admission=score_admission,
        score_eviction=score_eviction,
    )
"""

    assert scoring_fn_complexity(nested_policy) > 0


def test_form_aware_complexity_subsidizes_only_canonical_primitives() -> None:
    primitive_composer = """
from randomize_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay


class Policy:
    def __init__(self):
        self.decay = MultiTimescaleDecay((4.0,))

    def on_cache_hit(self, block, request, now):
        self.decay.observe(block.prefix_hash, 1.0, now)

    def score_admission(self, block, now):
        return self.decay.values(block.prefix_hash, now)[0]

    def score_eviction(self, block, now):
        return -self.decay.values(block.prefix_hash, now)[0]
"""
    hand_rolled = """
class Policy:
    def __init__(self):
        self.values_by_key = {}

    def on_cache_hit(self, block, request, now):
        value = self._value(block, now)
        self.values_by_key[block.prefix_hash] = (value + 1.0, now)

    def _value(self, block, now):
        value, observed_at = self.values_by_key.get(block.prefix_hash, (0.0, now))
        return value * 2.0 ** (-(now - observed_at) / 4.0)

    def score_admission(self, block, now):
        return self._value(block, now)

    def score_eviction(self, block, now):
        return -self._value(block, now)
"""

    primitive_raw = scoring_fn_complexity(primitive_composer)
    primitive_form_aware = scoring_fn_complexity(
        primitive_composer,
        form_aware=True,
    )
    hand_raw = scoring_fn_complexity(hand_rolled)
    hand_form_aware = scoring_fn_complexity(hand_rolled, form_aware=True)

    assert primitive_form_aware < primitive_raw
    assert hand_form_aware == hand_raw
    assert primitive_form_aware < hand_form_aware
    assert hand_raw <= math.ceil(1.6 * primitive_raw)
    assert primitive_form_aware >= math.ceil(0.75 * primitive_raw)


def test_active_complexity_mode_is_configurable(monkeypatch) -> None:
    source = """
from randomize_evolve.problems.prefix_kv_cache.primitives import MultiTimescaleDecay


class Policy:
    def __init__(self):
        self.decay = MultiTimescaleDecay((4.0,))

    def score_admission(self, block, now):
        return self.decay.combine(block.prefix_hash, now, (1.0,))
"""
    raw = scoring_fn_complexity(source)
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(form_aware_complexity=False),
    )
    assert levi_evaluator._source_complexity(source) == raw
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(form_aware_complexity=True),
    )
    assert levi_evaluator._source_complexity(source) < raw


def test_complexity_penalty_is_unbounded_and_concave() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [TrialMetrics(split="validation", workload="unit", seed=1)]
    penalties = [
        -evaluator._score_trials(trials, invalid_fraction=0.0, complexity=complexity)
        for complexity in (3_000, 4_000, 5_000)
    ]

    assert penalties[0] < penalties[1] < penalties[2]
    assert penalties[2] - penalties[1] < penalties[1] - penalties[0]


def test_invalid_score_is_below_large_representative_valid_complexity() -> None:
    config = EvaluatorConfig()
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [TrialMetrics(split="validation", workload="unit", seed=1)]

    invalid_score = evaluator._score_trials(trials, invalid_fraction=1.0, complexity=0)
    valid_score = evaluator._score_trials(
        trials, invalid_fraction=0.0, complexity=100_000
    )

    assert invalid_score < valid_score


def test_score_combines_mean_and_min_workload_score() -> None:
    config = EvaluatorConfig(
        w_avg_tok=100.0,
        w_avg_blk=0.0,
        min_workload_weight=0.5,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="strong",
            seed=1,
            token_hit_rate=0.8,
        ),
        TrialMetrics(
            split="validation",
            workload="weak",
            seed=1,
            token_hit_rate=0.2,
        ),
    ]

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0) == 60.0


def test_score_blends_mean_with_worst_seed() -> None:
    config = EvaluatorConfig(
        w_avg_tok=100.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        min_seed_weight=0.25,
        request_tail_weight=0.0,
        worst_window_weight=0.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=0.0,
        avoidable_eviction_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="unit",
            seed=1,
            token_hit_rate=0.8,
        ),
        TrialMetrics(
            split="validation",
            workload="unit",
            seed=2,
            token_hit_rate=0.4,
        ),
    ]

    assert math.isclose(
        evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0),
        55.0,
    )


def test_score_rewards_temporal_tail_and_penalizes_avoidable_evictions() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        min_seed_weight=0.0,
        request_tail_weight=10.0,
        worst_window_weight=20.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=6.0,
        avoidable_eviction_weight=8.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trial = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        request_token_hit_rate_p10=0.4,
        worst_quarter_token_hit_rate=0.5,
        wasted_admission_rate=0.75,
        wasted_admission_token_rate=0.25,
        avoidable_eviction_rate=0.125,
    )

    assert evaluator._score_trials([trial], invalid_fraction=0.0, complexity=0) == 11.5


def test_score_rewards_token_utility_concavely() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        min_seed_weight=0.0,
        request_tail_weight=0.0,
        worst_window_weight=0.0,
        priority_hit_weight=0.0,
        wasted_admission_weight=0.0,
        admission_utility_weight=2.0,
        avoidable_eviction_weight=0.0,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    full = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        admission_token_utility=1.0,
    )
    partial = TrialMetrics(
        split="validation",
        workload="unit",
        seed=1,
        admission_token_utility=0.25,
    )

    full_score = evaluator._score_trials([full], invalid_fraction=0.0, complexity=0)
    partial_score = evaluator._score_trials(
        [partial], invalid_fraction=0.0, complexity=0
    )

    assert math.isclose(full_score, 2.0 * math.log1p(1.0))
    assert math.isclose(partial_score, 2.0 * math.log1p(0.25))
    assert full_score > partial_score


def test_auto_latency_normalization_is_scoped_per_workload() -> None:
    config = EvaluatorConfig(
        w_avg_tok=0.0,
        w_avg_blk=0.0,
        min_workload_weight=0.0,
        latency_weight=100.0,
        latency_cap=1_000.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="short",
            seed=1,
            p95_latency_proxy=50.0,
            max_prefill_cost=100.0,
        ),
        TrialMetrics(
            split="validation",
            workload="long",
            seed=1,
            p95_latency_proxy=0.0,
            max_prefill_cost=10_000.0,
        ),
    ]

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0) == -25.0


def test_capacity_sweep_reports_capacity_metrics() -> None:
    config = EvaluatorConfig(
        request_count=24,
        seeds=(3,),
        capacity_blocks=12,
        capacity_sweep_blocks=(8, 16),
        validation_families=("agent_trace_branching",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)

    assert set(result.capacity_metrics) == {"capacity_8", "capacity_16"}
    assert {trial.capacity_blocks for trial in result.trials} == {8, 16}
    assert result.candidate_metadata["capacity_sweep_blocks"] == "8,16"
    assert result.candidate_metadata["complexity_exponent"] == 0.75
    assert result.candidate_metadata["complexity_mode"] == "legacy_ast_nodes"


def test_aggregate_trials_preserves_peak_active_request_count() -> None:
    metrics = _aggregate_trials(
        [
            TrialMetrics(
                split="validation",
                workload="unit",
                seed=1,
                active_request_count_peak=3,
            ),
            TrialMetrics(
                split="validation",
                workload="unit",
                seed=2,
                active_request_count_peak=11,
            ),
        ]
    )

    assert metrics["active_request_count_peak"] == 11


def test_score_min_term_includes_capacity_variants() -> None:
    config = EvaluatorConfig(
        w_avg_tok=100.0,
        w_avg_blk=0.0,
        min_workload_weight=0.5,
        latency_weight=0.0,
        churn_weight=0.0,
        fairness_weight=0.0,
        k_complex=0.0,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    trials = [
        TrialMetrics(
            split="validation",
            workload="agentic",
            seed=1,
            capacity_blocks=24,
            token_hit_rate=0.9,
        ),
        TrialMetrics(
            split="validation",
            workload="agentic",
            seed=1,
            capacity_blocks=48,
            token_hit_rate=0.1,
        ),
    ]

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=0) == 55.0


def test_complexity_penalty_orders(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(request_count=12, seeds=(3,)),
    )
    simple = levi_evaluator.evaluate_source(_minimal_policy_source("-1.0", "0.0"))
    complex_source = _minimal_policy_source(
        "-1.0 + 0.0 * (block.depth + block.hit_count + block.descendant_count)",
        "0.0 + 0.0 * (now + block.depth + block.hit_count + block.token_count)",
    )
    complex_result = levi_evaluator.evaluate_source(complex_source)

    assert complex_result.metrics["success"] is True
    assert simple.metrics["combined_score"] > complex_result.metrics["combined_score"]
    assert (
        simple.artifacts["candidate_metadata"]["scoring_fn_complexity"]
        < complex_result.artifacts["candidate_metadata"]["scoring_fn_complexity"]
    )


def test_existing_trials_can_be_rescored_without_rerunning_simulation() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("hotset_cold_scan",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    stricter = replace(config, churn_weight=1.0)

    rescored = PrefixKVCacheEvaluator(
        stricter,
        splits=("validation",),
    ).rescore_trials(result.trials)

    assert rescored.trials == result.trials
    assert rescored.combined_score < result.combined_score


def test_score_weight_sensitivity_uses_fixed_trials() -> None:
    config = EvaluatorConfig(
        request_count=36,
        seeds=(3,),
        capacity_blocks=8,
        validation_families=("hotset_cold_scan",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    results = {
        "candidate": evaluator(baseline_lru_blocks),
        "no_cache": evaluator(baseline_no_cache),
    }

    rows = _score_weight_sensitivity_rows(
        results,
        config,
        weights=("churn_weight",),
        factors=(0.0, 1.0, 2.0),
    )

    assert len(rows) == 3
    assert rows[1]["candidate_score"] == results["candidate"].combined_score
    assert rows[2]["candidate_score"] <= rows[1]["candidate_score"]


def test_evaluate_hidden_is_separate(monkeypatch) -> None:
    monkeypatch.setattr(
        levi_evaluator,
        "DEFAULT_CONFIG",
        EvaluatorConfig(request_count=12, seeds=(3,)),
    )

    result = levi_evaluator.evaluate_hidden(baseline_lru_blocks)

    assert "hidden" in result.artifacts["split_metrics"]
    assert result.metrics["success"] is True


def test_runner_default_report_matches_levi_capacity_sweep() -> None:
    default_config = _config_from_args(
        quick=True,
        capacity_blocks=None,
        block_size_tokens=None,
    )
    explicit_config = _config_from_args(
        quick=True,
        capacity_blocks=12,
        block_size_tokens=None,
    )

    assert default_config.effective_capacity_blocks() == (24, 48)
    assert explicit_config.effective_capacity_blocks() == (12,)


def test_saved_artifact_report_uses_full_panel() -> None:
    config = _artifact_report_config()

    assert config.request_count == 96
    assert config.seeds == (11, 23, 37)
    assert config.effective_capacity_blocks() == (24, 48)


def test_candidate_prompt_names_only_supported_lifecycle_callbacks() -> None:
    config = prefix_runner._CONFIG_LOADER.load(Path("configs/prefix_kv_cache.yaml"))
    message = config.raw["prompt"]["system_message"]

    assert config.run_cost == {}
    assert "No other lifecycle callback fires." in message
    assert "session_id is request-only metadata" in message
    assert "now argument is a logical arrival step" in message
    assert "Priority is a deployable request signal, not proof of reuse" in message
    assert "Long-horizon tenant workloads repeatedly shift" in message
    for callback in (
        "on_request_start",
        "on_cache_hit",
        "on_cache_miss",
        "on_request_end",
        "on_block_admitted",
        "on_block_evicted",
    ):
        assert callback in message


def test_candidate_config_matches_default_verifier_panel_and_score_weights() -> None:
    config = prefix_runner._CONFIG_LOADER.load(Path("configs/prefix_kv_cache.yaml"))
    settings = config.raw["problem"]["settings"]
    default = EvaluatorConfig()
    loaded = load_evaluator_config(Path("configs/prefix_kv_cache.yaml"))

    assert tuple(settings["train_families"]) == default.train_families
    assert tuple(settings["validation_families"]) == default.validation_families
    assert tuple(settings["probe_families"]) == default.probe_families
    assert tuple(settings["hidden_families"]) == default.hidden_families
    assert loaded.form_aware_complexity is True
    assert loaded.family_request_multipliers == default.family_request_multipliers
    assert loaded.timeout_s == 90
    for field in (
        "w_avg_tok",
        "w_avg_blk",
        "min_workload_weight",
        "min_seed_weight",
        "request_tail_weight",
        "worst_window_weight",
        "priority_hit_weight",
        "wasted_admission_weight",
        "admission_utility_weight",
        "avoidable_eviction_weight",
        "latency_weight",
        "latency_cap",
        "churn_weight",
        "churn_cap",
        "fairness_weight",
        "fairness_cap",
        "k_complex",
        "complexity_exponent",
        "v_min",
        "invalid_surcharge",
    ):
        assert settings["scoring"][field] == getattr(default, field)


def test_candidate_yaml_contains_only_forwarded_top_level_fields() -> None:
    config = prefix_runner._CONFIG_LOADER.load(Path("configs/prefix_kv_cache.yaml"))
    raw = config.raw

    assert "checkpoint_interval" not in raw
    assert set(raw["llm"]) == {
        "primary_model",
        "temperature",
        "max_tokens",
    }
    assert set(raw["evaluator"]) == {
        "timeout",
        "cascade_evaluation",
        "parallel_evaluations",
    }
    assert set(raw["search"]) == {"notes"}
    assert raw["pipeline"]["output_mode"] == "diff"
    assert raw["init"] == {"n_diverse_seeds": 0, "n_variants_per_seed": 8}
    assert raw["cvt"] == {"n_centroids": 8, "data_driven_centroids": True}
    assert raw["meta_advice"] == {
        "enabled": True,
        "interval": 24,
        "max_tokens": 500,
    }
    assert raw["punctuated_equilibrium"] == {"enabled": False}
    assert config.paradigm_model == "openai/gpt-5.4-mini"
    assert config.mutation_model == "openai/gpt-5.4-mini"
    assert config.evolve_kwargs()["cvt"] == raw["cvt"]
    assert config.evolve_kwargs()["meta_advice"] == raw["meta_advice"]


def test_prefix_evaluator_config_rejects_inactive_settings() -> None:
    with pytest.raises(ValueError, match="unsupported prefix KV-cache evaluator"):
        evaluator_config_from_settings({"seed_count": 3})

    with pytest.raises(ValueError, match="unsupported prefix KV-cache scoring"):
        evaluator_config_from_settings({"scoring": {"complexity_cap": 500}})


def test_active_evaluator_config_applies_quick_worker_override(monkeypatch) -> None:
    monkeypatch.setenv(PREFIX_KV_QUICK_ENV, "1")

    config = active_evaluator_config(EvaluatorConfig())

    assert config.request_count == 36
    assert config.seeds == (3,)
    assert config.family_request_multipliers == {}


def test_load_seed_program_source_accepts_saved_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate_path = run_dir / "best_program.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")

    source = prefix_runner._load_seed_program_source(run_dir)

    assert source.text() == candidate_path.read_text(encoding="utf-8")


def test_seed_program_cli_accepts_saved_run_directory(tmp_path) -> None:
    args = prefix_runner.build_arg_parser().parse_args(
        ["--seed-program", str(tmp_path)]
    )

    assert args.seed_program == tmp_path


def test_demo_run_evolution_uses_requested_seed_program(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_program.py").write_text("chosen seed\n", encoding="utf-8")
    captured = {}

    class FakeWorkflow:
        def execute(self, iterations):
            captured["iterations"] = iterations
            captured["config_path"] = os.environ[PREFIX_KV_CONFIG_ENV]
            captured["quick"] = os.environ[PREFIX_KV_QUICK_ENV]
            return SimpleNamespace()

    def fake_build_workflow(provider, *, program_source):
        captured["provider"] = provider
        captured["source"] = program_source.text()
        return FakeWorkflow()

    monkeypatch.setattr(prefix_runner, "_build_workflow", fake_build_workflow)

    prefix_runner.demo_run_evolution(
        iterations=7,
        quick=True,
        seed_program=run_dir,
        artifact_output=None,
    )

    assert captured["iterations"] == 7
    assert captured["source"] == "chosen seed\n"
    assert captured["config_path"] == str(
        Path("configs/prefix_kv_cache.yaml").resolve()
    )
    assert captured["quick"] == "1"


def test_hidden_report_evaluates_requested_candidate(
    tmp_path, monkeypatch, capsys
) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")
    captured = {}

    def fake_evaluate_candidate(config, path, *, splits):
        captured["path"] = path
        captured["splits"] = splits
        return SimpleNamespace(combined_score=12.5)

    monkeypatch.setattr(
        prefix_runner, "_evaluate_candidate_program", fake_evaluate_candidate
    )
    monkeypatch.setattr(prefix_runner, "REPORTING_BASELINES", {})

    prefix_runner.hidden_report(quick=True, candidate_program=candidate_path)

    assert captured == {"path": candidate_path, "splits": ("hidden",)}
    assert f"candidate={candidate_path}" in capsys.readouterr().out


def test_probe_report_evaluates_requested_candidate(
    tmp_path, monkeypatch, capsys
) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")
    output_path = tmp_path / "probe.json"
    captured = {}

    def fake_evaluate_candidate(config, path, *, splits):
        captured["path"] = path
        captured["splits"] = splits
        return SimpleNamespace(
            combined_score=12.5,
            success=True,
            invalid_fraction=0.0,
            split_metrics={"probe": {"token_hit_rate": 0.5}},
            workload_metrics={
                "probe/agent_trace_branching": {
                    "token_hit_rate": 0.5,
                    "block_hit_rate": 0.4,
                    "cache_churn_per_1k": 10.0,
                }
            },
            capacity_metrics={},
            candidate_metadata={},
            score_breakdown={"combined_score": 12.5},
        )

    monkeypatch.setattr(
        prefix_runner, "_evaluate_candidate_program", fake_evaluate_candidate
    )
    monkeypatch.setattr(prefix_runner, "REPORTING_BASELINES", {})

    payload = prefix_runner.probe_report(
        output_path=output_path,
        quick=True,
        candidate_program=candidate_path,
    )

    assert captured == {"path": candidate_path, "splits": ("probe",)}
    assert payload["selection_score_excludes_probe"] is True
    assert output_path.exists()
    assert f"structure_probe={output_path}" in capsys.readouterr().out


def test_workload_builder_uses_predicted_not_true_output_length() -> None:
    request = build_workload(
        "shared_system_prompt",
        request_count=1,
        block_size_tokens=8,
        seed=3,
    )[0]

    assert request.info.predicted_output_length is None
    assert request.info.prompt_tokens == ()
    assert request.prompt_tokens
    assert isinstance(request.true_output_length, int)


def test_session_continuation_growth_resumes_and_extends_prefix() -> None:
    requests = build_workload(
        "session_continuation_growth",
        request_count=8,
        block_size_tokens=8,
        seed=3,
    )

    first_turn = requests[0]
    resumed_session = requests[4]
    assert first_turn.info.session_id == resumed_session.info.session_id
    assert (
        resumed_session.prompt_tokens[: len(first_turn.prompt_tokens)]
        == first_turn.prompt_tokens
    )
    assert resumed_session.info.prompt_length == first_turn.info.prompt_length + 8


def test_agent_trace_branching_accumulates_tool_history_and_retries() -> None:
    requests = build_workload(
        "agent_trace_branching",
        request_count=48,
        block_size_tokens=8,
        seed=3,
    )

    prompt_lengths = [request.info.prompt_length for request in requests]
    request_types = {request.info.request_type for request in requests}
    assert max(prompt_lengths) > min(prompt_lengths) + 10 * 8
    assert request_types == {"agent_loop", "agent_retry"}


def test_stochastic_serving_mix_interleaves_classes_in_bursts() -> None:
    requests = build_workload(
        "stochastic_serving_mix",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    request_classes = [
        request.info.request_type.split("_", maxsplit=2)[1] for request in requests
    ]
    assert len(set(request_classes)) >= 4
    assert any(
        request_classes[index]
        == request_classes[index + 1]
        != request_classes[index + 2]
        for index in range(len(request_classes) - 2)
    )
    assert any(
        request_classes[index] != request_classes[index + 1]
        for index in range(len(request_classes) - 1)
    )
    arrival_steps = [request.arrival_step for request in requests]
    assert all(step is not None for step in arrival_steps)
    arrival_gaps = [
        right - left for left, right in zip(arrival_steps, arrival_steps[1:])
    ]
    assert 0 in arrival_gaps
    assert max(arrival_gaps) > 1


def test_rolling_template_versions_models_canary_rollout_and_rollback() -> None:
    requests = build_workload(
        "rolling_template_versions",
        request_count=64,
        block_size_tokens=8,
        seed=3,
    )

    versions = [request.info.request_type for request in requests]
    assert set(versions[:16]) == {"rolling_template_v0"}
    assert set(versions[16:32]) == {"rolling_template_v0", "rolling_template_v1"}
    assert set(versions[32:48]) == {"rolling_template_v0", "rolling_template_v1"}
    assert versions[32:48].count("rolling_template_v1") > versions[32:48].count(
        "rolling_template_v0"
    )
    assert set(versions[48:]) == {"rolling_template_v0", "rolling_template_v1"}
    assert versions[48:].count("rolling_template_v0") > versions[48:].count(
        "rolling_template_v1"
    )


def test_heavy_tailed_prefix_lengths_include_expensive_outliers() -> None:
    requests = build_workload(
        "heavy_tailed_prefix_lengths",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    prompt_lengths = sorted(request.info.prompt_length for request in requests)
    median_prompt_length = prompt_lengths[len(prompt_lengths) // 2]
    assert len(set(prompt_lengths)) >= 8
    assert prompt_lengths[-1] >= 2 * median_prompt_length


def test_priority_burst_recovery_models_qos_pollution_and_recovery() -> None:
    requests = build_workload(
        "priority_burst_recovery",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    request_types = {request.info.request_type for request in requests}
    priorities = {request.info.priority for request in requests}
    warm_prompts = {
        request.prompt_tokens
        for request in requests[:24]
        if request.info.request_type == "priority_hot_warm"
    }
    assert request_types == {
        "priority_hot_warm",
        "priority_hot_during_burst",
        "priority_background_scan",
        "priority_medium_recovery",
        "priority_hot_recovery",
    }
    assert priorities == {0, 1, 3}
    assert any(
        request.info.request_type == "priority_background_scan"
        for request in requests[24:72]
    )
    assert any(
        request.info.request_type == "priority_hot_recovery"
        and request.prompt_tokens in warm_prompts
        for request in requests[72:]
    )

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("priority_burst_recovery",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/priority_burst_recovery"]
    assert metrics["recovery_request_count"] > 0
    assert metrics["recovery_token_hit_rate"] > 0.0
    assert metrics["recovery_p95_latency_proxy"] > 0.0


def test_priority_aware_admission_protects_high_priority_hit_rate() -> None:
    class PriorityAwareAdmission(AdmitAllLRU):
        def __init__(self) -> None:
            self.priority = 0

        def on_request_start(self, request, now: int) -> None:
            self.priority = request.priority

        def score_admission(self, block, now: int) -> float:
            return 1.0 if self.priority > 0 else -1.0

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("priority_burst_recovery",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    priority_aware = evaluator(lambda *_: PriorityAwareAdmission())
    lru_metrics = lru.workload_metrics["validation/priority_burst_recovery"]
    priority_metrics = priority_aware.workload_metrics[
        "validation/priority_burst_recovery"
    ]

    assert (
        priority_metrics["high_priority_token_hit_rate"]
        > lru_metrics["high_priority_token_hit_rate"]
    )
    assert (
        priority_metrics["priority_weighted_token_hit_rate"]
        > lru_metrics["priority_weighted_token_hit_rate"]
    )
    assert priority_metrics["policy_bypass_token_rate"] > 0.0
    assert priority_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_cyclic_working_set_pressure_exposes_short_horizon_eviction_regret() -> None:
    requests = build_workload(
        "cyclic_working_set_pressure",
        request_count=96,
        block_size_tokens=8,
        seed=3,
    )

    assert {request.info.request_type for request in requests[:48]} == {
        "cyclic_working_set_small"
    }
    assert {request.info.request_type for request in requests[48:]} == {
        "cyclic_working_set_large"
    }
    assert len({request.prompt_tokens for request in requests[:48]}) == 9
    assert len({request.prompt_tokens for request in requests[48:]}) == 17

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("cyclic_working_set_pressure",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)
    lru_metrics = lru.workload_metrics["validation/cyclic_working_set_pressure"]
    tinylfu_metrics = tinylfu.workload_metrics["validation/cyclic_working_set_pressure"]

    assert lru_metrics["short_reuse_after_eviction_missed_token_rate"] > 0.0
    assert (
        tinylfu_metrics["short_reuse_after_eviction_missed_token_rate"]
        < lru_metrics["short_reuse_after_eviction_missed_token_rate"]
    )


def test_priority_one_off_noise_penalizes_blind_priority_admission() -> None:
    class PriorityOnlyAdmission(AdmitAllLRU):
        def __init__(self) -> None:
            self.priority = 0

        def on_request_start(self, request, now: int) -> None:
            self.priority = request.priority

        def score_admission(self, block, now: int) -> float:
            return 1.0 if self.priority > 0 else -1.0

    requests = build_workload(
        "priority_one_off_noise",
        request_count=50,
        block_size_tokens=8,
        seed=3,
    )
    high_priority = [
        request
        for request in requests
        if request.info.request_type == "priority_one_off_noise"
    ]
    normal = [
        request
        for request in requests
        if request.info.request_type == "priority_normal_recurring"
    ]
    assert len(high_priority) == 20
    assert len({request.prompt_tokens for request in high_priority}) == 20
    assert len({request.prompt_tokens for request in normal}) == 5

    config = EvaluatorConfig(
        request_count=96,
        seeds=(3,),
        capacity_blocks=24,
        validation_families=("priority_one_off_noise",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    priority_only = evaluator(lambda *_: PriorityOnlyAdmission())
    tinylfu = evaluator(baseline_tinylfu_lru)
    priority_metrics = priority_only.workload_metrics[
        "validation/priority_one_off_noise"
    ]
    tinylfu_metrics = tinylfu.workload_metrics["validation/priority_one_off_noise"]

    assert tinylfu_metrics["token_hit_rate"] > priority_metrics["token_hit_rate"]
    assert (
        tinylfu_metrics["wasted_admission_rate"]
        < priority_metrics["wasted_admission_rate"]
    )


def test_tenant_phase_shift_cycles_repeat_pollution_and_delayed_recovery() -> None:
    config = EvaluatorConfig(
        request_count=96,
        validation_families=("tenant_phase_shift_cycles",),
    )
    workload_config = config.workload_configs(("validation",))[0]
    requests = build_workload(
        workload_config.family,
        request_count=workload_config.request_count,
        block_size_tokens=config.block_size_tokens,
        seed=3,
    )
    request_types = [request.info.request_type for request in requests]
    warm_prompts = {
        (request.info.tenant_id, request.prompt_tokens)
        for request in requests
        if request.info.request_type == "tenant_cycle_warm"
    }
    recovery_prompts = {
        (request.info.tenant_id, request.prompt_tokens)
        for request in requests
        if request.info.request_type == "tenant_cycle_recovery"
    }

    assert workload_config.request_count == 288
    assert set(request_types) == {
        "tenant_cycle_warm",
        "tenant_cycle_pollution",
        "tenant_cycle_recovery",
    }
    assert recovery_prompts <= warm_prompts
    assert request_types.count("tenant_cycle_recovery") >= 60
    assert max(request.arrival_step or 0 for request in requests) > 200

    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/tenant_phase_shift_cycles"]
    assert metrics["recovery_phase_count"] == 6
    assert (
        metrics["worst_recovery_phase_token_hit_rate"]
        <= metrics["recovery_token_hit_rate"]
    )
    assert metrics["worst_recovery_phase_p95_latency_proxy"] > 0.0


def test_default_splits_include_production_shaped_workloads() -> None:
    config = EvaluatorConfig()

    assert {
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
        "priority_burst_recovery",
        "priority_one_off_noise",
        "tenant_phase_shift_cycles",
    }.issubset(config.validation_families)
    assert {
        "agent_trace_branching",
        "cyclic_working_set_pressure",
    } == set(config.probe_families)
    assert set(config.probe_families).isdisjoint(config.validation_families)
    assert {
        "stochastic_serving_mix_shifted",
        "rolling_template_versions_shifted",
        "heavy_tailed_prefix_lengths_shifted",
        "priority_burst_recovery_shifted",
        "cyclic_working_set_pressure_shifted",
        "priority_one_off_noise_shifted",
        "tenant_phase_shift_cycles_shifted",
    }.issubset(config.hidden_families)


def test_production_shaped_workloads_reward_selective_admission() -> None:
    families = (
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
    )
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=families,
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)

    for family in families:
        lru_metrics = lru.workload_metrics[f"validation/{family}"]
        tinylfu_metrics = tinylfu.workload_metrics[f"validation/{family}"]
        assert tinylfu_metrics["token_hit_rate"] > lru_metrics["token_hit_rate"]
        assert tinylfu_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_tenant_session_reentry_revisits_paused_context_with_new_tail() -> None:
    requests = build_workload(
        "tenant_session_reentry",
        request_count=40,
        block_size_tokens=8,
        seed=3,
    )

    first_visit = requests[0]
    resumed_session = requests[32]
    stable_prefix_tokens = 4 * 8
    assert first_visit.info.tenant_id == resumed_session.info.tenant_id
    assert first_visit.info.session_id == resumed_session.info.session_id
    assert (
        first_visit.prompt_tokens[:stable_prefix_tokens]
        == resumed_session.prompt_tokens[:stable_prefix_tokens]
    )
    assert first_visit.prompt_tokens != resumed_session.prompt_tokens


def test_tenant_session_reentry_rewards_selective_admission() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        hidden_families=("tenant_session_reentry",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("hidden",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)
    lru_metrics = lru.workload_metrics["hidden/tenant_session_reentry"]
    tinylfu_metrics = tinylfu.workload_metrics["hidden/tenant_session_reentry"]

    assert tinylfu_metrics["token_hit_rate"] > lru_metrics["token_hit_rate"]
    assert tinylfu_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_hotset_cold_scan_displaces_lru_and_rewards_scan_resistance() -> None:
    requests = build_workload(
        "hotset_cold_scan",
        request_count=24,
        block_size_tokens=8,
        seed=3,
    )
    assert requests[16].prompt_tokens == requests[0].prompt_tokens
    assert {request.info.request_type for request in requests[8:16]} == {"cold_scan"}

    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("hotset_cold_scan",),
    )
    evaluator = PrefixKVCacheEvaluator(config, splits=("validation",))
    lru = evaluator(baseline_lru_blocks)
    tinylfu = evaluator(baseline_tinylfu_lru)
    lru_metrics = lru.workload_metrics["validation/hotset_cold_scan"]
    tinylfu_metrics = tinylfu.workload_metrics["validation/hotset_cold_scan"]

    assert lru_metrics["reuse_after_eviction_missed_blocks"] > 0
    assert tinylfu_metrics["cache_churn_per_1k"] < lru_metrics["cache_churn_per_1k"]


def test_concurrent_long_generation_exercises_pinned_capacity_pressure() -> None:
    requests = build_workload(
        "concurrent_long_generation",
        request_count=24,
        block_size_tokens=8,
        seed=3,
    )
    assert all(request.info.predicted_output_length is not None for request in requests)
    assert min(request.true_output_length for request in requests) > 400
    assert [request.arrival_step for request in requests[:6]] == [0, 0, 1, 1, 2, 2]

    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=6,
        validation_families=("concurrent_long_generation",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/concurrent_long_generation"]

    assert metrics["forced_bypass_count"] > 0
    assert metrics["arrival_span_steps"] == 24
    assert metrics["active_request_count_peak"] > 2


def test_token_and_block_hit_rates_are_not_identical() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("agent_trace_branching",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/agent_trace_branching"]

    assert metrics["token_hit_rate"] != metrics["block_hit_rate"]


def test_structural_prefix_metrics_are_reported() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("agent_trace_branching",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(
        baseline_prefix_fanout
    )
    metrics = result.workload_metrics["validation/agent_trace_branching"]

    assert "depth_1_2_block_hit_rate" in metrics
    assert "depth_3_4_token_hit_rate" in metrics
    assert "depth_5_8_recompute_tokens_saved" in metrics
    assert "high_descendant_eviction_rate" in metrics
    assert "cold_deep_admission_rate" in metrics
    assert "reuse_after_eviction_missed_tokens" in metrics
    assert "system_prefix_hit_contribution" in metrics
    assert "developer_prefix_hit_contribution" in metrics
    assert "user_prefix_hit_contribution" in metrics
    assert "priority_weighted_token_hit_rate" in metrics
    assert "high_priority_p95_latency_proxy" in metrics
    assert "request_token_hit_rate_p10" in metrics
    assert "p95_recompute_cost" in metrics
    assert "policy_bypass_token_rate" in metrics
    assert "forced_bypass_token_rate" in metrics
    assert "useful_admission_rate" in metrics
    assert "wasted_admission_rate" in metrics
    assert "useful_admission_token_rate" in metrics
    assert "wasted_admission_token_rate" in metrics
    assert "admission_saved_tokens_per_admission" in metrics
    assert "admission_token_utility" in metrics
    assert "short_reuse_after_eviction_missed_token_rate" in metrics
    assert "avoidable_eviction_rate" in metrics
    assert "avoidable_short_reuse_eviction_rate" in metrics
    assert "worst_quarter_token_hit_rate" in metrics
    assert "final_quarter_token_hit_rate" in metrics
    assert "worst_recovery_phase_token_hit_rate" in metrics
    assert "tenant_token_hit_rate_p10" in metrics
    assert "token_hit_rate_worst_trial" in metrics
    assert "token_hit_rate_stddev_across_trials" in metrics
    assert metrics["depth_1_2_token_hit_rate"] > 0.0
    assert metrics["developer_prefix_hit_tokens"] > 0.0


def test_shared_system_prompt_reports_role_hit_contributions() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(baseline_lru_blocks)
    metrics = result.workload_metrics["train/shared_system_prompt"]

    assert metrics["system_prefix_hit_tokens"] > 0.0
    assert metrics["developer_prefix_hit_tokens"] > 0.0
    assert "user_prefix_hit_tokens" in metrics
    assert (
        metrics["system_prefix_hit_contribution"]
        + metrics["developer_prefix_hit_contribution"]
        + metrics["user_prefix_hit_contribution"]
    ) <= 1.0


def test_recompute_cost_varies_with_depth() -> None:
    class CapturePolicy(AdmitAllLRU):
        def __init__(self) -> None:
            self.costs: list[float] = []

        def score_admission(self, block, now: int) -> float:
            self.costs.append(block.estimated_recompute_cost)
            return 1.0

    policy = CapturePolicy()
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        train_families=("shared_system_prompt",),
    )
    PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: policy)

    assert len(set(policy.costs)) > 1
    assert policy.costs == sorted(policy.costs)


def test_admission_stays_prefix_contiguous() -> None:
    class RejectRootAdmitChildren(AdmitAllLRU):
        def score_admission(self, block, now: int) -> float:
            return -1.0 if block.depth == 1 else 1.0

    policy = RejectRootAdmitChildren()
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        capacity_blocks=8,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: policy)
    metrics = result.workload_metrics["train/shared_system_prompt"]

    assert metrics["admission_count"] == 0
    assert metrics["memory_occupancy_peak"] == 0


def test_rejected_admission_still_observes_rest_of_missed_chain() -> None:
    class RejectRootCaptureMisses(AdmitAllLRU):
        def __init__(self) -> None:
            self.missed_depths: list[int] = []

        def score_admission(self, block, now: int) -> float:
            return -1.0 if block.depth == 1 else 1.0

        def on_cache_miss(self, block, request, now: int) -> None:
            self.missed_depths.append(block.depth)

    policy = RejectRootCaptureMisses()
    config = EvaluatorConfig(
        request_count=1,
        seeds=(3,),
        capacity_blocks=8,
        train_families=("shared_system_prompt",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: policy)
    metrics = result.workload_metrics["train/shared_system_prompt"]
    expected_tokens = build_workload(
        "shared_system_prompt",
        request_count=1,
        block_size_tokens=config.block_size_tokens,
        seed=1003,
    )[0].info.prompt_length

    assert policy.missed_depths == [1, 2, 3, 4]
    assert metrics["admission_count"] == 0
    assert metrics["recompute_tokens"] == expected_tokens


def test_future_reuse_metadata_is_live_after_current_request() -> None:
    class CaptureFutureReuse(AdmitAllLRU):
        def __init__(self) -> None:
            self.observed: list[tuple[int, int, float | None, float | None]] = []

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observed.append(
                (
                    now,
                    block.depth,
                    block.estimated_future_reuse,
                    block.estimated_next_reuse_distance,
                )
            )

        def score_admission(self, block, now: int) -> float:
            return -1.0

    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        expose_future_reuse=True,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=8,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple(range(8)),
            ),
            true_output_length=8,
        )
        for request_id in range(2)
    )
    policy = CaptureFutureReuse()

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observed[:2] == [(0, 1, 1.0, 1.0), (0, 2, 1.0, 1.0)]
    assert policy.observed[2:] == [(1, 1, 0.0, float("inf")), (1, 2, 0.0, float("inf"))]


def test_future_reuse_metadata_preserves_same_step_next_use() -> None:
    class CaptureFutureReuse(AdmitAllLRU):
        def __init__(self) -> None:
            self.observed: list[tuple[int, int, float | None, float | None]] = []

        def on_cache_miss(self, block, request, now: int) -> None:
            self.observed.append(
                (
                    now,
                    block.depth,
                    block.estimated_future_reuse,
                    block.estimated_next_reuse_distance,
                )
            )

        def score_admission(self, block, now: int) -> float:
            return -1.0

    simulator = PrefixKVCacheSimulator(
        capacity_blocks=4,
        block_size_tokens=4,
        prefill_cost_per_token=1.0,
        lookup_cost_per_block=0.0,
        eviction_cost_per_block=0.0,
        expose_future_reuse=True,
    )
    requests = tuple(
        WorkloadRequest(
            info=RequestInfo(
                request_id=request_id,
                tenant_id=0,
                session_id=0,
                prompt_length=8,
                priority=0,
                request_type="unit",
                prompt_tokens=tuple(range(8)),
            ),
            true_output_length=8,
            arrival_step=0,
        )
        for request_id in range(2)
    )
    policy = CaptureFutureReuse()

    simulator.run(policy, requests, split="train", workload="unit", seed=1)

    assert policy.observed[:2] == [(0, 1, 1.0, 0.0), (0, 2, 1.0, 0.0)]
    assert policy.observed[2:] == [(0, 1, 0.0, float("inf")), (0, 2, 0.0, float("inf"))]


def test_prefix_anchor_is_distinct_from_prefix_fanout() -> None:
    block = PrefixBlockInfo(
        block_id=1,
        prefix_hash=1,
        parent_hash=None,
        depth=2,
        start_token=0,
        end_token=8,
        token_count=8,
        tenant_id=0,
        created_at=0,
        last_accessed_at=3,
        hit_count=0,
        descendant_count=5,
        active_ref_count=0,
        estimated_recompute_cost=8.0,
    )
    fanout = baseline_prefix_fanout(8, 4)
    anchor = baseline_prefix_anchor(8, 4)

    assert fanout.score_eviction(block, now=10) != anchor.score_eviction(block, now=10)


def test_write_baseline_plots_creates_svg_files(tmp_path) -> None:
    paths = write_baseline_plots(tmp_path, quick=True)

    assert {path.name for path in paths} == {
        "baseline_combined_scores.svg",
        "validation_token_hit_heatmap.svg",
        "token_vs_block_hit.svg",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "</svg>" in text


def test_save_run_artifacts_persists_best_program_and_metadata(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        prefix_runner,
        "_artifact_report_config",
        lambda: EvaluatorConfig(
            request_count=4,
            seeds=(3,),
            capacity_sweep_blocks=(8,),
        ),
    )
    best_program = textwrap.dedent(
        """
        class NoCachePolicy:
            def on_request_start(self, request, now):
                pass

            def score_admission(self, block, now):
                return -1.0

            def score_eviction(self, block, now):
                return 0.0

            def on_cache_hit(self, block, request, now):
                pass

            def on_cache_miss(self, block, request, now):
                pass


        def build_candidate(capacity_blocks, block_size_tokens, seed=None):
            return NoCachePolicy()
        """
    )
    result = SimpleNamespace(
        best_program=best_program,
        best_score=12.5,
        total_evaluations=7,
        total_cost=0.25,
        archive_size=3,
        runtime_seconds=4.0,
        metrics={"combined_score": 12.5},
        artifacts={"split_metrics": {"validation": {"token_hit_rate": 0.5}}},
        metadata={"levi_runtime_seconds": 4.0},
    )
    config_snapshot = tmp_path / "input-config.yaml"
    config_snapshot.write_text("max_iterations: 3\n", encoding="utf-8")

    run_dir = save_run_artifacts(
        result,
        tmp_path,
        iterations=3,
        config_label="unit-config",
        seed_label="artifacts/source-run",
        config_snapshot=config_snapshot,
        timestamp=datetime(2026, 6, 2, 1, 2, 3, tzinfo=UTC),
    )

    assert run_dir == tmp_path / "20260602T010203Z"
    assert "def build_candidate" in (run_dir / "best_program.py").read_text(
        encoding="utf-8"
    )
    assert '"combined_score": 12.5' in (run_dir / "metrics.json").read_text(
        encoding="utf-8"
    )
    assert '"config": "unit-config"' in (run_dir / "run_summary.json").read_text(
        encoding="utf-8"
    )
    assert '"seed_program": "artifacts/source-run"' in (
        run_dir / "run_summary.json"
    ).read_text(encoding="utf-8")
    assert '"config_snapshot": "config_snapshot.yaml"' in (
        run_dir / "run_summary.json"
    ).read_text(encoding="utf-8")
    assert (run_dir / "config_snapshot.yaml").read_text(
        encoding="utf-8"
    ) == config_snapshot.read_text(encoding="utf-8")
    assert (tmp_path / "latest_run.txt").read_text(encoding="utf-8") == str(run_dir)
    report = (run_dir / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "Prefix KV-Cache Best Program Baseline Comparison" in report
    assert "`candidate`" in report
    assert "`oracle_future_reuse`" in report
    assert "reporting-only/future-knowledge" in report
    assert "Held-Out Structure-Generalization Probe" in report
    assert "--baseline-report --capacity-sweep-blocks 8" in report
    assert "--config configs/prefix_kv_cache.yaml" in report
    assert "--baseline-report --quick" not in report


def _minimal_policy_source(admission_expr: str, eviction_expr: str) -> str:
    return textwrap.dedent(
        f"""
        class Policy:
            def on_request_start(self, request, now):
                pass

            def score_admission(self, block, now):
                return float({admission_expr})

            def score_eviction(self, block, now):
                return float({eviction_expr})

            def on_cache_hit(self, block, request, now):
                pass

            def on_cache_miss(self, block, request, now):
                pass

        def build_candidate(capacity_blocks, block_size_tokens, seed=None):
            return Policy()
        """
    )
