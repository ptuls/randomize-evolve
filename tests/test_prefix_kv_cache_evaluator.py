"""Tests for the prefix KV-cache evaluator."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
import textwrap
import time

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
    _baseline_report_headline,
    _config_from_args,
    _evaluate_candidate_program,
    compare_baselines,
    save_run_artifacts,
    write_baseline_plots,
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


def test_discrete_baselines_break_equal_priority_ties_with_lru() -> None:
    older = _block_info(last_accessed_at=1)
    newer = _block_info(last_accessed_at=9)

    for factory in (
        baseline_lfu_blocks,
        baseline_depth_prefer_shallow,
        baseline_prefix_fanout,
    ):
        policy = factory(8, 4)
        assert policy.score_eviction(older, now=10) > policy.score_eviction(newer, now=10)


def test_lfu_still_prefers_to_evict_a_less_frequent_block() -> None:
    unused = _block_info(last_accessed_at=9, hit_count=0)
    frequent = _block_info(last_accessed_at=1, hit_count=1)
    policy = baseline_lfu_blocks(8, 4)

    assert policy.score_eviction(unused, now=10) > policy.score_eviction(frequent, now=10)


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
    assert oracle.score_eviction(later_often, now=0) > oracle.score_eviction(sooner_once, now=0)


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

    assert policy.score_eviction(served, now=10) > policy.score_eviction(underserved, now=10)


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

    lru_gap = lru.workload_metrics["validation/multi_tenant_skew"]["tenant_fairness_penalty"]
    tenant_fair_gap = tenant_fair.workload_metrics["validation/multi_tenant_skew"][
        "tenant_fairness_penalty"
    ]
    assert tenant_fair_gap < lru_gap


def test_prefix_fanout_beats_lru_on_branching() -> None:
    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=12,
        validation_families=("agent_trace_branching",),
    )
    lru = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    fanout = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_prefix_fanout)

    lru_hit_rate = lru.workload_metrics["validation/agent_trace_branching"]["token_hit_rate"]
    fanout_hit_rate = fanout.workload_metrics["validation/agent_trace_branching"]["token_hit_rate"]
    assert fanout_hit_rate > lru_hit_rate


def test_adversarial_over_admission_high_churn() -> None:
    config = EvaluatorConfig(
        request_count=36,
        seeds=(3,),
        capacity_blocks=8,
        hidden_families=("adversarial_unique_prompts",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("hidden",))(lambda *_: AdmitAllLRU())
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
        PrefixKVCacheEvaluator(config)(factory).combined_score for factory in BASELINES.values()
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

    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: MissingHooks())

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

    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: MemoryHeavyPolicy())

    assert result.invalid_fraction == 1.0
    assert (
        "candidate used" in result.workload_metrics["train/shared_system_prompt"]["invalid_reason"]
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
    result = PrefixKVCacheEvaluator(config, splits=("train",))(lambda *_: AdmitEverything())
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
    assert metrics.admission_count == 2
    assert metrics.eviction_count == 1


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
    assert all(not key.startswith("hidden/") for key in first.artifacts["workload_metrics"])


def test_baselines_separate_on_validation() -> None:
    config = EvaluatorConfig(request_count=48, seeds=(3,), capacity_blocks=12)
    scores = {
        name: PrefixKVCacheEvaluator(config, splits=("validation",))(factory).combined_score
        for name, factory in BASELINES.items()
    }

    assert len({round(score, 6) for score in scores.values()}) >= 5
    assert max(scores.values()) - min(scores.values()) > 80.0


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
    assert "candidate: combined_score=" in output
    assert "capacity_8:" in output
    assert "capacity_16:" in output
    assert "lru: combined_score=" in output
    assert "[deployable]" in output
    assert "future_reuse_heuristic: combined_score=" in output
    assert "oracle_future_reuse: combined_score=" in output
    assert "[oracle/reporting-only]" in output
    report = (tmp_path / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "Candidate `scoring_fn_complexity`" in report


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


def test_complexity_penalty_stays_linear_until_four_thousand_nodes() -> None:
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

    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=3_000) == -30.0
    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=4_000) == -40.0
    assert evaluator._score_trials(trials, invalid_fraction=0.0, complexity=5_000) == -40.0


def test_invalid_floor_is_below_largest_valid_total_deduction() -> None:
    config = EvaluatorConfig()
    largest_valid_total_deduction = (
        (1.0 + config.min_workload_weight) * config.latency_cap
        + config.churn_cap
        + config.fairness_cap
        + config.complex_cap
    )

    assert config.v_min < -largest_valid_total_deduction


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


def test_hidden_report_evaluates_requested_candidate(tmp_path, monkeypatch, capsys) -> None:
    candidate_path = tmp_path / "best_program.py"
    candidate_path.write_text("def build_candidate(): pass\n", encoding="utf-8")
    captured = {}

    def fake_evaluate_candidate(config, path, *, splits):
        captured["path"] = path
        captured["splits"] = splits
        return SimpleNamespace(combined_score=12.5)

    monkeypatch.setattr(prefix_runner, "_evaluate_candidate_program", fake_evaluate_candidate)
    monkeypatch.setattr(prefix_runner, "REPORTING_BASELINES", {})

    prefix_runner.hidden_report(quick=True, candidate_program=candidate_path)

    assert captured == {"path": candidate_path, "splits": ("hidden",)}
    assert f"candidate={candidate_path}" in capsys.readouterr().out


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
        resumed_session.prompt_tokens[: len(first_turn.prompt_tokens)] == first_turn.prompt_tokens
    )
    assert resumed_session.info.prompt_length == first_turn.info.prompt_length + 8


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

    config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_blocks=6,
        validation_families=("concurrent_long_generation",),
    )
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_lru_blocks)
    metrics = result.workload_metrics["validation/concurrent_long_generation"]

    assert metrics["forced_bypass_count"] > 0


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
    result = PrefixKVCacheEvaluator(config, splits=("validation",))(baseline_prefix_fanout)
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


def test_save_run_artifacts_persists_best_program_and_metadata(tmp_path) -> None:
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

    run_dir = save_run_artifacts(
        result,
        tmp_path,
        iterations=3,
        config_label="unit-config",
        timestamp=datetime(2026, 6, 2, 1, 2, 3, tzinfo=UTC),
    )

    assert run_dir == tmp_path / "20260602T010203Z"
    assert "def build_candidate" in (run_dir / "best_program.py").read_text(encoding="utf-8")
    assert '"combined_score": 12.5' in (run_dir / "metrics.json").read_text(encoding="utf-8")
    assert '"config": "unit-config"' in (run_dir / "run_summary.json").read_text(encoding="utf-8")
    assert (tmp_path / "latest_run.txt").read_text(encoding="utf-8") == str(run_dir)
    report = (run_dir / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "Prefix KV-Cache Best Program Baseline Comparison" in report
    assert "`candidate`" in report
    assert "`oracle_future_reuse`" in report
    assert "oracle/reporting-only" in report


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
