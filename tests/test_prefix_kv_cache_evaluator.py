"""Tests for the prefix KV-cache evaluator."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
import textwrap

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
    baseline_lru_blocks,
    baseline_no_cache,
    baseline_prefix_anchor,
    baseline_prefix_fanout,
    build_workload,
)
from randomize_evolve.problems.prefix_kv_cache import evaluator as levi_evaluator
from randomize_evolve.problems.prefix_kv_cache.runner import (
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


def test_prefix_fanout_beats_lru_on_branching() -> None:
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
    assert fanout_hit_rate > lru_hit_rate


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


def test_baselines_separate_on_validation() -> None:
    config = EvaluatorConfig(request_count=48, seeds=(3,), capacity_blocks=12)
    scores = {
        name: PrefixKVCacheEvaluator(config, splits=("validation",))(
            factory
        ).combined_score
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
    assert "def build_candidate" in (run_dir / "best_program.py").read_text(
        encoding="utf-8"
    )
    assert '"combined_score": 12.5' in (run_dir / "metrics.json").read_text(
        encoding="utf-8"
    )
    assert '"config": "unit-config"' in (run_dir / "run_summary.json").read_text(
        encoding="utf-8"
    )
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
