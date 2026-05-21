"""Tests for LLM usage and run cost accounting."""

from randomize_evolve.workflow.cost_tracking import RunCostTracker


def test_run_cost_tracker_aggregates_usage_and_cost() -> None:
    tracker = RunCostTracker(
        {
            "gpt-5.4-mini": {
                "input_per_1m_tokens": 0.75,
                "cached_input_per_1m_tokens": 0.075,
                "output_per_1m_tokens": 4.5,
            }
        }
    )

    tracker.record_usage(
        "gpt-5.4-mini",
        prompt_tokens=1_000_000,
        completion_tokens=100_000,
        cached_prompt_tokens=200_000,
        reasoning_tokens=50_000,
    )
    tracker.record_usage(
        "gpt-5.4-mini",
        prompt_tokens=500_000,
        completion_tokens=50_000,
        cached_prompt_tokens=0,
        reasoning_tokens=20_000,
    )

    summary = tracker.build_summary()

    assert summary.requests == 2
    assert summary.prompt_tokens == 1_500_000
    assert summary.completion_tokens == 150_000
    assert summary.total_tokens == 1_650_000
    assert summary.cached_prompt_tokens == 200_000
    assert summary.reasoning_tokens == 70_000
    assert summary.estimated_cost_usd == 1.665

    model_usage = summary.per_model["gpt-5.4-mini"]
    assert model_usage.estimated_cost_usd == 1.665


def test_run_cost_tracker_returns_no_estimate_without_pricing() -> None:
    tracker = RunCostTracker()
    tracker.record_usage("unknown-model", prompt_tokens=10, completion_tokens=5)

    summary = tracker.build_summary()

    assert summary.requests == 1
    assert summary.estimated_cost_usd is None
    assert summary.per_model["unknown-model"].estimated_cost_usd is None
