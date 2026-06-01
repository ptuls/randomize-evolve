"""Regression tests for the Levi workflow adapter."""

from randomize_evolve.evaluator_entry import EvaluatorResult
from randomize_evolve.workflow.execution import LeviScoreFunction


def test_levi_score_function_exposes_combined_score() -> None:
    def evaluate_factory(factory):
        return EvaluatorResult(
            metrics={
                "combined_score": factory(),
                "ignored_inf": float("inf"),
                "ignored_text": "n/a",
            },
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: 2.5) == {"score": 2.5, "combined_score": 2.5}


def test_levi_score_function_clamps_invalid_score() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(metrics={"combined_score": float("nan")}, artifacts={})

    score_fn = LeviScoreFunction(evaluate_factory)

    assert score_fn(lambda: None) == {"score": 0.0}
