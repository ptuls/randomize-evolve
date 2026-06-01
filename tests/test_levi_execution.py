"""Regression tests for the Levi workflow adapter."""

import pickle
from pathlib import Path

from randomize_evolve.evaluator_entry import EvaluatorResult
from randomize_evolve.workflow.execution import (
    LeviRunner,
    LeviScoreFunction,
    _module_name_from_package_path,
)


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


def test_levi_runner_loads_package_evaluator_as_picklable_function() -> None:
    evaluator_path = (
        Path(__file__).resolve().parents[1]
        / "src/randomize_evolve/problems/set_membership/evaluator.py"
    )

    assert (
        _module_name_from_package_path(evaluator_path)
        == "randomize_evolve.problems.set_membership.evaluator"
    )

    runner = LeviRunner(
        evolve_code=lambda *_args, **_kwargs: None,
        evaluator_path=evaluator_path,
        problem_description="test",
        function_signature="def candidate_factory(key_bits: int, capacity: int):",
    )
    score_fn = LeviScoreFunction(runner._evaluate_factory)

    pickle.loads(pickle.dumps(score_fn))
