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


def test_levi_score_function_prefers_source_aware_evaluator() -> None:
    source = "def build_candidate():\n    return 1\n"

    namespace = {"__source_code__": source}
    exec(source, namespace)

    def evaluate_factory(_factory):
        return EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={})

    def evaluate_source(candidate_source: str):
        assert candidate_source == source
        return EvaluatorResult(
            metrics={"combined_score": 2.0, "scoring_fn_complexity": 7},
            artifacts={},
        )

    score_fn = LeviScoreFunction(evaluate_factory, evaluate_source)

    assert score_fn(namespace["build_candidate"]) == {
        "score": 2.0,
        "combined_score": 2.0,
        "scoring_fn_complexity": 7.0,
    }


def test_levi_score_function_fails_closed_when_source_is_required() -> None:
    def evaluate_factory(_factory):
        return EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={})

    def evaluate_source(_candidate_source: str):
        return EvaluatorResult(metrics={"combined_score": 2.0}, artifacts={})

    score_fn = LeviScoreFunction(evaluate_factory, evaluate_source)

    assert score_fn(lambda: None) == {
        "score": 0.0,
        "combined_score": 0.0,
        "complexity_source_missing": 1.0,
    }


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


def test_levi_runner_loads_file_evaluator_with_future_dataclass(tmp_path) -> None:
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text(
        """
from __future__ import annotations

from dataclasses import dataclass

from randomize_evolve.evaluator_entry import EvaluatorResult


@dataclass
class Payload:
    child: Payload | None = None


def evaluate_factory(factory):
    return EvaluatorResult(metrics={"combined_score": 1.0}, artifacts={"payload": Payload()})


def evaluate_source(source):
    return EvaluatorResult(
        metrics={"combined_score": 2.0, "source_length": len(source)},
        artifacts={"payload": Payload()},
    )
""",
        encoding="utf-8",
    )

    runner = LeviRunner(
        evolve_code=lambda *_args, **_kwargs: None,
        evaluator_path=evaluator_path,
        problem_description="test",
        function_signature="def candidate_factory():",
    )

    assert runner._evaluate_factory(lambda: None).metrics["combined_score"] == 1.0
    assert (
        runner._evaluate_best_program(
            "def build_candidate():\n    return None\n"
        ).metrics["combined_score"]
        == 2.0
    )
