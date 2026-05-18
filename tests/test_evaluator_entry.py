"""Tests for shared evaluator entry-point helpers."""

import textwrap
import time

import pytest

from randomize_evolve.evaluator_entry import (
    EvaluationEntryPoint,
    extract_exported_callable,
    load_candidate_factory,
    run_with_timeout,
    score_to_reward,
)


def test_load_candidate_factory_accepts_build_candidate(tmp_path) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(
        "def build_candidate(value):\n    return value * 2\n",
        encoding="utf-8",
    )

    factory = load_candidate_factory(str(module_path))

    assert factory(3) == 6


def test_extract_exported_callable_raises_for_missing_names() -> None:
    class DummyModule:
        pass

    with pytest.raises(AttributeError, match="candidate_factory"):
        extract_exported_callable(DummyModule(), ("candidate_factory",))


def test_run_with_timeout_raises_timeout_error() -> None:
    def slow_operation() -> str:
        time.sleep(0.05)
        return "done"

    with pytest.raises(TimeoutError, match="wall-clock limit"):
        run_with_timeout(slow_operation, timeout_seconds=0.001)


def test_evaluation_entry_point_returns_adapted_success(tmp_path) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def candidate_factory():
                return "ok"
            """
        ),
        encoding="utf-8",
    )

    entry_point = EvaluationEntryPoint(
        evaluator_factory=lambda: lambda factory: {"value": factory()},
        timeout_seconds=1.0,
        load_error_suggestion="load hint",
        timeout_suggestion="timeout hint",
        success_result_builder=lambda result: {"status": "success", **result},  # type: ignore[arg-type]
        error_result_builder=lambda message, artifacts: {  # type: ignore[arg-type]
            "status": "error",
            "message": message,
            "artifacts": artifacts,
        },
    )

    result = entry_point.evaluate(str(module_path))

    assert result == {"status": "success", "value": "ok"}


def test_evaluation_entry_point_returns_structured_load_error() -> None:
    entry_point = EvaluationEntryPoint(
        evaluator_factory=lambda: lambda factory: factory,
        timeout_seconds=1.0,
        load_error_suggestion="expected load hint",
        timeout_suggestion="timeout hint",
        success_result_builder=lambda result: result,  # type: ignore[arg-type]
        error_result_builder=lambda message, artifacts: {  # type: ignore[arg-type]
            "message": message,
            "artifacts": artifacts,
        },
    )

    result = entry_point.evaluate("/definitely/missing/candidate.py")

    assert result["message"] == "failed to load candidate factory"
    assert result["artifacts"]["suggestion"] == "expected load hint"


def test_score_to_reward_handles_non_finite_values() -> None:
    assert score_to_reward(0.0) == 1.0
    assert score_to_reward(float("inf")) == 0.0
