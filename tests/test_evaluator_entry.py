"""Tests for shared evaluator entry-point helpers."""

import multiprocessing
import textwrap
import time

import pytest

from randomize_evolve.evaluator_entry import (
    EvaluationEntryPoint,
    extract_exported_callable,
    load_candidate_factory,
    load_candidate_factory_from_source,
    run_with_timeout,
    score_to_reward,
)


def _run_timeout_helper_in_daemon(send_conn) -> None:
    try:
        send_conn.send(run_with_timeout(lambda: "done", timeout_seconds=0.1))
    finally:
        send_conn.close()


def test_load_candidate_factory_accepts_build_candidate(tmp_path) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(
        "def build_candidate(value):\n    return value * 2\n",
        encoding="utf-8",
    )

    factory = load_candidate_factory(str(module_path))

    assert factory(3) == 6


def test_load_candidate_factory_accepts_future_dataclass(tmp_path) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass


            @dataclass
            class Candidate:
                child: Candidate | None = None


            def build_candidate():
                return Candidate()
            """
        ),
        encoding="utf-8",
    )

    factory = load_candidate_factory(str(module_path))

    assert factory().__class__.__name__ == "Candidate"


def test_load_candidate_factory_from_source_accepts_future_dataclass() -> None:
    source = textwrap.dedent(
        """
        from __future__ import annotations

        from dataclasses import dataclass


        @dataclass
        class Candidate:
            child: Candidate | None = None


        def build_candidate():
            return Candidate()
        """
    )

    factory = load_candidate_factory_from_source(source)

    assert factory().__class__.__name__ == "Candidate"


def test_extract_exported_callable_raises_for_missing_names() -> None:
    class DummyModule:
        pass

    with pytest.raises(AttributeError, match="candidate_factory"):
        extract_exported_callable(DummyModule(), ("candidate_factory",))


def test_run_with_timeout_raises_timeout_error() -> None:
    def slow_operation() -> str:
        time.sleep(0.5)
        return "done"

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="wall-clock limit"):
        run_with_timeout(slow_operation, timeout_seconds=0.01)
    assert time.perf_counter() - started < 0.3


def test_run_with_timeout_executes_inline_inside_daemon_worker() -> None:
    context = multiprocessing.get_context("fork")
    receive_conn, send_conn = context.Pipe(duplex=False)
    process = context.Process(target=_run_timeout_helper_in_daemon, args=(send_conn,))
    process.daemon = True
    process.start()
    send_conn.close()
    try:
        assert receive_conn.poll(1.0)
        assert receive_conn.recv() == "done"
    finally:
        receive_conn.close()
        process.join(timeout=1.0)

    assert process.exitcode == 0


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
