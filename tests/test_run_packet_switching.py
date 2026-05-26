"""Tests for the packet-switching runner wiring."""

from pathlib import Path
from types import SimpleNamespace

import run_packet_switching
from run_packet_switching import (
    EVOLUTION_PROGRAM_SOURCE,
    INITIAL_PROGRAM_SOURCE,
    _allocate_portfolio_iterations,
    _load_seed_portfolio,
    _snapshot_run_cost_summary,
    build_arg_parser,
)


def test_initial_program_source_matches_repo_seed() -> None:
    seed_path = Path(__file__).resolve().parent.parent / "initial_program_packet_switching.py"

    assert INITIAL_PROGRAM_SOURCE.text() == seed_path.read_text(encoding="utf-8")


def test_evolution_program_source_matches_scaffold_seed() -> None:
    seed_path = (
        Path(__file__).resolve().parent.parent / "initial_program_packet_switching_evolution.py"
    )

    assert EVOLUTION_PROGRAM_SOURCE.text() == seed_path.read_text(encoding="utf-8")


def test_build_arg_parser_uses_packet_switching_defaults() -> None:
    args = build_arg_parser().parse_args([])

    assert args.iterations == 25
    assert args.config == "configs/packet_switching_workload.yaml"
    assert args.compare_only is False
    assert args.portfolio is False


def test_build_arg_parser_accepts_compare_flag() -> None:
    args = build_arg_parser().parse_args(["--compare-only", "--iterations", "7"])

    assert args.compare_only is True
    assert args.iterations == 7


def test_build_arg_parser_accepts_portfolio_flag() -> None:
    args = build_arg_parser().parse_args(["--portfolio", "--iterations", "11"])

    assert args.portfolio is True
    assert args.iterations == 11


def test_allocate_portfolio_iterations_spreads_remainder() -> None:
    assert _allocate_portfolio_iterations(10, 3) == (4, 3, 3)


def test_seed_portfolio_includes_diverse_scheduler_families() -> None:
    seed_names = [seed.name for seed in _load_seed_portfolio()]

    assert seed_names == [
        "voq_round_robin",
        "evolved_oldest_cell_first",
        "exact_max_weight",
    ]


def test_demo_run_evolution_uses_yaml_config(monkeypatch) -> None:
    observed = {}

    class FakeWorkflow:
        def execute(self, iterations: int):
            observed["iterations"] = iterations
            return SimpleNamespace(code="candidate", metrics={"combined_score": 1.0})

    def fake_build_workflow(provider):
        observed["provider"] = provider
        return FakeWorkflow()

    monkeypatch.setattr(
        run_packet_switching,
        "_build_workflow",
        fake_build_workflow,
    )

    run_packet_switching.demo_run_evolution(
        iterations=9,
        config_file="configs/packet_switching_workload.yaml",
    )

    assert observed["iterations"] == 9
    assert observed["provider"].path == Path("configs/packet_switching_workload.yaml")


def test_build_workflow_uses_abstract_evolution_seed(monkeypatch) -> None:
    observed = {}

    def fake_build_workflow_with_source(program_source, provider):
        observed["program_source"] = program_source
        observed["provider"] = provider
        return SimpleNamespace(execute=lambda iterations: None)

    monkeypatch.setattr(
        run_packet_switching,
        "_build_workflow_with_source",
        fake_build_workflow_with_source,
    )

    provider = object()
    run_packet_switching._build_workflow(provider)

    assert observed["program_source"].text() == EVOLUTION_PROGRAM_SOURCE.text()
    assert observed["provider"] is provider


def test_compare_baselines_runs_without_open_evolve() -> None:
    run_packet_switching.compare_baselines()


def test_demo_run_portfolio_selects_best_seed_then_exploits(monkeypatch) -> None:
    portfolio = (
        run_packet_switching.NamedProgramSource("seed_a", run_packet_switching.ProgramSource("a")),
        run_packet_switching.NamedProgramSource("seed_b", run_packet_switching.ProgramSource("b")),
        run_packet_switching.NamedProgramSource("seed_c", run_packet_switching.ProgramSource("c")),
    )
    observed_runs: list[tuple[str, int]] = []
    observed_exploit_sources: list[str] = []
    scores = {"a": 1.0, "b": 3.0, "c": 2.0}

    class FakeWorkflow:
        def __init__(self, source_text: str) -> None:
            self._source_text = source_text

        def execute(self, iterations: int):
            observed_runs.append((self._source_text, iterations))
            if self._source_text.startswith("winner:"):
                observed_exploit_sources.append(self._source_text)
                return SimpleNamespace(
                    code=self._source_text,
                    metrics={"combined_score": 9.0},
                )
            return SimpleNamespace(
                code=f"winner:{self._source_text}",
                metrics={"combined_score": scores[self._source_text]},
            )

    monkeypatch.setattr(run_packet_switching, "_load_seed_portfolio", lambda: portfolio)
    monkeypatch.setattr(
        run_packet_switching,
        "_build_workflow_with_source",
        lambda program_source, provider: FakeWorkflow(program_source.text()),
    )

    run_packet_switching.demo_run_portfolio(
        total_iterations=13,
        config_file="configs/packet_switching_workload.yaml",
    )

    assert observed_runs[:3] == [("a", 4), ("b", 4), ("c", 3)]
    assert observed_exploit_sources == ["winner:b"]


def test_snapshot_run_cost_summary_copies_artifact(tmp_path: Path) -> None:
    source = tmp_path / "run_cost_summary.json"
    source.write_text('{"estimated_cost_usd": 1.23}', encoding="utf-8")
    result = SimpleNamespace(metadata={"run_cost_summary_path": str(source)})

    copied = _snapshot_run_cost_summary(
        result,
        stage="explore",
        seed_name="seed_a",
        iterations=5,
    )

    assert copied == tmp_path / "portfolio_run_costs" / "explore_seed_a_5iters.json"
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert result.metadata["portfolio_run_cost_summary_path"] == str(copied)
