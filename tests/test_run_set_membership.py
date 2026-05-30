"""Tests for the set-membership runner wiring."""

from pathlib import Path
from types import SimpleNamespace

import run_set_membership
from run_set_membership import (
    INITIAL_PROGRAM_SOURCE,
    SKELETAL_DISTRIBUTION_PROGRAM_SOURCE,
    _allocate_portfolio_iterations,
    _load_curriculum_seed_portfolio,
    _read_workload_distribution,
    _select_initial_program_source,
    _uses_distribution_specific_seed,
    build_arg_parser,
)
from set_membership_seeds.skeletal_distribution import (
    candidate_factory as skeletal_candidate_factory,
)


def test_initial_program_source_matches_repo_seed() -> None:
    seed_path = Path(__file__).resolve().parent.parent / "initial_program.py"

    assert INITIAL_PROGRAM_SOURCE.text() == seed_path.read_text(encoding="utf-8")


def test_skeletal_distribution_source_matches_repo_seed() -> None:
    seed_path = (
        Path(__file__).resolve().parent.parent
        / "set_membership_seeds"
        / "skeletal_distribution.py"
    )

    assert SKELETAL_DISTRIBUTION_PROGRAM_SOURCE.text() == seed_path.read_text(
        encoding="utf-8"
    )


def test_build_arg_parser_uses_uniform_defaults() -> None:
    args = build_arg_parser().parse_args([])

    assert args.iterations == 25
    assert args.config == "configs/uniform_workload.yaml"
    assert args.curriculum is False
    assert args.explore_iterations == 10
    assert args.exploit_iterations == 15


def test_build_arg_parser_accepts_curriculum_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--curriculum",
            "--explore-iterations",
            "7",
            "--exploit-iterations",
            "9",
        ]
    )

    assert args.curriculum is True
    assert args.explore_iterations == 7
    assert args.exploit_iterations == 9


def test_allocate_portfolio_iterations_spreads_remainder() -> None:
    assert _allocate_portfolio_iterations(8, 3) == (3, 3, 2)


def test_curriculum_seed_portfolio_includes_non_bloom_families() -> None:
    seed_names = [seed.name for seed in _load_curriculum_seed_portfolio()]

    assert seed_names == ["bloom", "cuckoo_filter", "fingerprint_probe"]


def test_distribution_specific_workloads_use_skeletal_seed() -> None:
    assert _read_workload_distribution("configs/uniform_workload.yaml") is None
    assert _read_workload_distribution("configs/clustered_workload.yaml") == "clustered"
    assert _read_workload_distribution("configs/power_law_workload.yaml") == "power_law"
    assert _read_workload_distribution("configs/minimal_hints_workload.yaml") == "power_law"

    assert not _uses_distribution_specific_seed("configs/uniform_workload.yaml")
    assert _uses_distribution_specific_seed("configs/clustered_workload.yaml")
    assert _uses_distribution_specific_seed("configs/power_law_workload.yaml")
    assert _uses_distribution_specific_seed("configs/minimal_hints_workload.yaml")

    assert (
        _select_initial_program_source("configs/uniform_workload.yaml")
        is INITIAL_PROGRAM_SOURCE
    )
    assert (
        _select_initial_program_source("configs/clustered_workload.yaml")
        is SKELETAL_DISTRIBUTION_PROGRAM_SOURCE
    )


def test_skeletal_distribution_seed_preserves_membership_contract() -> None:
    candidate = skeletal_candidate_factory(key_bits=32, capacity=100)
    positives = list(range(75))

    for value in positives:
        candidate.add(value)

    assert all(candidate.query(value) for value in positives)


def test_curriculum_exploration_selects_best_portfolio_result(
    monkeypatch,
) -> None:
    portfolio = (
        run_set_membership.NamedProgramSource("bloom", run_set_membership.ProgramSource("a")),
        run_set_membership.NamedProgramSource(
            "cuckoo_filter", run_set_membership.ProgramSource("b")
        ),
        run_set_membership.NamedProgramSource(
            "fingerprint_probe", run_set_membership.ProgramSource("c")
        ),
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

    monkeypatch.setattr(run_set_membership, "_load_curriculum_seed_portfolio", lambda: portfolio)
    monkeypatch.setattr(
        run_set_membership,
        "_build_workflow_with_source",
        lambda program_source, provider: FakeWorkflow(program_source.text()),
    )

    run_set_membership.demo_run_curriculum(
        explore_iterations=5,
        exploit_iterations=4,
    )

    assert observed_runs[:3] == [("a", 2), ("b", 2), ("c", 1)]
    assert observed_exploit_sources == ["winner:b"]
