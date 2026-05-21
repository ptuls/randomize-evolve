"""Tests for the set-membership runner wiring."""

from pathlib import Path

from run_set_membership import INITIAL_PROGRAM_SOURCE, build_arg_parser


def test_initial_program_source_matches_repo_seed() -> None:
    seed_path = Path(__file__).resolve().parent.parent / "initial_program.py"

    assert INITIAL_PROGRAM_SOURCE.text() == seed_path.read_text(encoding="utf-8")


def test_build_arg_parser_uses_uniform_defaults() -> None:
    args = build_arg_parser().parse_args([])

    assert args.iterations == 25
    assert args.config == "configs/uniform_workload.yaml"
