"""Run cache eviction/admission evolution workflows."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import LeviRunner
from randomize_evolve.workflow.functions import FunctionEvolutionScenario
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

from .initial_program import candidate_factory


def _load_initial_program_source() -> ProgramSource:
    seed_path = Path(__file__).with_name("initial_program.py")
    return ProgramSource(seed_path.read_text(encoding="utf-8"))


INITIAL_PROGRAM_SOURCE = _load_initial_program_source()

_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_CONFIG_LOADER = ConfigLoader()


@dataclass(frozen=True)
class NamedProgramSource:
    """Associates a human-readable seed name with a seed program source."""

    name: str
    source: ProgramSource


def _build_runner() -> LeviRunner:
    import levi

    return LeviRunner(
        levi.evolve_code,
        _EVALUATOR_PATH,
        problem_description=(
            "Search for fixed-capacity cache policies under mixed Zipf, scan, and drift traces. "
            "The framework owns the cache map; candidates implement on_access(), "
            "should_admit(), and pick_victim(). Fitness trades hit rate against metadata "
            "bytes per cached item and per-access work."
        ),
        function_signature="def candidate_factory(key_bits: int, capacity: int):",
    )


def _build_workflow(provider) -> object:
    from randomize_evolve.workflow.workflow import EvolutionWorkflow

    runner = _build_runner()
    reporter = EvolutionReporter()
    return EvolutionWorkflow(
        program_source=INITIAL_PROGRAM_SOURCE,
        config_provider=provider,
        runner=runner,
        reporter=reporter,
    )


def demo_run_evolution_simple(iterations: int = 5) -> None:
    workflow = _build_workflow(MinimalConfigProvider())
    workflow.execute(iterations)


def demo_run_evolution(
    iterations: int = 25,
    config_file: str = "configs/cache_eviction_workload.yaml",
) -> None:
    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def demo_evolve_function(iterations: int = 10) -> None:
    FunctionEvolutionScenario(candidate_factory).run(iterations)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of evolution iterations to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/cache_eviction_workload.yaml",
        help="Path to the Levi YAML config file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logger.info("Starting cache eviction evolution with config {}", args.config)
    demo_run_evolution(iterations=args.iterations, config_file=args.config)


if __name__ == "__main__":
    main()
