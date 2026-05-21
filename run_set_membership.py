import argparse
from pathlib import Path
from typing import Sequence

from initial_program import candidate_factory
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import OpenEvolveRunner
from randomize_evolve.workflow.functions import FunctionEvolutionScenario
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

try:  # Newer releases expose OpenEvolve at package root
    from openevolve import OpenEvolve
except ImportError:  # pragma: no cover - compatibility shim
    from openevolve.core import OpenEvolve  # type: ignore


def _load_initial_program_source() -> ProgramSource:
    """Load the set-membership seed program from the repo baseline file."""
    seed_path = Path(__file__).with_name("initial_program.py")
    return ProgramSource(seed_path.read_text(encoding="utf-8"))


INITIAL_PROGRAM_SOURCE = _load_initial_program_source()

_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_CONFIG_LOADER = ConfigLoader()


def _build_runner() -> OpenEvolveRunner:
    return OpenEvolveRunner(OpenEvolve, _EVALUATOR_PATH)


def _build_workflow(provider) -> "EvolutionWorkflow":
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
    iterations: int = 25, config_file: str = "configs/uniform_workload.yaml"
) -> None:
    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def demo_evolve_function(iterations: int = 10) -> None:
    FunctionEvolutionScenario(candidate_factory).run(iterations)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for set-membership evolution runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of evolution iterations to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/uniform_workload.yaml",
        help="Path to the OpenEvolve YAML config file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the set-membership workflow from a real file-backed CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    demo_run_evolution(iterations=args.iterations, config_file=args.config)


if __name__ == "__main__":
    main()
