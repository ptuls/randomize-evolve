import argparse
from pathlib import Path
from typing import Sequence

from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import OpenEvolveRunner
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

try:
    from openevolve import OpenEvolve
except ImportError:  # pragma: no cover - compatibility shim
    from openevolve.core import OpenEvolve  # type: ignore


def _load_initial_program_source() -> ProgramSource:
    seed_path = Path(__file__).with_name("initial_program_prompt_cache.py")
    return ProgramSource(seed_path.read_text(encoding="utf-8"))


INITIAL_PROGRAM_SOURCE = _load_initial_program_source()
_EVALUATOR_PATH = Path(__file__).parent / "prompt_cache_evaluator.py"
_CONFIG_LOADER = ConfigLoader()


def _build_runner() -> OpenEvolveRunner:
    return OpenEvolveRunner(OpenEvolve, _EVALUATOR_PATH)


def _build_workflow(provider) -> "EvolutionWorkflow":
    from randomize_evolve.workflow.workflow import EvolutionWorkflow

    return EvolutionWorkflow(
        program_source=INITIAL_PROGRAM_SOURCE,
        config_provider=provider,
        runner=_build_runner(),
        reporter=EvolutionReporter(),
    )


def demo_run_evolution_simple(iterations: int = 5) -> None:
    workflow = _build_workflow(MinimalConfigProvider())
    workflow.execute(iterations)


def demo_run_evolution(
    iterations: int = 30, config_file: str = "configs/prompt_cache_compaction.yaml"
) -> None:
    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of evolution iterations to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/prompt_cache_compaction.yaml",
        help="Path to the OpenEvolve YAML config file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    demo_run_evolution(iterations=args.iterations, config_file=args.config)


if __name__ == "__main__":
    main()
