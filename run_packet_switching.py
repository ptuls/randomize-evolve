from pathlib import Path

from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import OpenEvolveRunner
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

try:  # Newer releases expose OpenEvolve at package root
    from openevolve import OpenEvolve
except ImportError:  # pragma: no cover - compatibility shim
    from openevolve.core import OpenEvolve  # type: ignore


_INITIAL_PROGRAM_PATH = Path(__file__).parent / "initial_program_packet_switching.py"
INITIAL_PROGRAM_SOURCE = ProgramSource(
    _INITIAL_PROGRAM_PATH.read_text(encoding="utf-8")
)

_EVALUATOR_PATH = Path(__file__).parent / "packet_switching_evaluator.py"
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
    iterations: int = 50,
    config_file: str = "configs/packet_switching_workload.yaml",
) -> None:
    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


if __name__ == "__main__":
    demo_run_evolution(
        iterations=50,
        config_file="configs/packet_switching_workload.yaml",
    )
