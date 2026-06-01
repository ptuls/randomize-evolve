import pathlib
from pathlib import Path

from initial_program_heavy_hitters import candidate_factory
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import LeviRunner
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter


_INITIAL_PROGRAM_PATH = pathlib.Path(__file__).parent / "initial_program_heavy_hitters.py"
INITIAL_PROGRAM_SOURCE = ProgramSource(_INITIAL_PROGRAM_PATH.read_text(encoding="utf-8"))

_EVALUATOR_PATH = Path(__file__).parent / "heavy_hitters_evaluator.py"
_CONFIG_LOADER = ConfigLoader()


def _build_runner() -> LeviRunner:
    import levi

    return LeviRunner(
        levi.evolve_code,
        _EVALUATOR_PATH,
        problem_description=(
            "Search for compact streaming algorithms that accurately identify the "
            "heaviest keys in skewed data streams with limited memory."
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
    iterations: int = 150, config_file: str = "configs/heavy_hitters_workload.yaml"
) -> None:
    provider = YamlConfigProvider(pathlib.Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def demo_smoke_test(top_k: int = 5) -> None:
    sketch = candidate_factory(key_bits=18, capacity=64)

    heavy_items = list(range(5))
    for epoch in range(12):
        for value in heavy_items:
            sketch.observe(value, weight=5 - (value % 3))
        sketch.observe(10_000 + epoch, weight=1)

    print(f"Top-{top_k} heavy hitters: {sketch.top_k(top_k)}")
    for probe in [0, 1, 2, 10_001]:
        print(f"estimate({probe}) -> {sketch.estimate(probe)}")


if __name__ == "__main__":
    demo_run_evolution(iterations=50, config_file="configs/heavy_hitters_workload.yaml")
