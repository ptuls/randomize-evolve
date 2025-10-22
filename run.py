from pathlib import Path

from initial_program import candidate_factory
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import OpenEvolveRunner
from randomize_evolve.workflow.functions import FunctionEvolutionScenario
from randomize_evolve.workflow.program import ProgramSource, TemporaryProgramFile
from randomize_evolve.workflow.reporting import EvolutionReporter

try:  # Newer releases expose OpenEvolve at package root
    from openevolve import OpenEvolve
except ImportError:  # pragma: no cover - compatibility shim
    from openevolve.core import OpenEvolve  # type: ignore


INITIAL_PROGRAM_SOURCE = ProgramSource(
    """
    import hashlib

    # EVOLVE-BLOCK-START
    class Candidate:
        def __init__(self, key_bits, capacity, bits_per_item=10):
            self.key_bits = key_bits
            self.capacity = capacity
            self.bits_per_item = bits_per_item
            self.fingerprint_bits = 16
            self.table = {}

        def add(self, item):
            h = hashlib.blake2b(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
            key = int.from_bytes(h[:4], 'little') % self.capacity
            fingerprint = int.from_bytes(h[4:6], 'little')
            self.table[key] = fingerprint

        def query(self, item):
            h = hashlib.blake2b(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
            key = int.from_bytes(h[:4], 'little') % self.capacity
            fingerprint = int.from_bytes(h[4:6], 'little')
            return self.table.get(key) == fingerprint
    # EVOLVE-BLOCK-END


    def candidate_factory(key_bits, capacity):
        return Candidate(key_bits, capacity)
    """
)

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


if __name__ == "__main__":
    demo_run_evolution(iterations=50, config_file="configs/minimal_hints_workload.yaml")
