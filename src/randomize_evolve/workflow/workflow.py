from typing import Any

from randomize_evolve.workflow.program import ProgramSource, TemporaryProgramFile


class EvolutionWorkflow:
    """High-level façade that ties together the supporting services."""

    def __init__(
        self,
        program_source: ProgramSource,
        config_provider,
        runner,
        reporter,
    ) -> None:
        self._program_source = program_source
        self._config_provider = config_provider
        self._runner = runner
        self._reporter = reporter

    def execute(self, iterations: int) -> Any:
        config = self._config_provider.load(iterations)
        with TemporaryProgramFile(self._program_source) as program_path:
            result = self._runner.run(program_path, config)
        self._reporter.report(result, iterations, self._config_provider.describe())
        return result
