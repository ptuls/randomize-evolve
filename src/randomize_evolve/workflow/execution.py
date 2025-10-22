import asyncio
from pathlib import Path
from typing import Any, Callable

from openevolve import OpenEvolve


class OpenEvolveRunner:
    """Coordinates asynchronous execution of OpenEvolve."""

    def __init__(
        self, open_evolve_factory: Callable[..., OpenEvolve], evaluator_path: Path
    ) -> None:
        self._factory = open_evolve_factory
        self._evaluator_path = evaluator_path

    async def _run_async(self, program_path: Path, config) -> Any:
        orchestrator = self._factory(
            initial_program_path=str(program_path),
            evaluation_file=str(self._evaluator_path),
            config=config,
        )
        return await orchestrator.run()

    def run(self, program_path: Path, config) -> Any:
        return asyncio.run(self._run_async(program_path, config))
