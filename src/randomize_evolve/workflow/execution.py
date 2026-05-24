import asyncio
from pathlib import Path
from typing import Any, Callable

from openevolve import OpenEvolve

from randomize_evolve.workflow.cost_tracking import (
    build_summary_from_events_file,
    configure_tracked_model_clients,
    extract_run_cost_config,
    run_cost_tracking_environment,
    save_run_cost_summary,
)


class OpenEvolveRunner:
    """Coordinates asynchronous execution of OpenEvolve."""

    def __init__(
        self, open_evolve_factory: Callable[..., OpenEvolve], evaluator_path: Path
    ) -> None:
        self._factory = open_evolve_factory
        self._evaluator_path = evaluator_path

    async def _run_async(self, program_path: Path, config) -> Any:
        run_cost_config = extract_run_cost_config(config)
        configure_tracked_model_clients(config)

        orchestrator = self._factory(
            initial_program_path=str(program_path),
            evaluation_file=str(self._evaluator_path),
            config=config,
        )
        events_path = Path(orchestrator.output_dir) / "run_cost_events.jsonl"

        with run_cost_tracking_environment(
            events_path,
            prompt_cache_key_prefix=run_cost_config.get("prompt_cache_key_prefix"),
            prompt_cache_retention=run_cost_config.get("prompt_cache_retention"),
        ):
            result = await orchestrator.run()

        summary = build_summary_from_events_file(events_path, run_cost_config.get("model_pricing"))
        summary_dict = summary.to_dict()

        save_path = save_run_cost_summary(Path(orchestrator.output_dir), summary)
        if result is not None and hasattr(result, "metadata"):
            result.metadata["run_cost"] = summary_dict
            result.metadata["run_cost_summary_path"] = str(save_path)

        return result

    def run(self, program_path: Path, config) -> Any:
        return asyncio.run(self._run_async(program_path, config))
