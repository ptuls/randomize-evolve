"""Demonstration helpers for direct function evolution."""

from typing import Any, Callable

from loguru import logger


def _score_bits(fn) -> dict[str, float]:
    selected = fn()
    return {"score": 1.0 / (1.0 + abs(10 - selected))}


class FunctionEvolutionScenario:
    """Encapsulates the direct-function evolution example."""

    def __init__(self, factory: Callable[[int, int], Any]) -> None:
        self._factory = factory

    def run(self, iterations: int) -> None:
        import levi

        result = levi.evolve_code(
            "Choose a compact bits-per-item value close to 10.",
            function_signature="def choose_bits_per_item() -> int:",
            seed_program="def choose_bits_per_item() -> int:\n    return 10\n",
            score_fn=_score_bits,
            model="openai/gpt-4o-mini",
            budget_evals=iterations,
        )
        logger.info("=== Function evolution summary ===")
        logger.info("iterations: {}", iterations)
        logger.info("best score: {}", getattr(result, "best_score", "n/a"))
        logger.info("best code:\n{}", getattr(result, "best_program", ""))
