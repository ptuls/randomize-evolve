"""Demonstration helpers for direct function evolution."""

from typing import Any, Callable

from loguru import logger
from openevolve import evolve_function


class FunctionEvolutionScenario:
    """Encapsulates the direct-function evolution example."""

    def __init__(self, factory: Callable[[int, int], Any]) -> None:
        self._factory = factory

    def run(self, iterations: int) -> None:
        def wrapper(bits_per_item: int):
            bloom = self._factory(key_bits=32, capacity=5000)
            bloom.bits_per_item = bits_per_item
            return bloom.bits_per_item

        def score_fn(bits: int) -> int:
            return abs(10 - bits)

        test_cases = [(value, score_fn(value)) for value in (8, 10, 12)]

        result = evolve_function(wrapper, test_cases=test_cases, iterations=iterations)
        logger.info("=== Function evolution summary ===")
        logger.info("iterations: {}", iterations)
        logger.info("best score: {}", getattr(result, "best_score", "n/a"))
        logger.info("best code:\n{}", getattr(result, "best_code", ""))
