"""Reporting helpers for evolution runs."""

from __future__ import annotations

from loguru import logger


class EvolutionReporter:
    """Formats evolution results for terminal output."""

    def report(self, result, iterations: int, config_label: str) -> None:
        logger.info("\n{}", "=" * 60)
        logger.info("EVOLUTION SUMMARY")
        logger.info("{}", "=" * 60)
        logger.info("Config: {}", config_label)
        logger.info("Iterations: {}", iterations)
        logger.info("Best score: {}", getattr(result, "best_score", "n/a"))
        snippet = getattr(result, "best_code", "")
        logger.info("\nBest program snippet:\n{}...\n", snippet[:200])
