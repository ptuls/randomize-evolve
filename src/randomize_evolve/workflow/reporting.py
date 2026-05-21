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
        self._report_run_cost(result)
        snippet = getattr(result, "best_code", "")
        logger.info("\nBest program snippet:\n{}...\n", snippet[:200])

    def _report_run_cost(self, result) -> None:
        metadata = getattr(result, "metadata", None) or {}
        run_cost = metadata.get("run_cost")
        if not run_cost:
            return

        logger.info(
            (
                "LLM usage: {} requests, {} prompt tokens "
                "({} cached), {} completion tokens, {} total tokens"
            ),
            run_cost.get("requests", 0),
            run_cost.get("prompt_tokens", 0),
            run_cost.get("cached_prompt_tokens", 0),
            run_cost.get("completion_tokens", 0),
            run_cost.get("total_tokens", 0),
        )

        estimated_cost_usd = run_cost.get("estimated_cost_usd")
        if estimated_cost_usd is None:
            logger.info("Estimated LLM cost (USD): n/a")
        else:
            logger.info("Estimated LLM cost (USD): ${:.4f}", estimated_cost_usd)

        summary_path = metadata.get("run_cost_summary_path")
        if summary_path:
            logger.info("Run cost summary: {}", summary_path)
