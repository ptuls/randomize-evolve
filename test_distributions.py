from loguru import logger
import pytest

from evaluator import DEFAULT_CONFIG, Distribution, Evaluator, EvaluatorConfig
from initial_program import candidate_factory

pytestmark = pytest.mark.skip(reason="distribution comparison script is not a pytest test")


def test_distribution(name: str, config: EvaluatorConfig) -> None:
    """Test a single distribution."""
    delimiter = "=" * 60
    logger.info("\n{}", delimiter)
    logger.info("Testing: {}", name)
    logger.info("{}", delimiter)

    evaluator = Evaluator(config)
    result = evaluator(candidate_factory)

    logger.info("Success: {}", result.success)
    logger.info("False Positive Rate: {:.4%}", result.false_positive_rate)
    logger.info("False Negative Rate: {:.4%}", result.false_negative_rate)
    logger.info("Mean Memory: {:,.0f} bytes", result.mean_peak_memory_bytes)
    logger.info("Mean Build Time: {:.2f} ms", result.mean_build_time_ms)
    logger.info("Mean Query Time: {:.2f} ms", result.mean_query_time_ms)
    logger.info("Score: {:.2f}", result.score)


def main():
    """Run tests on all distribution types."""
    delimiter = "=" * 60
    logger.info("\n{}", delimiter)
    logger.info("DISTRIBUTION COMPARISON TEST")
    logger.info("Testing baseline program across different data patterns")
    logger.info("{}", delimiter)

    # Test 1: Uniform Random (default)
    uniform_config = DEFAULT_CONFIG.model_copy(update={"distribution": Distribution.UNIFORM})
    test_distribution("UNIFORM - Random across keyspace", uniform_config)

    # Test 2: Clustered
    clustered_config = DEFAULT_CONFIG.model_copy(
        update={
            "distribution": Distribution.CLUSTERED,
            "num_clusters": 10,
            "cluster_radius": 1000,
        }
    )
    test_distribution("CLUSTERED - 10 clusters, radius 1000", clustered_config)

    # Test 3: Sequential IDs
    sequential_config = DEFAULT_CONFIG.model_copy(update={"distribution": Distribution.SEQUENTIAL})
    test_distribution("SEQUENTIAL - Contiguous ID range", sequential_config)

    # Test 4: Power-Law (Zipf)
    power_law_config = DEFAULT_CONFIG.model_copy(
        update={
            "distribution": Distribution.POWER_LAW,
            "power_law_exponent": 1.5,
        }
    )
    test_distribution("POWER LAW - Zipf with exponent 1.5", power_law_config)

    # Test 5: Power-Law with higher skew
    power_law_high_config = DEFAULT_CONFIG.model_copy(
        update={
            "distribution": Distribution.POWER_LAW,
            "power_law_exponent": 2.5,
        }
    )
    test_distribution("POWER LAW - Zipf with exponent 2.5 (more skewed)", power_law_high_config)

    logger.info("\n{}", delimiter)
    logger.info("TEST COMPLETE")
    logger.info("{}", delimiter)


if __name__ == "__main__":
    main()
