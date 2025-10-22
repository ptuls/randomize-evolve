from loguru import logger

from evaluator import DEFAULT_CONFIG, Distribution, Evaluator, EvaluatorConfig
from initial_program import candidate_factory


def test_distribution(name: str, config: EvaluatorConfig) -> None:
    """Test a single distribution."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    evaluator = Evaluator(config)
    result = evaluator(candidate_factory)

    print(f"Success: {result.success}")
    print(f"False Positive Rate: {result.false_positive_rate:.4%}")
    print(f"False Negative Rate: {result.false_negative_rate:.4%}")
    print(f"Mean Memory: {result.mean_peak_memory_bytes:,.0f} bytes")
    print(f"Mean Build Time: {result.mean_build_time_ms:.2f} ms")
    print(f"Mean Query Time: {result.mean_query_time_ms:.2f} ms")
    print(f"Score: {result.score:.2f}")


def main():
    """Run tests on all distribution types."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPARISON TEST")
    print("Testing baseline program across different data patterns")
    print("=" * 60)

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

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
