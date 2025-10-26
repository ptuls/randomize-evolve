from loguru import logger
import pytest

from evaluator import DEFAULT_CONFIG, Distribution, Evaluator
from initial_program import candidate_factory


SCENARIOS = [
    (
        "UNIFORM - Random across keyspace",
        DEFAULT_CONFIG.model_copy(update={"distribution": Distribution.UNIFORM}),
    ),
    (
        "CLUSTERED - 10 clusters, radius 1000",
        DEFAULT_CONFIG.model_copy(
            update={
                "distribution": Distribution.CLUSTERED,
                "num_clusters": 10,
                "cluster_radius": 1000,
            }
        ),
    ),
    (
        "SEQUENTIAL - Contiguous ID range",
        DEFAULT_CONFIG.model_copy(update={"distribution": Distribution.SEQUENTIAL}),
    ),
    (
        "POWER LAW - Zipf with exponent 1.5",
        DEFAULT_CONFIG.model_copy(
            update={
                "distribution": Distribution.POWER_LAW,
                "power_law_exponent": 1.5,
            }
        ),
    ),
    (
        "POWER LAW - Zipf with exponent 2.5 (more skewed)",
        DEFAULT_CONFIG.model_copy(
            update={
                "distribution": Distribution.POWER_LAW,
                "power_law_exponent": 2.5,
            }
        ),
    ),
]


@pytest.mark.parametrize(("name", "config"), SCENARIOS)
def test_distribution(name, config):
    evaluator = Evaluator(config)
    result = evaluator(candidate_factory)

    logger.info(
        "Evaluated %s -> throughput score %.4f, fp_rate %.4f", name, result.score, result.false_positive_rate
    )
    assert result.success
