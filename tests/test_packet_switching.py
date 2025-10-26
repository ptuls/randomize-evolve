import math
from typing import Dict, List, MutableMapping, Sequence

import pytest

from randomize_evolve.packet_switching import RoundRobinScheduler
from randomize_evolve.traffic import (
    SwitchTrafficSimulator,
    TrafficPatternConfig,
    TrafficPatternType,
    build_pattern,
)
from randomize_evolve.evaluators.packet_switching import (
    PacketSwitchingEvaluator,
    PacketSwitchingEvaluatorConfig,
    default_scenarios,
)


class GreedyScheduler:
    """Scheduler that greedily serves the lowest-numbered input."""

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
    ) -> MutableMapping[int, int]:
        matches: Dict[int, int] = {}
        used_outputs = set()
        for output_idx, inputs in requests.items():
            if output_idx in used_outputs:
                continue
            matches[min(inputs)] = output_idx
            used_outputs.add(output_idx)
        return matches


def test_round_robin_handles_uniform_load():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.UNIFORM,
            offered_load=0.5,
        )
    )
    simulator = SwitchTrafficSimulator(
        pattern,
        num_inputs=4,
        num_outputs=4,
        time_slots=600,
        warmup_slots=100,
        seed=11,
    )
    result = simulator.run(RoundRobinScheduler(4, 4))

    assert result.total_generated > 0
    assert result.throughput > 0.9
    assert result.fairness_inputs > 0.95
    assert math.isclose(result.drop_rate, 0.0)


def test_simulator_flags_unfair_scheduler():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.HOTSPOT,
            offered_load=0.8,
            hotspot_probability=0.7,
        )
    )
    simulator = SwitchTrafficSimulator(
        pattern,
        num_inputs=4,
        num_outputs=4,
        time_slots=600,
        warmup_slots=100,
        seed=9,
    )
    result = simulator.run(GreedyScheduler())

    assert result.fairness_inputs < 0.7
    assert result.throughput < 0.8


@pytest.mark.parametrize("ports", [4])
def test_packet_switching_evaluator_runs(ports: int):
    scenarios = default_scenarios()[:2]
    for scenario in scenarios:
        scenario.time_slots = 500
        scenario.warmup_slots = 100
    config = PacketSwitchingEvaluatorConfig(ports=ports, scenarios=scenarios, seed=12)
    evaluator = PacketSwitchingEvaluator(config)
    result = evaluator(lambda p: RoundRobinScheduler(p, p))

    assert result.success
    assert len(result.scenario_results) == len(scenarios)
    assert result.score >= 0.0
    assert any(r.metrics.throughput < 0.95 for r in result.scenario_results)
