"""Tests for the packet-switching simulator and evaluator."""

import math
from random import Random
from typing import Dict, List, MutableMapping, Sequence

import packet_switching_evaluator
import pytest

from initial_program_packet_switching import (
    candidate_factory as packet_candidate_factory,
)
from packet_switching_seeds.exact_max_weight import ExactMaxWeightScheduler
from packet_switching_seeds.oldest_cell_first import OldestCellFirstScheduler
from packet_switching_seeds.pure_islip import ISLIPScheduler
from randomize_evolve.packet_switching import RoundRobinScheduler
from randomize_evolve.traffic import (
    SimulationResult,
    SwitchTrafficSimulator,
    TrafficPatternConfig,
    TrafficPatternType,
    build_pattern,
)
from randomize_evolve.evaluators.packet_switching import (
    PacketSwitchingEvaluator,
    PacketSwitchingEvaluatorConfig,
    ScenarioConfig,
    default_scenarios,
    matrix_stress_scenarios,
)


class GreedyScheduler:
    """Scheduler that greedily serves the lowest-numbered input."""

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        del time_slot, queue_lengths, voq_lengths
        matches: Dict[int, int] = {}
        used_outputs = set()
        for output_idx, inputs in requests.items():
            if output_idx in used_outputs:
                continue
            matches[min(inputs)] = output_idx
            used_outputs.add(output_idx)
        return matches


class LongestQueueScheduler:
    """Scheduler that prioritizes the longest visible VOQ per output."""

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        del time_slot, queue_lengths
        matches: Dict[int, int] = {}
        used_inputs = set()
        for output_idx, inputs in requests.items():
            ranked_inputs = sorted(
                inputs,
                key=lambda input_idx: (
                    -voq_lengths[input_idx][output_idx],
                    input_idx,
                ),
            )
            for input_idx in ranked_inputs:
                if input_idx in used_inputs:
                    continue
                matches[input_idx] = output_idx
                used_inputs.add(input_idx)
                break
        return matches


class OldestQueueScheduler:
    """Scheduler that prioritizes the oldest visible VOQ age per output."""

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
        voq_ages: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        del time_slot, queue_lengths, voq_lengths
        matches: Dict[int, int] = {}
        used_inputs = set()
        for output_idx, inputs in requests.items():
            ranked_inputs = sorted(
                inputs,
                key=lambda input_idx: (
                    -voq_ages[input_idx][output_idx],
                    input_idx,
                ),
            )
            for input_idx in ranked_inputs:
                if input_idx in used_inputs:
                    continue
                matches[input_idx] = output_idx
                used_inputs.add(input_idx)
                break
        return matches


class SingleBurstPattern:
    """Produces one deterministic burst of destinations on the first slot."""

    def __init__(self, arrivals: Sequence[Sequence[int]]):
        self._arrivals = [list(destinations) for destinations in arrivals]

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
        del rng, num_outputs
        if time_slot == 0:
            assert num_inputs == len(self._arrivals)
            return [list(destinations) for destinations in self._arrivals]
        return [[] for _ in range(num_inputs)]


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

    assert result.fairness_inputs < 0.8
    assert result.fairness_flows < 0.5
    assert result.throughput < 0.95


def test_simulator_uses_virtual_output_queues():
    simulator = SwitchTrafficSimulator(
        SingleBurstPattern([[0, 1], [0]]),
        num_inputs=2,
        num_outputs=2,
        time_slots=1,
        warmup_slots=0,
        seed=5,
    )

    result = simulator.run(RoundRobinScheduler(2, 2))

    assert result.total_generated == 3
    assert result.total_served == 2
    assert math.isclose(result.utilization, 1.0)
    assert math.isclose(result.average_total_queue, 3.0)


def test_scheduler_receives_voq_lengths():
    simulator = SwitchTrafficSimulator(
        SingleBurstPattern([[0, 0], [0]]),
        num_inputs=2,
        num_outputs=1,
        time_slots=1,
        warmup_slots=0,
        seed=7,
    )

    result = simulator.run(LongestQueueScheduler())

    assert result.total_generated == 3
    assert result.total_served == 1
    assert result.fairness_inputs == 0.5


def test_scheduler_receives_voq_ages():
    simulator = SwitchTrafficSimulator(
        SingleBurstPattern([[0], [0]]),
        num_inputs=2,
        num_outputs=1,
        time_slots=2,
        warmup_slots=0,
        seed=13,
    )

    result = simulator.run(OldestQueueScheduler())

    assert result.total_generated == 2
    assert result.total_served == 2
    assert result.fairness_inputs == 1.0


def test_exact_max_weight_scheduler_finds_best_matching():
    simulator = SwitchTrafficSimulator(
        SingleBurstPattern([[0, 1], [0]]),
        num_inputs=2,
        num_outputs=2,
        time_slots=1,
        warmup_slots=0,
        seed=23,
    )

    result = simulator.run(ExactMaxWeightScheduler(2, 2))

    assert result.total_served == 2
    assert math.isclose(result.utilization, 1.0)


def test_pure_islip_scheduler_runs_on_voq_simulator():
    simulator = SwitchTrafficSimulator(
        SingleBurstPattern([[0, 1], [0, 1]]),
        num_inputs=2,
        num_outputs=2,
        time_slots=5,
        warmup_slots=0,
        seed=29,
    )

    result = simulator.run(ISLIPScheduler(2, 2))

    assert result.total_generated > 0
    assert result.total_served > 0
    assert result.fairness_inputs > 0.0


def test_oldest_cell_first_scheduler_runs_with_age_snapshot():
    simulator = SwitchTrafficSimulator(
        SingleBurstPattern([[0], [0, 0]]),
        num_inputs=2,
        num_outputs=1,
        time_slots=2,
        warmup_slots=0,
        seed=31,
    )

    result = simulator.run(OldestCellFirstScheduler(2, 1))

    assert result.total_generated == 3
    assert result.total_served == 2
    assert result.average_total_queue >= 1.0


def test_simulator_repeated_runs_are_deterministic():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.BURSTY,
            offered_load=0.4,
            burst_rate=5,
            burst_length=6,
            burst_probability=0.12,
        )
    )
    simulator = SwitchTrafficSimulator(
        pattern,
        num_inputs=4,
        num_outputs=4,
        time_slots=250,
        warmup_slots=50,
        seed=11,
    )

    first = simulator.run(RoundRobinScheduler(4, 4))
    second = simulator.run(RoundRobinScheduler(4, 4))

    assert first == second


def test_arrival_matrix_pattern_targets_specified_queues():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
            arrival_matrix=((0.8, 0.0), (0.0, 0.8)),
        )
    )
    simulator = SwitchTrafficSimulator(
        pattern,
        num_inputs=2,
        num_outputs=2,
        time_slots=200,
        warmup_slots=0,
        seed=19,
    )

    result = simulator.run(RoundRobinScheduler(2, 2))

    assert result.total_generated > 0
    assert result.total_served > 0
    assert result.fairness_flows > 0.9


def test_hotspot_pattern_matches_configured_probability():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.HOTSPOT,
            offered_load=4.0,
            hotspot_probability=0.6,
            hotspot_output=0,
        )
    )
    rng = Random(1)
    total_packets = 0
    hotspot_packets = 0

    for time_slot in range(400):
        arrivals = pattern.sample(rng, time_slot, 4, 8)
        for outputs in arrivals:
            total_packets += len(outputs)
            hotspot_packets += sum(1 for output_idx in outputs if output_idx == 0)

    assert total_packets > 0
    assert hotspot_packets / total_packets == pytest.approx(0.6, abs=0.03)


def test_simulator_drop_rate_uses_offered_packets():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.UNIFORM,
            offered_load=5.0,
        )
    )
    simulator = SwitchTrafficSimulator(
        pattern,
        num_inputs=2,
        num_outputs=1,
        time_slots=200,
        warmup_slots=0,
        queue_limit=1,
        seed=3,
    )

    result = simulator.run(RoundRobinScheduler(2, 1))

    assert result.total_generated > 0
    assert result.total_dropped > 0
    assert 0.0 <= result.drop_rate <= 1.0
    assert math.isclose(
        result.drop_rate,
        result.total_dropped / result.total_generated,
    )


def test_simulator_rejects_negative_queue_limit():
    pattern = build_pattern(
        TrafficPatternConfig(
            pattern_type=TrafficPatternType.UNIFORM,
            offered_load=1.0,
        )
    )

    with pytest.raises(ValueError, match="queue_limit must be non-negative"):
        SwitchTrafficSimulator(
            pattern,
            num_inputs=1,
            num_outputs=1,
            time_slots=10,
            queue_limit=-1,
        )


def test_scenario_config_rejects_bounded_buffers():
    pattern = TrafficPatternConfig(
        pattern_type=TrafficPatternType.UNIFORM,
        offered_load=0.5,
    )

    with pytest.raises(ValueError, match="paper-matching evaluator"):
        ScenarioConfig(
            name="bounded-buffer",
            pattern=pattern,
            queue_limit=4,
        )


def test_packet_switching_evaluator_prioritizes_total_queue_size():
    evaluator = PacketSwitchingEvaluator(PacketSwitchingEvaluatorConfig(ports=2))
    scenario = ScenarioConfig(
        name="queue-weighted",
        pattern=TrafficPatternConfig(
            pattern_type=TrafficPatternType.UNIFORM,
            offered_load=0.5,
        ),
    )
    low_queue = SimulationResult(
        throughput=0.2,
        fairness_inputs=0.2,
        fairness_flows=0.2,
        utilization=0.2,
        drop_rate=0.0,
        average_total_queue=2.0,
        total_generated=10,
        total_served=2,
        total_dropped=0,
    )
    high_queue = SimulationResult(
        throughput=0.95,
        fairness_inputs=0.95,
        fairness_flows=0.95,
        utilization=0.95,
        drop_rate=0.0,
        average_total_queue=20.0,
        total_generated=10,
        total_served=9,
        total_dropped=0,
    )

    low_queue_score, _ = evaluator._score(low_queue, scenario)
    high_queue_score, _ = evaluator._score(high_queue, scenario)

    assert low_queue_score < high_queue_score


def test_packet_switching_normalizes_scenario_scores_against_baseline():
    assert PacketSwitchingEvaluator._normalize_scenario_score(5.0, 10.0) == pytest.approx(0.5)
    assert PacketSwitchingEvaluator._normalize_scenario_score(
        500.0,
        1000.0,
    ) == pytest.approx(0.5)


def test_round_robin_is_unit_baseline_under_normalized_evaluator():
    scenarios = default_scenarios()[:2]
    for scenario in scenarios:
        scenario.time_slots = 250
        scenario.warmup_slots = 50

    evaluator = PacketSwitchingEvaluator(
        PacketSwitchingEvaluatorConfig(
            ports=4,
            scenarios=scenarios,
            seed=12,
        )
    )

    result = evaluator(lambda ports: RoundRobinScheduler(ports, ports))

    assert result.score == pytest.approx(1.0)
    assert all(
        scenario_result.score == pytest.approx(1.0) for scenario_result in result.scenario_results
    )


def test_packet_switching_combined_reward_preserves_queue_differences():
    better_reward = packet_switching_evaluator._score_to_combined_reward(1134.9)
    worse_reward = packet_switching_evaluator._score_to_combined_reward(1140.2)

    assert better_reward > worse_reward
    assert better_reward - worse_reward > 1.0


def test_default_scenarios_include_matrix_stress_cases():
    scenarios = default_scenarios(ports=4)

    matrix_names = {
        scenario.name
        for scenario in scenarios
        if scenario.pattern.pattern_type == TrafficPatternType.ARRIVAL_MATRIX
    }

    assert matrix_names == {
        "cross-coupled",
        "hotspot-funnel",
        "asymmetric-boundary",
    }


def test_matrix_stress_scenarios_match_requested_port_count():
    scenarios = matrix_stress_scenarios(ports=4)

    for scenario in scenarios:
        matrix = scenario.pattern.arrival_matrix
        assert matrix is not None
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)


@pytest.mark.parametrize("ports", [4])
def test_packet_switching_evaluator_runs(ports: int):
    scenarios = default_scenarios()[:2]
    for scenario in scenarios:
        scenario.time_slots = 500
        scenario.warmup_slots = 100
    config = PacketSwitchingEvaluatorConfig(
        ports=ports,
        scenarios=scenarios,
        seed=12,
    )
    evaluator = PacketSwitchingEvaluator(config)
    result = evaluator(lambda p: RoundRobinScheduler(p, p))

    assert result.success
    assert len(result.scenario_results) == len(scenarios)
    assert result.score >= 0.0
    assert all(r.metrics.average_total_queue > 0.0 for r in result.scenario_results)


def test_voq_aware_seed_runs_on_matrix_suite():
    scenarios = [
        ScenarioConfig(
            name="diag-balanced",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
                arrival_matrix=(
                    (0.55, 0.05, 0.0, 0.0),
                    (0.05, 0.55, 0.0, 0.0),
                    (0.0, 0.0, 0.55, 0.05),
                    (0.0, 0.0, 0.05, 0.55),
                ),
            ),
            time_slots=900,
            warmup_slots=150,
        ),
        ScenarioConfig(
            name="cross-coupled",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
                arrival_matrix=(
                    (0.0, 0.62, 0.0, 0.08),
                    (0.62, 0.0, 0.08, 0.0),
                    (0.0, 0.08, 0.0, 0.62),
                    (0.08, 0.0, 0.62, 0.0),
                ),
            ),
            time_slots=900,
            warmup_slots=150,
        ),
    ]
    evaluator = PacketSwitchingEvaluator(
        PacketSwitchingEvaluatorConfig(
            ports=4,
            scenarios=scenarios,
            seed=17,
        )
    )

    round_robin_result = evaluator(lambda ports: RoundRobinScheduler(ports, ports))
    packet_seed_result = evaluator(packet_candidate_factory)

    assert packet_seed_result.success
    assert math.isfinite(packet_seed_result.score)
    assert packet_seed_result.score <= round_robin_result.score
