"""Evaluator for packet switching scheduling strategies."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from randomize_evolve.packet_switching import RoundRobinScheduler, SwitchScheduler
from randomize_evolve.traffic import (
    SimulationResult,
    SwitchTrafficSimulator,
    TrafficPatternConfig,
    TrafficPatternType,
    build_pattern,
)


SchedulerFactory = Callable[[int], SwitchScheduler]


@dataclass
class ScenarioConfig:
    """Configures a single traffic scenario for evaluation."""

    name: str
    pattern: TrafficPatternConfig
    time_slots: int = 1500
    warmup_slots: int = 200
    queue_limit: Optional[int] = None
    queue_size_weight: float = 1.0
    throughput_weight: float = 0.05
    fairness_weight: float = 0.02
    flow_fairness_weight: float = 0.01
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.queue_limit is not None:
            raise ValueError(
                "ScenarioConfig.queue_limit is not supported in the "
                "paper-matching evaluator; use SwitchTrafficSimulator "
                "directly for bounded-buffer experiments."
            )


@dataclass
class ScenarioResult:
    """Stores the outcome of running a scheduler in a scenario."""

    config: ScenarioConfig
    metrics: SimulationResult
    score: float


@dataclass
class PacketSwitchingEvaluation:
    """Aggregated evaluation result across scenarios."""

    score: float
    scenario_results: List[ScenarioResult]
    success: bool


@dataclass
class PacketSwitchingEvaluatorConfig:
    """High level configuration for the packet switching evaluator."""

    ports: int = 4
    scenarios: Sequence[ScenarioConfig] = field(default_factory=list)
    seed: int = 7

    def __post_init__(self) -> None:
        if not self.scenarios:
            self.scenarios = default_scenarios(self.ports)


def _cross_coupled_matrix(ports: int) -> tuple[tuple[float, ...], ...]:
    matrix = [[0.0 for _ in range(ports)] for _ in range(ports)]
    for input_idx in range(ports):
        matrix[input_idx][(input_idx + 1) % ports] = 0.62
        matrix[input_idx][(input_idx - 1) % ports] = 0.08
    return tuple(tuple(row) for row in matrix)


def _hotspot_funnel_matrix(ports: int) -> tuple[tuple[float, ...], ...]:
    matrix = [[0.0 for _ in range(ports)] for _ in range(ports)]
    hotspot_rate = 0.52 / ports
    diagonal_rate = 0.42
    for input_idx in range(ports):
        matrix[input_idx][0] = hotspot_rate
        matrix[input_idx][input_idx] += diagonal_rate
    return tuple(tuple(row) for row in matrix)


def _asymmetric_boundary_matrix(ports: int) -> tuple[tuple[float, ...], ...]:
    row_templates = (
        (0.65, 0.18, 0.0, 0.0),
        (0.12, 0.35, 0.2, 0.0),
        (0.0, 0.18, 0.5, 0.12),
        (0.0, 0.0, 0.12, 0.58),
    )
    matrix = [[0.0 for _ in range(ports)] for _ in range(ports)]
    for input_idx in range(ports):
        template = row_templates[input_idx % len(row_templates)]
        for offset, rate in enumerate(template):
            output_idx = (input_idx + offset) % ports
            matrix[input_idx][output_idx] += rate
    return tuple(tuple(row) for row in matrix)


def matrix_stress_scenarios(ports: int = 4) -> List[ScenarioConfig]:
    """Returns matrix-driven scenarios that punish weak round-robin policies."""

    return [
        ScenarioConfig(
            name="cross-coupled",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
                arrival_matrix=_cross_coupled_matrix(ports),
            ),
            time_slots=1500,
            warmup_slots=200,
            queue_size_weight=1.0,
            throughput_weight=0.05,
            fairness_weight=0.02,
            flow_fairness_weight=0.01,
        ),
        ScenarioConfig(
            name="hotspot-funnel",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
                arrival_matrix=_hotspot_funnel_matrix(ports),
            ),
            time_slots=1800,
            warmup_slots=300,
            queue_size_weight=1.0,
            throughput_weight=0.05,
            fairness_weight=0.02,
            flow_fairness_weight=0.01,
        ),
        ScenarioConfig(
            name="asymmetric-boundary",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
                arrival_matrix=_asymmetric_boundary_matrix(ports),
            ),
            time_slots=1800,
            warmup_slots=300,
            queue_size_weight=1.0,
            throughput_weight=0.05,
            fairness_weight=0.02,
            flow_fairness_weight=0.01,
        ),
    ]


def default_scenarios(ports: int = 4) -> List[ScenarioConfig]:
    """Returns a curated set of default traffic scenarios."""

    return matrix_stress_scenarios(ports) + [
        ScenarioConfig(
            name="bursty",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.BURSTY,
                offered_load=0.4,
                burst_rate=5,
                burst_length=6,
                burst_probability=0.12,
            ),
            queue_size_weight=1.0,
            throughput_weight=0.05,
            fairness_weight=0.02,
            flow_fairness_weight=0.01,
        ),
        ScenarioConfig(
            name="hotspot-heavy",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.HOTSPOT,
                offered_load=0.75,
                hotspot_probability=0.65,
            ),
            queue_size_weight=1.0,
            throughput_weight=0.05,
            fairness_weight=0.03,
            flow_fairness_weight=0.02,
        ),
        ScenarioConfig(
            name="heavy-cycle",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.HEAVY_LOAD,
                heavy_load=0.98,
                light_load=0.35,
                heavy_duration=80,
                light_duration=40,
            ),
            queue_size_weight=1.0,
            throughput_weight=0.05,
            fairness_weight=0.02,
            flow_fairness_weight=0.01,
        ),
    ]


class PacketSwitchingEvaluator:
    """Callable evaluator compatible with the OpenEvolve workflow."""

    def __init__(self, config: Optional[PacketSwitchingEvaluatorConfig] = None) -> None:
        self.config = config or PacketSwitchingEvaluatorConfig()

    def __call__(
        self,
        factory: Optional[SchedulerFactory] = None,
    ) -> PacketSwitchingEvaluation:
        """Evaluates a scheduler factory across the configured scenarios.

        Args:
            factory: Optional scheduler factory. When omitted, uses the built-in
                round-robin scheduler.

        Returns:
            Aggregated evaluation results across all scenarios.
        """
        factory = factory or (lambda ports: RoundRobinScheduler(ports, ports))
        scenario_results: List[ScenarioResult] = []
        total_weight = 0.0
        total_score = 0.0

        for index, scenario in enumerate(self.config.scenarios):
            scheduler = factory(self.config.ports)
            pattern = build_pattern(scenario.pattern)
            simulator = SwitchTrafficSimulator(
                pattern,
                num_inputs=self.config.ports,
                num_outputs=self.config.ports,
                time_slots=scenario.time_slots,
                warmup_slots=scenario.warmup_slots,
                seed=self.config.seed + scenario.seed_offset + index,
            )
            metrics = simulator.run(scheduler)
            scenario_score, scenario_weight = self._score(metrics, scenario)
            total_score += scenario_score
            total_weight += scenario_weight
            scenario_results.append(
                ScenarioResult(
                    config=scenario,
                    metrics=metrics,
                    score=scenario_score / scenario_weight if scenario_weight else 0.0,
                )
            )

        aggregate_score = total_score / total_weight if total_weight else float("inf")
        success = bool(scenario_results)
        return PacketSwitchingEvaluation(
            score=aggregate_score,
            scenario_results=scenario_results,
            success=success,
        )

    def _score(
        self,
        metrics: SimulationResult,
        scenario: ScenarioConfig,
    ) -> tuple[float, float]:
        weights = (
            scenario.queue_size_weight,
            scenario.throughput_weight,
            scenario.fairness_weight,
            scenario.flow_fairness_weight,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Scenario weight configuration must be positive")
        queue_term = scenario.queue_size_weight * metrics.average_total_queue
        throughput_term = scenario.throughput_weight * (1.0 - metrics.throughput)
        fairness_term = scenario.fairness_weight * (1.0 - metrics.fairness_inputs)
        flow_fairness_term = scenario.flow_fairness_weight * (
            1.0 - metrics.fairness_flows
        )
        scenario_score = (
            queue_term + throughput_term + fairness_term + flow_fairness_term
        )
        return scenario_score, total_weight
