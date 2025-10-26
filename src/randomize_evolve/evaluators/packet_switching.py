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
    throughput_weight: float = 0.6
    fairness_weight: float = 0.3
    flow_fairness_weight: float = 0.1
    drop_weight: float = 0.4
    seed_offset: int = 0


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

    ports: int = 8
    scenarios: Sequence[ScenarioConfig] = field(default_factory=list)
    seed: int = 7

    def __post_init__(self) -> None:
        if not self.scenarios:
            self.scenarios = default_scenarios()


def default_scenarios() -> List[ScenarioConfig]:
    """Return a curated set of default traffic scenarios."""

    return [
        ScenarioConfig(
            name="uniform-medium",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.UNIFORM,
                offered_load=0.6,
            ),
            throughput_weight=0.7,
            fairness_weight=0.25,
            flow_fairness_weight=0.05,
            drop_weight=0.3,
        ),
        ScenarioConfig(
            name="bursty",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.BURSTY,
                offered_load=0.4,
                burst_rate=5,
                burst_length=6,
                burst_probability=0.12,
            ),
            throughput_weight=0.65,
            fairness_weight=0.25,
            flow_fairness_weight=0.1,
            drop_weight=0.35,
        ),
        ScenarioConfig(
            name="hotspot-heavy",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.HOTSPOT,
                offered_load=0.75,
                hotspot_probability=0.65,
            ),
            throughput_weight=0.55,
            fairness_weight=0.35,
            flow_fairness_weight=0.1,
            drop_weight=0.45,
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
            throughput_weight=0.6,
            fairness_weight=0.3,
            flow_fairness_weight=0.1,
            drop_weight=0.5,
        ),
    ]


class PacketSwitchingEvaluator:
    """Callable evaluator compatible with the OpenEvolve workflow."""

    def __init__(self, config: Optional[PacketSwitchingEvaluatorConfig] = None) -> None:
        self.config = config or PacketSwitchingEvaluatorConfig()

    def __call__(self, factory: Optional[SchedulerFactory] = None) -> PacketSwitchingEvaluation:
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
                queue_limit=scenario.queue_limit,
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

    def _score(self, metrics: SimulationResult, scenario: ScenarioConfig) -> tuple[float, float]:
        weights = (
            scenario.throughput_weight,
            scenario.fairness_weight,
            scenario.flow_fairness_weight,
            scenario.drop_weight,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Scenario weight configuration must be positive")
        throughput_term = scenario.throughput_weight * (1.0 - metrics.throughput)
        fairness_term = scenario.fairness_weight * (1.0 - metrics.fairness_inputs)
        flow_fairness_term = scenario.flow_fairness_weight * (1.0 - metrics.fairness_flows)
        drop_term = scenario.drop_weight * metrics.drop_rate
        scenario_score = throughput_term + fairness_term + flow_fairness_term + drop_term
        return scenario_score, total_weight
