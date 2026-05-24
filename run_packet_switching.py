"""Run and compare packet-switching scheduler evolution experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from initial_program_packet_switching import candidate_factory as baseline_candidate_factory
from loguru import logger
from packet_switching_seeds.max_weight_greedy import (
    candidate_factory as max_weight_greedy_candidate_factory,
)
from packet_switching_seeds.randomized_iterative import (
    candidate_factory as randomized_iterative_candidate_factory,
)
from packet_switching_seeds.sticky_matching import (
    candidate_factory as sticky_matching_candidate_factory,
)
from packet_switching_seeds.weighted_islip import (
    candidate_factory as weighted_islip_candidate_factory,
)
from randomize_evolve.evaluators.packet_switching import (
    PacketSwitchingEvaluator,
    PacketSwitchingEvaluatorConfig,
    ScenarioConfig,
)
from randomize_evolve.packet_switching import RoundRobinScheduler
from randomize_evolve.traffic import TrafficPatternConfig, TrafficPatternType
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import OpenEvolveRunner
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

try:
    from openevolve import OpenEvolve
except ImportError:  # pragma: no cover - compatibility shim.
    from openevolve.core import OpenEvolve  # type: ignore


def _load_initial_program_source() -> ProgramSource:
    """Load the packet-switching baseline program from disk."""

    seed_path = Path(__file__).with_name("initial_program_packet_switching.py")
    return ProgramSource(seed_path.read_text(encoding="utf-8"))


def _load_evolution_program_source() -> ProgramSource:
    """Load the abstract packet-switching evolution seed from disk."""

    seed_path = Path(__file__).with_name("initial_program_packet_switching_evolution.py")
    return ProgramSource(seed_path.read_text(encoding="utf-8"))


INITIAL_PROGRAM_SOURCE = _load_initial_program_source()
EVOLUTION_PROGRAM_SOURCE = _load_evolution_program_source()

_EVALUATOR_PATH = Path(__file__).parent / "packet_switching_evaluator.py"
_CONFIG_LOADER = ConfigLoader()
_PACKET_SWITCHING_SEED_DIR = Path(__file__).resolve().parent / "packet_switching_seeds"


@dataclass(frozen=True)
class NamedProgramSource:
    """Associates a human-readable seed name with the seed program source."""

    name: str
    source: ProgramSource


def _build_runner() -> OpenEvolveRunner:
    return OpenEvolveRunner(OpenEvolve, _EVALUATOR_PATH)


def _build_workflow(provider) -> "EvolutionWorkflow":
    return _build_workflow_with_source(EVOLUTION_PROGRAM_SOURCE, provider)


def _build_workflow_with_source(
    program_source: ProgramSource,
    provider,
) -> "EvolutionWorkflow":
    from randomize_evolve.workflow.workflow import EvolutionWorkflow

    runner = _build_runner()
    reporter = EvolutionReporter()
    return EvolutionWorkflow(
        program_source=program_source,
        config_provider=provider,
        runner=runner,
        reporter=reporter,
    )


def demo_run_evolution_simple(iterations: int = 5) -> None:
    """Run a minimal in-memory evolution configuration."""

    workflow = _build_workflow(MinimalConfigProvider())
    workflow.execute(iterations)


def demo_run_evolution(
    iterations: int = 25,
    config_file: str = "configs/packet_switching_workload.yaml",
) -> None:
    """Run packet-switching evolution using the configured YAML workload."""

    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def _load_program_source(path: Path) -> ProgramSource:
    """Load a packet-switching seed program from disk."""

    return ProgramSource(path.read_text(encoding="utf-8"))


def _load_seed_portfolio() -> tuple[NamedProgramSource, ...]:
    """Load diverse scheduler seeds for broad exploration."""

    return (
        NamedProgramSource("abstract_scaffold", EVOLUTION_PROGRAM_SOURCE),
        NamedProgramSource("voq_round_robin", INITIAL_PROGRAM_SOURCE),
        NamedProgramSource(
            "weighted_islip",
            _load_program_source(_PACKET_SWITCHING_SEED_DIR / "weighted_islip.py"),
        ),
        NamedProgramSource(
            "max_weight_greedy",
            _load_program_source(_PACKET_SWITCHING_SEED_DIR / "max_weight_greedy.py"),
        ),
        NamedProgramSource(
            "randomized_iterative",
            _load_program_source(_PACKET_SWITCHING_SEED_DIR / "randomized_iterative.py"),
        ),
        NamedProgramSource(
            "sticky_matching",
            _load_program_source(_PACKET_SWITCHING_SEED_DIR / "sticky_matching.py"),
        ),
        NamedProgramSource(
            "column_pressure",
            _load_program_source(_PACKET_SWITCHING_SEED_DIR / "column_pressure.py"),
        ),
        NamedProgramSource(
            "input_aged_round_robin",
            _load_program_source(_PACKET_SWITCHING_SEED_DIR / "input_aged_round_robin.py"),
        ),
    )


def _allocate_portfolio_iterations(
    total_iterations: int,
    portfolio_size: int,
) -> tuple[int, ...]:
    """Spread a fixed exploration budget across the seed portfolio."""

    if portfolio_size <= 0:
        return ()
    per_seed = total_iterations // portfolio_size
    remainder = total_iterations % portfolio_size
    return tuple(per_seed + (1 if index < remainder else 0) for index in range(portfolio_size))


def _result_score(result) -> float:
    """Extract a comparable scalar score from an evolution result."""

    metrics = getattr(result, "metrics", {}) or {}
    score = metrics.get("combined_score")
    if isinstance(score, (int, float)):
        return float(score)
    return float("-inf")


def _log_portfolio_result(seed_name: str, result) -> None:
    """Emit a compact summary for a portfolio exploration run."""

    metrics = getattr(result, "metrics", {}) or {}
    logger.info(
        "Portfolio seed {} finished: combined_score={:.6f}, "
        "mean_average_total_queue={:.4f}, throughput={:.4f}, "
        "flow_fairness={:.4f}".format(
            seed_name,
            _result_score(result),
            float(metrics.get("mean_average_total_queue", float("nan"))),
            float(metrics.get("mean_throughput", float("nan"))),
            float(metrics.get("mean_flow_fairness", float("nan"))),
        )
    )


def demo_run_portfolio(
    total_iterations: int = 25,
    config_file: str = "configs/packet_switching_workload.yaml",
) -> object:
    """Explore from multiple packet-switching seeds, then exploit the best."""

    portfolio = _load_seed_portfolio()
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if len(portfolio) == 1:
        return demo_run_evolution(iterations=total_iterations, config_file=config_file)

    reserved_exploit_iterations = min(5, max(1, total_iterations // 5))
    explore_iterations = min(
        total_iterations,
        max(len(portfolio) * 2, total_iterations - reserved_exploit_iterations),
    )
    exploit_iterations = max(0, total_iterations - explore_iterations)
    allocations = _allocate_portfolio_iterations(explore_iterations, len(portfolio))

    best_result = None
    best_seed_name = None
    best_score = float("-inf")

    for seed, iterations in zip(portfolio, allocations):
        if iterations <= 0:
            continue
        logger.info(
            "Starting packet-switching exploration from seed '{}' for {} iterations",
            seed.name,
            iterations,
        )
        provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
        workflow = _build_workflow_with_source(seed.source, provider)
        result = workflow.execute(iterations)
        if result is None or not getattr(result, "code", None):
            logger.warning(
                "Seed '{}' produced no usable best program during exploration",
                seed.name,
            )
            continue
        _log_portfolio_result(seed.name, result)
        score = _result_score(result)
        if score > best_score:
            best_result = result
            best_seed_name = seed.name
            best_score = score

    if best_result is None:
        raise RuntimeError("Packet-switching seed portfolio produced no usable result")

    logger.info(
        "Selected packet-switching portfolio winner '{}' with combined_score={:.6f}",
        best_seed_name,
        best_score,
    )

    if exploit_iterations <= 0:
        return best_result

    logger.info(
        "Exploiting packet-switching portfolio winner '{}' for {} iterations",
        best_seed_name,
        exploit_iterations,
    )
    exploit_provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    exploit_source = ProgramSource(best_result.code)
    exploit_workflow = _build_workflow_with_source(exploit_source, exploit_provider)
    return exploit_workflow.execute(exploit_iterations)


def _matrix_scenarios() -> list[ScenarioConfig]:
    """Build queueing scenarios directly from arrival-rate matrices."""

    return [
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
            time_slots=1500,
            warmup_slots=200,
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
            time_slots=1500,
            warmup_slots=200,
        ),
        ScenarioConfig(
            name="asymmetric-heavy",
            pattern=TrafficPatternConfig(
                pattern_type=TrafficPatternType.ARRIVAL_MATRIX,
                arrival_matrix=(
                    (0.65, 0.18, 0.0, 0.0),
                    (0.12, 0.35, 0.2, 0.0),
                    (0.0, 0.18, 0.5, 0.12),
                    (0.0, 0.0, 0.12, 0.58),
                ),
            ),
            time_slots=1800,
            warmup_slots=300,
        ),
    ]


def compare_baselines() -> None:
    """Print a compact comparison between scheduler baselines."""

    config = PacketSwitchingEvaluatorConfig(
        ports=4,
        scenarios=_matrix_scenarios(),
        seed=17,
    )
    evaluator = PacketSwitchingEvaluator(config)
    scheduler_factories = [
        ("round_robin", lambda ports: RoundRobinScheduler(ports, ports)),
        ("voq_round_robin", baseline_candidate_factory),
        ("weighted_islip", weighted_islip_candidate_factory),
        ("max_weight_greedy", max_weight_greedy_candidate_factory),
        ("randomized_iterative", randomized_iterative_candidate_factory),
        ("sticky_matching", sticky_matching_candidate_factory),
    ]

    for name, factory in scheduler_factories:
        evaluation = evaluator(factory)
        print(f"{name}: score={evaluation.score:.4f}")
        for scenario_result in evaluation.scenario_results:
            metrics = scenario_result.metrics
            print(
                "  "
                f"{scenario_result.config.name}: "
                f"avg_total_queue={metrics.average_total_queue:.2f}, "
                f"throughput={metrics.throughput:.3f}, "
                f"flow_fairness={metrics.fairness_flows:.3f}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for packet-switching runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of evolution iterations to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/packet_switching_workload.yaml",
        help="Path to the OpenEvolve YAML config file.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Print baseline comparisons without launching OpenEvolve.",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Explore from multiple packet-switching seeds before exploiting the best.",
    )
    return parser


def main() -> None:
    """Run the packet-switching comparison or evolution workflow."""

    args = build_arg_parser().parse_args()
    if args.compare_only:
        compare_baselines()
        return
    if args.portfolio:
        demo_run_portfolio(total_iterations=args.iterations, config_file=args.config)
        return
    demo_run_evolution(iterations=args.iterations, config_file=args.config)


if __name__ == "__main__":
    main()
