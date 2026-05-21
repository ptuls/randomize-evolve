import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from initial_program import candidate_factory
from loguru import logger
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import OpenEvolveRunner
from randomize_evolve.workflow.functions import FunctionEvolutionScenario
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

try:  # Newer releases expose OpenEvolve at package root
    from openevolve import OpenEvolve
except ImportError:  # pragma: no cover - compatibility shim
    from openevolve.core import OpenEvolve  # type: ignore


def _load_initial_program_source() -> ProgramSource:
    """Load the set-membership seed program from the repo baseline file."""
    seed_path = Path(__file__).with_name("initial_program.py")
    return ProgramSource(seed_path.read_text(encoding="utf-8"))


INITIAL_PROGRAM_SOURCE = _load_initial_program_source()

_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_CONFIG_LOADER = ConfigLoader()
_SET_MEMBERSHIP_SEED_DIR = Path(__file__).resolve().parent / "set_membership_seeds"


@dataclass(frozen=True)
class NamedProgramSource:
    """Associates a human-readable seed name with the seed program source."""

    name: str
    source: ProgramSource


def _build_runner() -> OpenEvolveRunner:
    return OpenEvolveRunner(OpenEvolve, _EVALUATOR_PATH)


def _build_workflow(provider) -> "EvolutionWorkflow":
    return _build_workflow_with_source(INITIAL_PROGRAM_SOURCE, provider)


def _build_workflow_with_source(
    program_source: ProgramSource, provider
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


def _load_program_source(path: Path) -> ProgramSource:
    """Load a seed program from disk."""
    return ProgramSource(path.read_text(encoding="utf-8"))


def _load_curriculum_seed_portfolio() -> tuple[NamedProgramSource, ...]:
    """Load diverse seed families for broad exploration."""
    return (
        NamedProgramSource("bloom", INITIAL_PROGRAM_SOURCE),
        NamedProgramSource(
            "cuckoo_filter",
            _load_program_source(_SET_MEMBERSHIP_SEED_DIR / "cuckoo_filter.py"),
        ),
        NamedProgramSource(
            "fingerprint_probe",
            _load_program_source(_SET_MEMBERSHIP_SEED_DIR / "fingerprint_probe.py"),
        ),
    )


def _allocate_portfolio_iterations(
    total_iterations: int, portfolio_size: int
) -> tuple[int, ...]:
    """Spread a fixed exploration budget across the portfolio."""
    if portfolio_size <= 0:
        return ()
    per_seed = total_iterations // portfolio_size
    remainder = total_iterations % portfolio_size
    return tuple(
        per_seed + (1 if index < remainder else 0)
        for index in range(portfolio_size)
    )


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
        "Portfolio seed {} finished: combined_score={:.4f}, fp_rate={:.4f}, "
        "bits_per_item={:.4f}, bloom_excess={:.4f}".format(
            seed_name,
            _result_score(result),
            float(metrics.get("false_positive_rate", float("nan"))),
            float(metrics.get("bits_per_item", float("nan"))),
            float(metrics.get("excess_bits_per_item_vs_bloom", float("nan"))),
        )
    )


def _run_portfolio_exploration(
    *,
    total_iterations: int,
    explore_config: str,
) -> object:
    """Run broad exploration from multiple seed families and keep the best result."""
    portfolio = _load_curriculum_seed_portfolio()
    allocations = _allocate_portfolio_iterations(total_iterations, len(portfolio))

    best_result = None
    best_seed_name = None
    best_score = float("-inf")

    for seed, iterations in zip(portfolio, allocations):
        if iterations <= 0:
            continue
        logger.info(
            "Starting portfolio exploration from seed '{}' for {} iterations",
            seed.name,
            iterations,
        )
        provider = YamlConfigProvider(Path(explore_config), _CONFIG_LOADER)
        workflow = _build_workflow_with_source(seed.source, provider)
        result = workflow.execute(iterations)
        if result is None or not getattr(result, "code", None):
            logger.warning(
                "Portfolio seed '{}' produced no usable best program", seed.name
            )
            continue
        _log_portfolio_result(seed.name, result)
        score = _result_score(result)
        if score > best_score:
            best_result = result
            best_seed_name = seed.name
            best_score = score

    if best_result is None:
        raise RuntimeError("Seed portfolio exploration did not produce a usable result")

    logger.info(
        "Selected portfolio winner '{}' with combined_score={:.4f}",
        best_seed_name,
        best_score,
    )
    return best_result


def demo_run_evolution_simple(iterations: int = 5) -> None:
    workflow = _build_workflow(MinimalConfigProvider())
    workflow.execute(iterations)


def demo_run_evolution(
    iterations: int = 25, config_file: str = "configs/uniform_workload.yaml"
) -> None:
    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def demo_evolve_function(iterations: int = 10) -> None:
    FunctionEvolutionScenario(candidate_factory).run(iterations)


def demo_run_curriculum(
    *,
    explore_iterations: int = 10,
    exploit_iterations: int = 15,
    explore_config: str = "configs/aggressive_exploration.yaml",
    exploit_config: str = "configs/uniform_workload.yaml",
) -> None:
    """Run a broad-then-narrow set-membership search curriculum."""
    explore_result = _run_portfolio_exploration(
        total_iterations=explore_iterations,
        explore_config=explore_config,
    )

    if explore_result is None or not getattr(explore_result, "code", None):
        raise RuntimeError("Exploration phase did not produce a usable best program")

    exploit_provider = YamlConfigProvider(Path(exploit_config), _CONFIG_LOADER)
    exploit_source = ProgramSource(explore_result.code)
    exploit_workflow = _build_workflow_with_source(exploit_source, exploit_provider)
    exploit_workflow.execute(exploit_iterations)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for set-membership evolution runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of evolution iterations to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/uniform_workload.yaml",
        help="Path to the OpenEvolve YAML config file.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Run a broad exploration phase followed by a narrowing phase.",
    )
    parser.add_argument(
        "--explore-iterations",
        type=int,
        default=10,
        help="Iterations to spend in the broad exploration phase.",
    )
    parser.add_argument(
        "--exploit-iterations",
        type=int,
        default=15,
        help="Iterations to spend in the narrowing phase.",
    )
    parser.add_argument(
        "--explore-config",
        default="configs/aggressive_exploration.yaml",
        help="Config to use for the broad exploration phase.",
    )
    parser.add_argument(
        "--exploit-config",
        default="configs/uniform_workload.yaml",
        help="Config to use for the narrowing phase.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the set-membership workflow from a real file-backed CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    if args.curriculum:
        demo_run_curriculum(
            explore_iterations=args.explore_iterations,
            exploit_iterations=args.exploit_iterations,
            explore_config=args.explore_config,
            exploit_config=args.exploit_config,
        )
        return

    demo_run_evolution(iterations=args.iterations, config_file=args.config)


if __name__ == "__main__":
    main()
