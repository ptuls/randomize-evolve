"""Run prefix KV-cache baseline reports or Levi evolution."""

from __future__ import annotations

import argparse
from pathlib import Path

from randomize_evolve.evaluators.prefix_kv_cache import (
    BASELINES,
    REPORTING_BASELINES,
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
)
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.workflow.execution import LeviRunner
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

from .initial_program import build_candidate

_INITIAL_PROGRAM_PATH = Path(__file__).parent / "initial_program.py"
INITIAL_PROGRAM_SOURCE = ProgramSource(_INITIAL_PROGRAM_PATH.read_text(encoding="utf-8"))
_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_CONFIG_LOADER = ConfigLoader()


def _build_runner() -> LeviRunner:
    import levi

    return LeviRunner(
        levi.evolve_code,
        _EVALUATOR_PATH,
        problem_description=(
            "Search for simple prefix KV-cache admission and eviction scoring "
            "heuristics that generalize across shifted LLM-serving workloads."
        ),
        function_signature=(
            "def build_candidate(capacity_blocks: int, block_size_tokens: int, "
            "seed: int | None = None):"
        ),
    )


def _build_workflow(provider) -> object:
    from randomize_evolve.workflow.workflow import EvolutionWorkflow

    return EvolutionWorkflow(
        program_source=INITIAL_PROGRAM_SOURCE,
        config_provider=provider,
        runner=_build_runner(),
        reporter=EvolutionReporter(),
    )


def demo_run_evolution(
    iterations: int = 25,
    config_file: str = "configs/prefix_kv_cache.yaml",
    *,
    quick: bool = False,
) -> None:
    provider = MinimalConfigProvider() if quick else YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def compare_baselines(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    block_size_tokens: int | None = None,
) -> None:
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    for name, factory in BASELINES.items():
        evaluator = PrefixKVCacheEvaluator(config)
        result = evaluator(factory)
        print(f"{name}: combined_score={result.combined_score:.3f}")
        for workload, metrics in result.workload_metrics.items():
            print(
                "  "
                f"{workload}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
                f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
                f"churn_per_1k={metrics['cache_churn_per_1k']:.1f}"
            )


def hidden_report(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    block_size_tokens: int | None = None,
) -> None:
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    hidden_evaluator = PrefixKVCacheEvaluator(config, splits=("hidden",))
    print("champion_initial:")
    champion = hidden_evaluator(build_candidate)
    print(f"  combined_score={champion.combined_score:.3f}")
    for name, factory in REPORTING_BASELINES.items():
        evaluator = PrefixKVCacheEvaluator(
            config,
            splits=("hidden",),
            expose_future_reuse=name == "future_reuse_heuristic",
        )
        result = evaluator(factory)
        print(f"{name}: combined_score={result.combined_score:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--workload-preset",
        default="default",
        choices=("default", "small"),
    )
    parser.add_argument("--capacity-blocks", type=int, default=None)
    parser.add_argument("--block-size-tokens", type=int, default=None)
    parser.add_argument("--baseline-report", action="store_true")
    parser.add_argument("--hidden-report", action="store_true")
    parser.add_argument(
        "--config",
        default="configs/prefix_kv_cache.yaml",
        help="Path to the Levi YAML config file.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.baseline_report:
        compare_baselines(
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        return
    if args.hidden_report:
        hidden_report(
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        return
    demo_run_evolution(
        iterations=args.iterations,
        config_file=args.config,
        quick=args.quick,
    )


def _config_from_args(
    *,
    quick: bool,
    capacity_blocks: int | None,
    block_size_tokens: int | None,
) -> EvaluatorConfig:
    config = EvaluatorConfig(
        request_count=36 if quick else EvaluatorConfig.request_count,
        seeds=(3,) if quick else EvaluatorConfig.seeds,
        capacity_blocks=capacity_blocks or EvaluatorConfig.capacity_blocks,
        block_size_tokens=block_size_tokens or EvaluatorConfig.block_size_tokens,
    )
    return config


if __name__ == "__main__":
    main()
