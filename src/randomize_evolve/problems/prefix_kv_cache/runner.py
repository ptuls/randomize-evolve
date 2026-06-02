"""Run prefix KV-cache baseline reports or Levi evolution."""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from randomize_evolve.evaluators.prefix_kv_cache import (
    BASELINES,
    REPORTING_BASELINES,
    EvaluatorConfig,
    EvaluationResult,
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
    artifact_output: Path | None = Path("artifacts/prefix_kv_cache_runs"),
) -> object:
    provider = (
        MinimalConfigProvider() if quick else YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    )
    workflow = _build_workflow(provider)
    result = workflow.execute(iterations)
    if artifact_output is not None:
        artifact_dir = save_run_artifacts(
            result,
            artifact_output,
            iterations=iterations,
            config_label=provider.describe(),
        )
        print(f"saved_run_artifacts={artifact_dir}")
    return result


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
    for name, result in _evaluate_baselines(config).items():
        print(f"{name}: combined_score={result.combined_score:.3f}")
        for workload, metrics in result.workload_metrics.items():
            print(
                "  "
                f"{workload}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
                f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
                f"churn_per_1k={metrics['cache_churn_per_1k']:.1f}"
            )


def write_baseline_plots(
    output_dir: Path,
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    block_size_tokens: int | None = None,
) -> tuple[Path, ...]:
    """Write lightweight SVG plots for baseline comparison and debugging."""

    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        block_size_tokens=block_size_tokens,
    )
    results = _evaluate_baselines(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "baseline_combined_scores.svg",
        output_dir / "validation_token_hit_heatmap.svg",
        output_dir / "token_vs_block_hit.svg",
    )
    paths[0].write_text(_combined_score_svg(results), encoding="utf-8")
    paths[1].write_text(_validation_heatmap_svg(results), encoding="utf-8")
    paths[2].write_text(_token_vs_block_svg(results), encoding="utf-8")
    return paths


def save_run_artifacts(
    result: object,
    output_root: Path,
    *,
    iterations: int,
    config_label: str,
    timestamp: datetime | None = None,
) -> Path:
    """Persist the best evolved program and evaluation metadata."""

    timestamp = timestamp or datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_program = (
        getattr(result, "best_program", None)
        or getattr(result, "best_code", None)
        or getattr(result, "code", "")
        or ""
    )
    (run_dir / "best_program.py").write_text(str(best_program), encoding="utf-8")

    metrics = getattr(result, "metrics", {}) or {}
    artifacts = getattr(result, "artifacts", {}) or {}
    metadata = getattr(result, "metadata", {}) or {}
    summary = {
        "run_id": run_id,
        "iterations": iterations,
        "config": config_label,
        "best_score": getattr(result, "best_score", None),
        "total_evaluations": getattr(result, "total_evaluations", None),
        "total_cost": getattr(result, "total_cost", None),
        "archive_size": getattr(result, "archive_size", None),
        "runtime_seconds": getattr(result, "runtime_seconds", None),
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "artifacts.json", artifacts)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "run_summary.json", summary)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    return run_dir


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
        "--plot-report",
        action="store_true",
        help="Write SVG baseline plots without launching Levi.",
    )
    parser.add_argument(
        "--plot-output",
        default="artifacts/prefix_kv_cache_plots",
        help="Directory for --plot-report SVG files.",
    )
    parser.add_argument(
        "--artifact-output",
        default="artifacts/prefix_kv_cache_runs",
        help="Directory for saved evolution run artifacts.",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not save best_program.py and run metadata after evolution.",
    )
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
    if args.plot_report:
        paths = write_baseline_plots(
            Path(args.plot_output),
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        for path in paths:
            print(path)
        return
    demo_run_evolution(
        iterations=args.iterations,
        config_file=args.config,
        quick=args.quick,
        artifact_output=None if args.no_save_artifacts else Path(args.artifact_output),
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


def _evaluate_baselines(config: EvaluatorConfig) -> dict[str, EvaluationResult]:
    results: dict[str, EvaluationResult] = {}
    for name, factory in BASELINES.items():
        evaluator = PrefixKVCacheEvaluator(config)
        results[name] = evaluator(factory)
    return results


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _combined_score_svg(results: dict[str, EvaluationResult]) -> str:
    width = 920
    row_height = 34
    left = 190
    top = 54
    right = 40
    bottom = 36
    height = top + row_height * len(results) + bottom
    min_score = min(result.combined_score for result in results.values())
    max_score = max(result.combined_score for result in results.values())
    span = max(max_score - min_score, 1.0)
    plot_width = width - left - right
    lines = [_svg_header(width, height, "Prefix KV-cache Baseline Scores")]
    lines.append(_text(24, 30, "Baseline combined scores", size=20, weight="700"))
    for index, (name, result) in enumerate(
        sorted(results.items(), key=lambda item: item[1].combined_score, reverse=True)
    ):
        y = top + index * row_height
        score = result.combined_score
        bar_width = max(2.0, (score - min_score) / span * plot_width)
        lines.append(_text(16, y + 21, name, size=13))
        lines.append(
            f'<rect x="{left}" y="{y + 5}" width="{bar_width:.2f}" '
            f'height="20" fill="#2563eb" rx="2" />'
        )
        lines.append(_text(left + bar_width + 8, y + 21, f"{score:.1f}", size=12))
    lines.append("</svg>")
    return "\n".join(lines)


def _validation_heatmap_svg(results: dict[str, EvaluationResult]) -> str:
    workloads = _validation_workloads(results)
    cell_w = 120
    cell_h = 32
    left = 190
    top = 86
    width = left + cell_w * len(workloads) + 36
    height = top + cell_h * len(results) + 42
    lines = [_svg_header(width, height, "Validation Token Hit Rate Heatmap")]
    lines.append(_text(24, 30, "Validation token hit rate", size=20, weight="700"))
    for col, workload in enumerate(workloads):
        label = workload.split("/", 1)[1]
        lines.append(_text(left + col * cell_w + 6, 68, label, size=11))
    for row, (name, result) in enumerate(results.items()):
        y = top + row * cell_h
        lines.append(_text(16, y + 21, name, size=13))
        for col, workload in enumerate(workloads):
            x = left + col * cell_w
            rate = float(result.workload_metrics[workload]["token_hit_rate"])
            fill = _blue_scale(rate)
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" '
                f'fill="{fill}" rx="2" />'
            )
            lines.append(_text(x + 8, y + 20, f"{rate:.3f}", size=12))
    lines.append("</svg>")
    return "\n".join(lines)


def _token_vs_block_svg(results: dict[str, EvaluationResult]) -> str:
    width = 720
    height = 520
    left = 72
    top = 48
    plot_w = 450
    plot_h = 380
    points = []
    for name, result in results.items():
        validation = result.split_metrics["validation"]
        points.append(
            (
                name,
                float(validation["block_hit_rate"]),
                float(validation["token_hit_rate"]),
                result.combined_score,
            )
        )
    lines = [_svg_header(width, height, "Token vs Block Hit Rate")]
    lines.append(_text(24, 30, "Validation token vs block hit rate", size=20, weight="700"))
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="#f8fafc" stroke="#cbd5e1" />'
    )
    for tick in range(0, 6):
        value = tick / 5
        x = left + value * plot_w
        y = top + plot_h - value * plot_h
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e8f0" />'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0" />'
        )
        lines.append(_text(x - 10, top + plot_h + 20, f"{value:.1f}", size=11))
        lines.append(_text(28, y + 4, f"{value:.1f}", size=11))
    lines.append(_text(left + 165, height - 26, "block hit rate", size=13))
    lines.append(_text(14, top + 190, "token", size=13))
    for index, (name, block_rate, token_rate, score) in enumerate(points):
        x = left + block_rate * plot_w
        y = top + plot_h - token_rate * plot_h
        color = _palette(index)
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}">'
            f"<title>{html.escape(name)} score={score:.1f}</title></circle>"
        )
        lines.append(_text(left + plot_w + 24, top + 24 + index * 24, name, size=12, fill=color))
    lines.append("</svg>")
    return "\n".join(lines)


def _validation_workloads(results: dict[str, EvaluationResult]) -> list[str]:
    first = next(iter(results.values()))
    return [key for key in first.workload_metrics if key.startswith("validation/")]


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" fill="white" />'
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    weight: str = "400",
    fill: str = "#0f172a",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f"{html.escape(value)}</text>"
    )


def _blue_scale(value: float) -> str:
    value = max(0.0, min(1.0, value))
    lightness = int(94 - value * 46)
    return f"hsl(214, 78%, {lightness}%)"


def _palette(index: int) -> str:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5")
    return colors[index % len(colors)]


if __name__ == "__main__":
    main()
