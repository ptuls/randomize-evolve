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
    scoring_fn_complexity,
)
from randomize_evolve.workflow.configuration import (
    ConfigLoader,
    MinimalConfigProvider,
    YamlConfigProvider,
)
from randomize_evolve.evaluator_entry import load_candidate_factory, run_with_timeout
from randomize_evolve.workflow.execution import LeviRunner
from randomize_evolve.workflow.program import ProgramSource
from randomize_evolve.workflow.reporting import EvolutionReporter

from .initial_program import build_candidate

_INITIAL_PROGRAM_PATH = Path(__file__).parent / "initial_program.py"
INITIAL_PROGRAM_SOURCE = ProgramSource(
    _INITIAL_PROGRAM_PATH.read_text(encoding="utf-8")
)
_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"
_CONFIG_LOADER = ConfigLoader()
_DEFAULT_CAPACITY_SWEEP_BLOCKS = (24, 48)
_QUICK_REPORT_WARNING = (
    "SMOKE-ONLY: `--quick` uses `request_count=36` and one seed. "
    "Do not use this table for policy ranking decisions; rerun without `--quick`."
)


def _build_runner() -> LeviRunner:
    import levi

    return LeviRunner(
        levi.evolve_code,
        _EVALUATOR_PATH,
        problem_description=(
            "Search for simple prefix KV-cache admission and eviction scoring "
            "heuristics that generalize across shifted LLM-serving workloads. "
            "PrefixBlockInfo is a frozen per-callback value object; use "
            "block.prefix_hash or block.block_id as the stable key, never "
            "id(block) or guessed fallback attributes. Documented block fields "
            "are block_id, prefix_hash, parent_hash, depth, start_token, "
            "end_token, token_count, tenant_id, created_at, last_accessed_at, "
            "hit_count, descendant_count, active_ref_count, "
            "estimated_recompute_cost, estimated_future_reuse, and "
            "estimated_next_reuse_distance. Future-reuse fields are None for "
            "deployable candidates. The only lifecycle callbacks that fire are "
            "on_request_start, on_cache_hit, and on_cache_miss. Do not add "
            "on_request_end, on_block_admitted, on_block_evicted, or state that "
            "depends on unsupported callbacks. session_id is request-only "
            "metadata; PrefixBlockInfo has tenant_id but no session_id."
        ),
        function_signature=(
            "def build_candidate(capacity_blocks: int, block_size_tokens: int, "
            "seed: int | None = None):"
        ),
    )


def _build_workflow(
    provider,
    *,
    program_source: ProgramSource = INITIAL_PROGRAM_SOURCE,
) -> object:
    from randomize_evolve.workflow.workflow import EvolutionWorkflow

    return EvolutionWorkflow(
        program_source=program_source,
        config_provider=provider,
        runner=_build_runner(),
        reporter=EvolutionReporter(),
    )


def _load_seed_program_source(path: Path) -> ProgramSource:
    """Load an evolution seed from a candidate file or saved run directory."""

    candidate_path = _resolve_candidate_program(path)
    return ProgramSource(candidate_path.read_text(encoding="utf-8"))


def demo_run_evolution(
    iterations: int = 25,
    config_file: str = "configs/prefix_kv_cache.yaml",
    *,
    quick: bool = False,
    seed_program: Path | None = None,
    artifact_output: Path | None = Path("artifacts/prefix_kv_cache_runs"),
) -> object:
    provider = (
        MinimalConfigProvider()
        if quick
        else YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    )
    program_source = (
        _load_seed_program_source(seed_program)
        if seed_program is not None
        else INITIAL_PROGRAM_SOURCE
    )
    workflow = _build_workflow(provider, program_source=program_source)
    result = workflow.execute(iterations)
    if artifact_output is not None:
        artifact_dir = save_run_artifacts(
            result,
            artifact_output,
            iterations=iterations,
            config_label=provider.describe(),
            seed_label=str(seed_program or _INITIAL_PROGRAM_PATH),
        )
        print(f"saved_run_artifacts={artifact_dir}")
        print(f"baseline_comparison={artifact_dir / 'baseline_comparison.md'}")
    return result


def compare_baselines(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
) -> None:
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
    )
    if quick:
        print(_QUICK_REPORT_WARNING)
    results = _evaluate_baselines(config, include_reporting=True)
    if candidate_program is not None:
        candidate_path = _resolve_candidate_program(candidate_program)
        results = {
            "candidate": _evaluate_candidate_program(config, candidate_path),
            **results,
        }
        report_path = candidate_path.parent / "baseline_comparison.md"
        write_baseline_comparison_report(
            report_path,
            results,
            candidate_path=candidate_path,
            command=_baseline_report_command(
                quick=quick,
                capacity_sweep_blocks=capacity_sweep_blocks,
                candidate_program=candidate_program,
            ),
            quick=quick,
            config=config,
        )
        print(f"baseline_comparison={report_path}")
    for name, result in results.items():
        print(
            f"{name}: combined_score={result.combined_score:.3f} [{_baseline_group(name)}]"
        )
        for capacity, metrics in result.capacity_metrics.items():
            print(
                "  "
                f"{capacity}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
                f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
                f"churn_per_1k={metrics['cache_churn_per_1k']:.1f}"
            )
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
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
) -> tuple[Path, ...]:
    """Write lightweight SVG plots for baseline comparison and debugging."""

    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
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
    seed_label: str | None = None,
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
        "seed_program": seed_label,
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

    try:
        config = _artifact_report_config()
        candidate_path = run_dir / "best_program.py"
        report_results = {
            "candidate": _evaluate_candidate_program(config, candidate_path),
            **_evaluate_baselines(config, include_reporting=True),
        }
        write_baseline_comparison_report(
            run_dir / "baseline_comparison.md",
            report_results,
            candidate_path=candidate_path,
            command=(
                ".venv/bin/python -m randomize_evolve.problems.prefix_kv_cache.runner "
                "--baseline-report --capacity-sweep-blocks 24,48 "
                f"--candidate-program {run_dir}"
            ),
            quick=False,
            config=config,
        )
    except Exception as exc:
        _write_json(
            run_dir / "baseline_comparison_error.json",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    return run_dir


def hidden_report(
    *,
    quick: bool = False,
    capacity_blocks: int | None = None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None = None,
    candidate_program: Path | None = None,
) -> None:
    config = _config_from_args(
        quick=quick,
        capacity_blocks=capacity_blocks,
        capacity_sweep_blocks=capacity_sweep_blocks,
        block_size_tokens=block_size_tokens,
    )
    if candidate_program is None:
        print("initial_candidate:")
        champion = PrefixKVCacheEvaluator(config, splits=("hidden",))(build_candidate)
    else:
        candidate_path = _resolve_candidate_program(candidate_program)
        print(f"candidate={candidate_path}")
        champion = _evaluate_candidate_program(
            config, candidate_path, splits=("hidden",)
        )
    print(f"  combined_score={champion.combined_score:.3f}")
    for name, factory in REPORTING_BASELINES.items():
        evaluator = PrefixKVCacheEvaluator(
            config,
            splits=("hidden",),
            expose_future_reuse=_requires_future_reuse(name),
        )
        result = evaluator(factory)
        print(f"{name}: combined_score={result.combined_score:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smoke-only single-seed slice; do not use it for ranking decisions.",
    )
    parser.add_argument(
        "--workload-preset",
        default="default",
        choices=("default", "small"),
    )
    parser.add_argument("--capacity-blocks", type=int, default=None)
    parser.add_argument(
        "--capacity-sweep-blocks",
        default="",
        help="Comma-separated capacities to evaluate, for example 24,48.",
    )
    parser.add_argument("--block-size-tokens", type=int, default=None)
    parser.add_argument("--baseline-report", action="store_true")
    parser.add_argument(
        "--candidate-program",
        type=Path,
        default=None,
        help="Candidate .py file or run directory to compare in --baseline-report.",
    )
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
        "--seed-program",
        type=Path,
        default=None,
        help="Candidate .py file or saved run directory to use as the evolution seed.",
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
    capacity_sweep_blocks = _parse_capacity_sweep(args.capacity_sweep_blocks)
    if args.baseline_report:
        compare_baselines(
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
        )
        return
    if args.hidden_report:
        hidden_report(
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
            candidate_program=args.candidate_program,
        )
        return
    if args.plot_report:
        paths = write_baseline_plots(
            Path(args.plot_output),
            quick=args.quick or args.workload_preset == "small",
            capacity_blocks=args.capacity_blocks,
            capacity_sweep_blocks=capacity_sweep_blocks,
            block_size_tokens=args.block_size_tokens,
        )
        for path in paths:
            print(path)
        return
    demo_run_evolution(
        iterations=args.iterations,
        config_file=args.config,
        quick=args.quick,
        seed_program=args.seed_program,
        artifact_output=None if args.no_save_artifacts else Path(args.artifact_output),
    )


def _config_from_args(
    *,
    quick: bool,
    capacity_blocks: int | None,
    capacity_sweep_blocks: tuple[int, ...] = (),
    block_size_tokens: int | None,
) -> EvaluatorConfig:
    effective_capacity_sweep = capacity_sweep_blocks
    if not effective_capacity_sweep and capacity_blocks is None:
        effective_capacity_sweep = _DEFAULT_CAPACITY_SWEEP_BLOCKS
    config = EvaluatorConfig(
        request_count=36 if quick else EvaluatorConfig.request_count,
        seeds=(3,) if quick else EvaluatorConfig.seeds,
        capacity_blocks=capacity_blocks or EvaluatorConfig.capacity_blocks,
        capacity_sweep_blocks=effective_capacity_sweep,
        block_size_tokens=block_size_tokens or EvaluatorConfig.block_size_tokens,
    )
    return config


def _parse_capacity_sweep(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    capacities = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not capacities:
        return ()
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("--capacity-sweep-blocks values must be positive")
    return capacities


def _evaluate_baselines(
    config: EvaluatorConfig,
    *,
    include_reporting: bool = False,
) -> dict[str, EvaluationResult]:
    results: dict[str, EvaluationResult] = {}
    baselines = REPORTING_BASELINES if include_reporting else BASELINES
    for name, factory in baselines.items():
        evaluator = PrefixKVCacheEvaluator(
            config,
            expose_future_reuse=_requires_future_reuse(name),
        )
        results[name] = evaluator(factory)
    return results


def write_baseline_comparison_report(
    path: Path,
    results: dict[str, EvaluationResult],
    *,
    candidate_path: Path,
    command: str,
    quick: bool,
    config: EvaluatorConfig,
) -> Path:
    """Write a Markdown comparison of the candidate and reporting baselines."""

    ranked = sorted(
        results.items(), key=lambda item: item[1].combined_score, reverse=True
    )
    lines = [
        "# Prefix KV-Cache Best Program Baseline Comparison",
        "",
        f"Candidate: `{candidate_path}`",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    if quick:
        lines.extend([f"> **{_QUICK_REPORT_WARNING}**", ""])
    lines.extend(
        [
            "## Headline",
            "",
            (
                "Smoke-only output; run the full panel before comparing policy rank."
                if quick
                else _baseline_report_headline(ranked)
            ),
            "",
            (
                "| Rank | Policy | Group | Combined score | Capacity 24 token hit | "
                "Capacity 48 token hit | Agentic token hit | Churn per 1k |"
            ),
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, (name, result) in enumerate(ranked, start=1):
        cap24 = result.capacity_metrics.get("capacity_24", {})
        cap48 = result.capacity_metrics.get("capacity_48", {})
        agentic = result.workload_metrics["validation/agent_trace_branching"]
        validation = result.split_metrics["validation"]
        lines.append(
            f"| {rank} | `{name}` | {_baseline_group(name)} | "
            f"{result.combined_score:.3f} | "
            f"{float(cap24.get('token_hit_rate', 0.0)):.3f} | "
            f"{float(cap48.get('token_hit_rate', 0.0)):.3f} | "
            f"{float(agentic['token_hit_rate']):.3f} | "
            f"{float(validation['cache_churn_per_1k']):.1f} |"
        )

    validation_workloads = _validation_workloads(results)
    detail_header = "| Policy | " + " | ".join(
        f"{workload.split('/', 1)[1]} token hit" for workload in validation_workloads
    )
    detail_header += " | Validation block hit | Validation churn per 1k |"
    detail_separator = "|---|" + "---:|" * (len(validation_workloads) + 2)
    lines.extend(
        ["", "## Validation Workload Detail", "", detail_header, detail_separator]
    )
    for name, result in ranked:
        validation = result.split_metrics["validation"]
        workload_cells = "".join(
            f" {float(result.workload_metrics[workload]['token_hit_rate']):.3f} |"
            for workload in validation_workloads
        )
        lines.append(
            f"| `{name}` |{workload_cells} "
            f"{float(validation['block_hit_rate']):.3f} | "
            f"{float(validation['cache_churn_per_1k']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "- Candidate `scoring_fn_complexity` in this report is "
                f"`{results['candidate'].candidate_metadata.get('scoring_fn_complexity')}`; "
                "the combined score includes that penalty."
            ),
            (
                "- `future_reuse_heuristic` and `oracle_future_reuse` use "
                "simulator-provided future knowledge and are not deployable. The "
                "former is count-weighted; the latter is a Belady-style next-use "
                "oracle constrained by the simulator's leaf-only eviction model."
            ),
            (
                "- `tinylfu_lru` admits only shallow or repeated blocks, so it "
                "often trades lower hit rate for lower churn."
            ),
            (
                "- `prefix_anchor` is a deployable structural anchor baseline; "
                "`prefix_fanout` is a simpler descendant-count protection baseline."
            ),
            (
                f"- This report uses `request_count={config.request_count}`, "
                f"seeds `{config.seeds}`, and capacity sweep "
                f"`{config.effective_capacity_blocks()}`."
            ),
        ]
    )
    if quick:
        lines.append(
            "- This is a smoke-only single-seed report, not a policy-ranking report."
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _baseline_report_headline(ranked: list[tuple[str, EvaluationResult]]) -> str:
    names = [name for name, _ in ranked]
    if "candidate" not in names:
        return "Reporting baselines ranked by combined score."
    scores = {name: result.combined_score for name, result in ranked}
    candidate_score = scores["candidate"]
    deployable_scores = [
        score
        for name, score in scores.items()
        if name != "candidate" and _baseline_group(name) == "deployable"
    ]
    oracle_scores = [
        score for name, score in scores.items() if _baseline_group(name) != "deployable"
    ]
    clears_deployable = not deployable_scores or candidate_score > max(
        deployable_scores
    )
    below_oracles = not oracle_scores or candidate_score < max(oracle_scores)
    if clears_deployable and below_oracles:
        return (
            "The candidate clears the deployable credibility baselines in this "
            "capacity sweep and remains below the reporting-only future-knowledge oracles."
        )
    return "The candidate ranking is shown against deployable and reporting-only baselines."


def _baseline_group(name: str) -> str:
    if name in {"future_reuse_heuristic", "oracle_future_reuse"}:
        return "oracle/reporting-only"
    return "deployable"


def _artifact_report_config() -> EvaluatorConfig:
    return EvaluatorConfig(capacity_sweep_blocks=(24, 48))


def _baseline_report_command(
    *,
    quick: bool,
    capacity_sweep_blocks: tuple[int, ...],
    candidate_program: Path,
) -> str:
    parts = [
        ".venv/bin/python -m randomize_evolve.problems.prefix_kv_cache.runner",
        "--baseline-report",
    ]
    if quick:
        parts.append("--quick")
    if capacity_sweep_blocks:
        parts.append(
            "--capacity-sweep-blocks "
            + ",".join(str(value) for value in capacity_sweep_blocks)
        )
    parts.append(f"--candidate-program {candidate_program}")
    return " ".join(parts)


def _requires_future_reuse(name: str) -> bool:
    return name in {"future_reuse_heuristic", "oracle_future_reuse"}


def _evaluate_candidate_program(
    config: EvaluatorConfig,
    candidate_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
) -> EvaluationResult:
    source = candidate_path.read_text(encoding="utf-8")
    return run_with_timeout(
        _evaluate_candidate_program_in_worker,
        config,
        candidate_path,
        splits,
        scoring_fn_complexity(source),
        timeout_seconds=config.timeout_s,
    )


def _evaluate_candidate_program_in_worker(
    config: EvaluatorConfig,
    candidate_path: Path,
    splits: tuple[str, ...],
    complexity: int,
) -> EvaluationResult:
    candidate_factory = load_candidate_factory(str(candidate_path))
    return PrefixKVCacheEvaluator(config, splits=splits)(
        candidate_factory,
        scoring_fn_complexity=complexity,
    )


def _resolve_candidate_program(path: Path) -> Path:
    if path.is_dir():
        path = path / "best_program.py"
    if not path.exists():
        raise FileNotFoundError(f"candidate program {path} does not exist")
    return path


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
    lines.append(
        _text(24, 30, "Validation token vs block hit rate", size=20, weight="700")
    )
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
        lines.append(
            _text(left + plot_w + 24, top + 24 + index * 24, name, size=12, fill=color)
        )
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
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
    )
    return colors[index % len(colors)]


if __name__ == "__main__":
    main()
