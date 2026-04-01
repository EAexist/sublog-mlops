# Markdown + JSON: per-task accuracy and latency (N/A until latency module is used). No composite; human chooses model.

import logging
from pathlib import Path

from llm_benchmark.benchmark.scorer import BenchmarkResult

logger = logging.getLogger(__name__)


def _format_per_task_section(result: BenchmarkResult) -> str:
    """Format per-task metrics: Model | Accuracy | Latency | Cost. Latency is N/A unless latency module is used."""
    lines = []
    for task_id, by_model in result.per_task_metrics.items():
        w = result.task_weights.get(task_id, 1.0)
        lines.append(f"### Task: {task_id} (weight={w})")
        lines.append("| Model | Accuracy | Latency (ms) | Cost |")
        lines.append("|-------|----------|---------------|------|")
        for model_id, m in sorted(by_model.items()):
            lat = m.get("latency_p50")
            lat_str = f"{lat:.0f}" if lat is not None and lat != 0.0 else "N/A"
            lines.append(
                f"| {model_id} | {m.get('correctness', 0):.3f} | "
                f"{lat_str} | ${m.get('cost', 0):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def generate_report(result: BenchmarkResult, output_dir: Path) -> tuple[Path, Path]:
    """
    Write report.md and results.json. No best-model or composite score.
    Developer chooses model from per-task accuracy (and latency when enabled).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "report.md"
    json_path = output_dir / "results.json"

    md_body = [
        "# Benchmark Report",
        "",
        "No composite score; choose a model from the tables below.",
        "",
        "## Per-task metrics",
        "",
        _format_per_task_section(result),
        "",
        "_Latency: N/A until `llm_benchmark.benchmark.latency` is integrated._",
    ]

    md_path.write_text("\n".join(md_body), encoding="utf-8")
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Report written to %s and %s", md_path, json_path)
    return md_path, json_path
