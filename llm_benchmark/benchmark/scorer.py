# Build BenchmarkResult from per_task_metrics. No weight-sum; human chooses model from report.

import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskMetrics(BaseModel):
    """Per-task per-model metrics (accuracy, cost; latency optional when enabled)."""
    correctness: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    mean_latency_ms: float = 0.0
    cost: float = 0.0


class BenchmarkResult(BaseModel):
    """Result with per-task metrics only. No composite score; developer chooses model from report."""
    best_model_id: str = ""
    scores: dict[str, float] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}
    per_task_metrics: dict[str, dict[str, dict[str, float]]] = {}
    task_weights: dict[str, float] = {}


def build_benchmark_result(
    per_task_metrics: dict[str, dict[str, dict[str, float]]],
    task_weights: dict[str, float],
) -> BenchmarkResult:
    """
    Build result from per_task_metrics. No weighting or ranking.
    Report shows accuracy (and latency when enabled); human picks model.
    """
    all_models = set()
    for by_model in per_task_metrics.values():
        all_models.update(by_model.keys())
    metrics_by_model: dict[str, dict[str, float]] = {}
    for model_id in all_models:
        correctnesses: list[float] = []
        total_cost = 0.0
        for by_model in per_task_metrics.values():
            m = by_model.get(model_id, {})
            if m:
                correctnesses.append(m.get("correctness", 0.0))
                total_cost += m.get("cost", 0.0)
        metrics_by_model[model_id] = {
            "correctness": sum(correctnesses) / len(correctnesses) if correctnesses else 0.0,
            "cost": total_cost,
        }
    return BenchmarkResult(
        best_model_id="",
        scores={},
        metrics_by_model=metrics_by_model,
        per_task_metrics=per_task_metrics,
        task_weights=task_weights,
    )