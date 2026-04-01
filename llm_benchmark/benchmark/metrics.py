# Accuracy (correctness) and cost only. No weighting; no latency in main flow.
# Latency: see llm_benchmark.benchmark.latency (TODO module, import when implementing).

import json
import logging
from typing import Any

from datasets_shared.schema import Dataset, Sample

from llm_benchmark.config_loader import ModelEntry

logger = logging.getLogger(__name__)


def compute_correctness(
    sample: Sample,
    model_output: str,
    task_id: str = "task_a",
    judge_model_id: str | None = None,
) -> float:
    """
    Compare model output to ground truth for the task. Score 0.0–1.0 (accuracy).
    task_id: "task_a" (categorization) or "task_b" (extraction); uses sample.get_expected(task_id).
    Stub: exact match only; extend for fuzzy/LLM-as-judge later.
    """
    expected = sample.get_expected(task_id)
    model_stripped = (model_output or "").strip()
    if not expected:
        return 0.0
    if task_id == "task_b":
        try:
            parsed: Any = json.loads(model_stripped)
            if not isinstance(parsed, dict):
                return 0.0
            subj = parsed.get("subject_regex") == sample.metadata.get("subject_regex", "")
            snip = parsed.get("snippet_regex") == sample.metadata.get("snippet_regex", "")
            return 1.0 if (subj and snip) else 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0
    return 1.0 if model_stripped == expected else 0.0


def compute_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_per_1k: float,
    output_per_1k: float,
) -> float:
    """Cost in dollars."""
    return (prompt_tokens / 1000.0) * input_per_1k + (completion_tokens / 1000.0) * output_per_1k


def compute_task_metrics(
    dataset: Dataset,
    raw_by_model: dict[str, list[dict]],
    model_entries: dict[str, ModelEntry],
    task_type: str = "task_a",
) -> dict[str, dict[str, float]]:
    """
    Compute accuracy (correctness) and cost per model for one task.
    task_type: "task_a" or "task_b" for ground-truth comparison.
    Latency is not computed here; use llm_benchmark.benchmark.latency when needed.
    raw_by_model: model_id -> list of {content, prompt_tokens, completion_tokens, latency_ms} per sample.
    Returns dict[model_id, {correctness, cost}].
    """
    out: dict[str, dict[str, float]] = {}
    samples = dataset.samples
    for model_id, raw_list in raw_by_model.items():
        entry = model_entries.get(model_id)
        if not entry:
            continue
        correctnesses = [
            compute_correctness(samples[i], raw_list[i].get("content", ""), task_id=task_type)
            for i in range(min(len(samples), len(raw_list)))
        ]
        total_cost = sum(
            compute_cost(
                r.get("prompt_tokens", 0),
                r.get("completion_tokens", 0),
                entry.input_price_per_1k_tokens,
                entry.output_price_per_1k_tokens,
            )
            for r in raw_list
        )
        out[model_id] = {
            "correctness": sum(correctnesses) / len(correctnesses) if correctnesses else 0.0,
            "cost": total_cost,
        }
    return out


def compute_all_task_metrics(
    datasets: dict[str, Dataset],
    raw_per_task: dict[str, dict[str, list[dict]]],
    model_entries: list[ModelEntry],
    task_types: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-task per-model metrics (accuracy, cost). task_types: task_id -> task_a|task_b."""
    by_id = {m.id: m for m in model_entries}
    task_types = task_types or {}
    return {
        task_id: compute_task_metrics(
            datasets[task_id], raw_by_model, by_id,
            task_type=task_types.get(task_id, "task_a"),
        )
        for task_id, raw_by_model in raw_per_task.items()
        if task_id in datasets
    }
