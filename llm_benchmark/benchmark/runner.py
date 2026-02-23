# Async fan-out: per-task benchmarks (all samples × all models per task), then aggregate

import logging
from llm_benchmark.dataset.schema import Dataset
from llm_benchmark.models.registry import get_client
from llm_benchmark.config_loader import load_models_config

logger = logging.getLogger(__name__)


async def run_benchmarks(
    dataset: Dataset,
    model_ids: list[str] | None = None,
    task_type: str = "task_a",
) -> dict[str, list[dict]]:
    """
    Run one dataset against all models (or given model_ids).
    task_type: "task_a" (categorize) or "task_b" (extract); used for sample.get_prompt(task_type).
    Returns dict[model_id, list of {content, prompt_tokens, completion_tokens, latency_ms}].
    """
    config = load_models_config()
    model_ids = model_ids or [m.id for m in config.models]
    results: dict[str, list[dict]] = {}
    for entry in config.models:
        if entry.id not in model_ids:
            continue
        client = get_client(entry.provider, entry.model_string)
        raw_list: list[dict] = []
        for sample in dataset.samples:
            prompt = sample.get_prompt(task_type)
            resp = await client.complete(prompt)
            raw_list.append({
                "content": resp.content,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "latency_ms": resp.latency_ms,
            })
        results[entry.id] = raw_list
    return results


async def run_benchmarks_multi_task(
    datasets: dict[str, Dataset],
    model_ids: list[str] | None = None,
    task_types: dict[str, str] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """
    Benchmark each task independently: for each task_id and its dataset, run all models.
    task_types: optional dict task_id -> "task_a"|"task_b"; default task_a for all.
    Returns dict[task_id, dict[model_id, list of raw sample results]].
    """
    task_types = task_types or {}
    out: dict[str, dict[str, list[dict]]] = {}
    for task_id, dataset in datasets.items():
        task_type = task_types.get(task_id, "task_a")
        logger.info("Running benchmarks for task=%s task_type=%s samples=%s", task_id, task_type, len(dataset.samples))
        out[task_id] = await run_benchmarks(dataset, model_ids=model_ids, task_type=task_type)
    return out
