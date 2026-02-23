# Pipeline steps for multi-task benchmark; each step reads/writes under run_dir (XCom = path only).

import json
import logging
from pathlib import Path
from typing import Any, cast

from llm_benchmark.config_loader import load_benchmark_config, load_models_config
from llm_benchmark.dataset.generator import generate_dataset # Changed import
from llm_benchmark.dataset.loader import save_dataset, load_dataset
from llm_benchmark.benchmark.runner import run_benchmarks_multi_task
from llm_benchmark.benchmark.metrics import compute_all_task_metrics
from llm_benchmark.benchmark.scorer import build_benchmark_result
from llm_benchmark.reporting.reporter import generate_report

logger = logging.getLogger(__name__)


def step_generate_datasets(run_id: str, output_dir: str | Path) -> str:
    """Generate one dataset per task, save to run_dir/datasets/<task_id>.json. Returns run_dir path."""
    config = load_benchmark_config()
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = run_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    # Assuming there's only one task as per user's statement "I currently only create one dataset"
    if not config.tasks:
        raise ValueError("No tasks defined in benchmark configuration.")
    if len(config.tasks) > 1:
        logger.warning("Multiple tasks found in config, but only generating for the first one as per current logic.")

    task = config.tasks[0] # Get the first (and assumed only) task

    # Call generate_dataset directly
    # DEFAULT_N_TEMPLATES is 2 from generator.py
    DEFAULT_N_TEMPLATES = 2
    dataset = generate_dataset(
        oracle_model_id=config.oracle_model_id,
        n_templates=DEFAULT_N_TEMPLATES,
        n_samples_per_template=task.n_samples,
        generation_config={"temperature": 0.7, "json_mode": True} # Pass generation_config
    )

    task_id = task.task_id
    save_dataset(dataset, datasets_dir / f"{task_id}.json")

    manifest = {"task_ids": [task_id]} # Manifest will only contain this single task_id
    (datasets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Generated dataset for task %s under %s", task_id, run_dir)
    return str(run_dir)


def step_run_benchmarks(run_dir: str) -> str:
    """Load datasets from run_dir, run benchmarks per task, save raw to run_dir/raw/<task_id>.json."""
    import asyncio
    run_dir = Path(run_dir)
    manifest_path = run_dir / "datasets" / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    task_ids: list[str] = [t for t in (manifest.get("task_ids") or []) if isinstance(t, str)]
    datasets = {tid: load_dataset(run_dir / "datasets" / f"{tid}.json") for tid in task_ids}
    config = load_benchmark_config()
    task_types = {t.task_id: t.task_type for t in config.tasks}
    raw_per_task = asyncio.run(run_benchmarks_multi_task(datasets, task_types=task_types))
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    for task_id, raw_by_model in raw_per_task.items():
        (raw_dir / f"{task_id}.json").write_text(
            json.dumps(raw_by_model, indent=2), encoding="utf-8"
        )
    return str(run_dir)


def step_compute_metrics(run_dir: str) -> str:
    """Load datasets and raw from run_dir, compute per_task_metrics, save to run_dir/per_task_metrics.json."""
    run_dir = Path(run_dir)
    manifest = cast(dict[str, Any], json.loads((run_dir / "datasets" / "manifest.json").read_text(encoding="utf-8")))
    task_ids = [t for t in (manifest.get("task_ids") or []) if isinstance(t, str)]
    datasets = {tid: load_dataset(run_dir / "datasets" / f"{tid}.json") for tid in task_ids}
    raw_per_task: dict[str, Any] = {
        tid: json.loads((run_dir / "raw" / f"{tid}.json").read_text(encoding="utf-8"))
        for tid in task_ids
    }
    models_config = load_models_config()
    config = load_benchmark_config()
    task_types = {t.task_id: t.task_type for t in config.tasks}
    per_task_metrics = compute_all_task_metrics(
        datasets,
        cast(dict[str, dict[str, list[dict]]], raw_per_task),
        models_config.models,
        task_types=task_types,
    )
    (run_dir / "per_task_metrics.json").write_text(
        json.dumps(per_task_metrics, indent=2), encoding="utf-8"
    )
    return str(run_dir)


def step_score_and_rank(run_dir: str) -> str:
    """Load per_task_metrics and config, build result (no weighting), save to run_dir/result.json."""
    run_dir = Path(run_dir)
    per_task_metrics = cast(
        dict[str, dict[str, dict[str, float]]],
        json.loads((run_dir / "per_task_metrics.json").read_text(encoding="utf-8")),
    )
    config = load_benchmark_config()
    # task_weights = {t.task_id: t.weight for t in config.tasks} # Deprecated
    result = build_benchmark_result(per_task_metrics) # Removed task_weights
    (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return str(run_dir)


def step_generate_report(run_dir: str) -> str:
    """Load result from run_dir, generate report.md and results.json in run_dir."""
    from llm_benchmark.benchmark.scorer import BenchmarkResult
    run_dir = Path(run_dir)
    result = BenchmarkResult.model_validate_json(
        (run_dir / "result.json").read_text(encoding="utf-8")
    )
    generate_report(result, run_dir)
    return str(run_dir)
