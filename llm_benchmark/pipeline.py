# Pipeline steps for multi-task benchmark; each step reads/writes under run_dir (XCom = path only).

import json
import logging
from pathlib import Path
from typing import Any, TypeVar, cast

from datasets_shared.schema import SubscriptionEventType
from pydantic import BaseModel

from llm_benchmark.benchmark.metrics import compute_all_task_metrics
from llm_benchmark.benchmark.runner import run_benchmarks_multi_task
from llm_benchmark.benchmark.scorer import build_benchmark_result
from llm_benchmark.config_loader import load_dataset_config, load_models_config
from llm_benchmark.dataset.constants import COMPANIES
from llm_benchmark.dataset.generator.generator import _call_oracle, assemble_dataset
from llm_benchmark.dataset.generator.parameter_generator import generate_parameters
from llm_benchmark.dataset.generator.template_generator import generate_templates
from llm_benchmark.dataset.loader import (
    HuggingFaceDatasetLoader,
)
from llm_benchmark.dataset.patch_utils import patch_template_ids_with_uuid
from llm_benchmark.dataset.publisher import DatasetPublisher
from llm_benchmark.reporting.reporter import generate_report

logger = logging.getLogger(__name__)


async def step_generate_datasets(run_id: str, output_dir: Path) -> bool:
    """Generate a single dataset, save to run_dir/datasets/dataset.json. Returns run_dir path."""
    config = load_dataset_config()
    output_path = output_dir / "data"

    do_update_templates: bool = config.do_update_templates
    do_update_parameters: bool = config.do_update_parameters

    templates_publisher = DatasetPublisher(output_path / "templates", "templates")
    parameters_publisher = DatasetPublisher(output_path / "parameters", "parameters")
    dataset_publisher = DatasetPublisher(output_path / "emails", "samples")
    dataset_loader = HuggingFaceDatasetLoader(config.hf_repo)

    if not do_update_templates and not do_update_parameters:
        logger.info("No config changes detected. Skipping generation.")
        return False

    import asyncio

    generation_config = {"temperature": 0.7, "json_mode": True}
    sem = asyncio.Semaphore(2)

    T = TypeVar("T", bound=BaseModel)

    async def throttled_call(prompt: str, schema: type[T]) -> T:
        async with sem:
            # 3. Add a small buffer to prevent 'burst' 429s
            await asyncio.sleep(0.5)
            return await _call_oracle(prompt, config.oracle_model_id, schema, generation_config)

    if do_update_templates:
        logger.info("Template config changed. Generating new templates...")
        templates = await generate_templates(
            n_templates=config.n_templates_per_event,
            oracle_fn=throttled_call,
        )
        template_path = templates_publisher.publish(templates, run_id)

    else:
        logger.info("Template config unchanged. Loading latest...")
        templates = dataset_loader.load_latest_templates()
        templates = patch_template_ids_with_uuid(templates)

        template_path = templates_publisher.publish(templates, run_id)

    if do_update_parameters:
        logger.info("Param config changed. Generating new parameters...")
        parameters = generate_parameters(
            count=len(COMPANIES)
            * len(SubscriptionEventType)
            * config.n_templates_per_event
            * config.n_samples_per_template,
            locales=config.locales,
        )
        parameters_path = parameters_publisher.publish(parameters, run_id)

    else:
        logger.info("Param config unchanged. Loading latest...")
        parameters = dataset_loader.load_latest_parameters()

    dataset = assemble_dataset(templates, parameters, config.n_samples_per_template)
    # Important: must save List[Sample] not Dataset as jsonl.
    dataset_path = dataset_publisher.publish(dataset.samples, run_id)

    # manifest_path = dataset_publisher.save_manifest(
    #     output_path, template_path, parameters_path, dataset_path, run_id
    # )
    # logger.info("Generated dataset under %s", manifest_path)
    return True


def step_run_benchmarks(run_dir: str | Path) -> str:
    """Load datasets from run_dir, run benchmarks per task, save raw to run_dir/raw/<task_id>.json."""
    import asyncio

    run_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
    manifest_path = run_path / "datasets" / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    task_ids: list[str] = [t for t in (manifest.get("task_ids") or []) if isinstance(t, str)]
    datasets = {tid: load_dataset(run_path / "datasets" / f"{tid}.json") for tid in task_ids}
    config = load_dataset_config()
    task_types = {t.task_id: t.task_type for t in config.tasks}
    raw_per_task = asyncio.run(run_benchmarks_multi_task(datasets, task_types=task_types))
    raw_dir = run_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    for task_id, raw_by_model in raw_per_task.items():
        (raw_dir / f"{task_id}.json").write_text(
            json.dumps(raw_by_model, indent=2), encoding="utf-8"
        )
    return str(run_dir)


def step_compute_metrics(run_dir: str | Path) -> str:
    """Load datasets and raw from run_dir, compute per_task_metrics, save to run_dir/per_task_metrics.json."""
    run_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
    manifest = cast(
        dict[str, Any],
        json.loads((run_path / "datasets" / "manifest.json").read_text(encoding="utf-8")),
    )
    task_ids = [t for t in (manifest.get("task_ids") or []) if isinstance(t, str)]
    datasets = {tid: load_dataset(run_path / "datasets" / f"{tid}.json") for tid in task_ids}
    raw_per_task: dict[str, Any] = {
        tid: json.loads((run_path / "raw" / f"{tid}.json").read_text(encoding="utf-8"))
        for tid in task_ids
    }
    models_config = load_models_config()
    config = load_dataset_config()
    task_types = {t.task_id: t.task_type for t in config.tasks}
    per_task_metrics = compute_all_task_metrics(
        datasets,
        cast(dict[str, dict[str, list[dict]]], raw_per_task),
        models_config.models,
        task_types=task_types,
    )
    (run_path / "per_task_metrics.json").write_text(
        json.dumps(per_task_metrics, indent=2), encoding="utf-8"
    )
    return str(run_path)


def step_score_and_rank(run_dir: str | Path) -> str:
    """Load per_task_metrics and config, build result (no weighting), save to run_dir/result.json."""
    run_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
    per_task_metrics = cast(
        dict[str, Any], json.loads((run_path / "per_task_metrics.json").read_text(encoding="utf-8"))
    )
    config = load_dataset_config()
    task_weights = {t.task_id: 1.0 for t in config.tasks}  # Default equal weights
    result = build_benchmark_result(per_task_metrics, task_weights)
    (run_path / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return str(run_path)


def step_generate_report(run_dir: str | Path) -> str:
    """Load result from run_dir, generate report.md and results.json in run_dir."""
    from llm_benchmark.benchmark.scorer import BenchmarkResult

    run_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
    result = BenchmarkResult.model_validate_json(
        (run_path / "result.json").read_text(encoding="utf-8")
    )
    generate_report(result, run_path)
    return str(run_path)
