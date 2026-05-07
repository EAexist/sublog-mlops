# Pipeline steps for multi-task benchmark; each step reads/writes under run_dir (XCom = path only).

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
from datasets_shared.schema import Dataset, Sample, SubscriptionEventType
from datasets_shared.schema.models import EmailTemplate
from pydantic import BaseModel

from llm_benchmark.benchmark.loader import download_langfuse_data_to_csv
from llm_benchmark.benchmark.runner import BenchmarkRunner
from llm_benchmark.benchmark.task.task_factory import task_factory
from llm_benchmark.config_loader import load_dataset_config, load_models_config
from llm_benchmark.dataset.constants import COMPANIES
from llm_benchmark.dataset.generator.generator import _call_oracle, assemble_dataset
from llm_benchmark.dataset.generator.parameter_generator import generate_parameters
from llm_benchmark.dataset.generator.template_generator import generate_templates
from llm_benchmark.dataset.loader import (
    dataset_loader,
)
from llm_benchmark.dataset.patch_utils import patch_template_ids_with_uuid
from llm_benchmark.dataset.publisher import DatasetPublisher
from llm_benchmark.reporting.visualizer import benchmark_visualizer
from llm_benchmark.utils.loader import read_jsonl, save_jsonl, select_test_samples

logger = logging.getLogger(__name__)


config = load_dataset_config()


def load_dataset(file_path: Path) -> Dataset:
    """Load a Dataset from a JSON file."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        # If it's a list of samples, create Dataset object
        samples = [Sample.model_validate(item) for item in data]
        return Dataset(samples=samples)
    else:
        # If it's already a Dataset dict
        return Dataset.model_validate(data)


async def step_generate_datasets(run_id: str, output_dir: Path) -> bool:
    """Generate a single dataset, save to run_dir/datasets/dataset.json. Returns run_dir path."""
    output_path = output_dir / "data"

    do_update_templates: bool = config.do_update_templates
    do_update_parameters: bool = config.do_update_parameters

    templates_publisher = DatasetPublisher(output_path / "templates", "templates")
    parameters_publisher = DatasetPublisher(output_path / "parameters", "parameters")
    dataset_publisher = DatasetPublisher(output_path / "emails", "samples")

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

    # Load existing data first (if available)
    templates = None
    parameters = None

    if not do_update_templates:
        logger.info("Template config unchanged. Loading latest...")
        try:
            templates = dataset_loader.load_latest_templates()
            templates = patch_template_ids_with_uuid(templates)
            template_path = templates_publisher.publish(templates, run_id)
        except Exception as e:
            logger.error(f"Failed to load existing templates: {e}")
            if do_update_parameters:
                logger.error(
                    "Cannot proceed with parameter generation due to template loading failure."
                )
                raise
            # If we need to update templates, we'll generate them below

    if not do_update_parameters:
        logger.info("Param config unchanged. Loading latest...")
        try:
            parameters = dataset_loader.load_latest_parameters()
        except Exception as e:
            logger.error(f"Failed to load existing parameters: {e}")
            if do_update_templates:
                logger.error(
                    "Cannot proceed with template generation due to parameter loading failure."
                )
                raise
            # If we need to update parameters, we'll generate them below

    # Generate new data if needed (only after loading existing data succeeded or if update is required)
    if do_update_templates:
        logger.info("Template config changed. Generating new templates...")
        templates = await generate_templates(
            n_templates=config.n_templates_per_event,
            oracle_fn=throttled_call,
        )
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

    dataset = assemble_dataset(templates, parameters, config.n_samples_per_template)
    # Important: must save List[Sample] not Dataset as jsonl.
    dataset_path = dataset_publisher.publish(dataset.samples, run_id)

    # manifest_path = dataset_publisher.save_manifest(
    #     output_path, template_path, parameters_path, dataset_path, run_id
    # )
    # logger.info("Generated dataset under %s", manifest_path)
    return True


async def step_run_benchmarks(run_dir: Path, use_dataset_cache: bool = False) -> str:
    """Load dataset using HuggingFaceDatasetLoader, run benchmarks, save raw to run_dir/raw/<task_id>.json."""

    path = Path(run_dir) / "latest_samples.jsonl"
    templates_path = Path(run_dir) / "latest_templates.jsonl"

    if not use_dataset_cache:
        samples = dataset_loader.load_latest_samples()
        samples = select_test_samples(samples)
        templates = dataset_loader.load_latest_templates()
        save_jsonl(samples, path)
        save_jsonl(templates, templates_path)

    samples = read_jsonl(path, Sample)
    templates = read_jsonl(templates_path, EmailTemplate)

    runner = BenchmarkRunner()

    experiment_id = f"experiment-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Running benchmark %s on %d samples...", experiment_id, len(samples))

    observation_ids = await runner.run_benchmarks(
        samples=samples,
        experiment_id=experiment_id,
        templates=templates,
    )

    from llm_benchmark.benchmark.loader import wait_langfuse_sync

    n_models = len(load_models_config().models)
    task_ids = task_factory.list_available_tasks()

    wait_langfuse_sync(
        experiment_id=experiment_id,
        expected_n_traces=n_models * len(task_ids),
        expected_n_generations=len(observation_ids),
    )

    download_langfuse_data_to_csv(experiment_id=experiment_id)

    return "success"


def step_generate_report(experiment_id: str, run_dir: str | Path) -> str:
    """Load result from langfuse, generate visualization."""

    csv_path = f"data/benchmark/{experiment_id}/result.csv"
    df = pd.read_csv(csv_path)

    # Calculate weighted average scores per model
    model_summary = (
        df.groupby(["model", "task_name"])
        .agg(
            score_accuracy=("score_accuracy", "sum"),
            score_specificity=("score_specificity", "sum"),
            n_data=("n_data", "sum"),
            cost_total=("cost_total", "mean"),
        )
        .reset_index()
    )
    model_summary["score_accuracy"] = model_summary["score_accuracy"] / model_summary["n_data"]
    model_summary["score_specificity"] = (
        model_summary["score_specificity"] / model_summary["n_data"]
    )
    # Convert cost to USD per thousand requests
    model_summary["cost_total"] = model_summary["cost_total"] * 1000

    benchmark_visualizer.plot_email_categorization_performance(
        df=model_summary[model_summary["task_name"] == "email_categorization"],
        output_path=f"data/benchmark/{experiment_id}/model_performance_email_categorization.png",
    )

    benchmark_visualizer.plot_email_template_extraction_performance(
        df=model_summary[model_summary["task_name"] == "email_template_extraction"],
        output_path=f"data/benchmark/{experiment_id}/model_performance_email_template_extraction.png",
        # anntation_pos_dict={
        #     "gemini-2.5-flash-lite": "left",
        # },
    )

    return str(csv_path)


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
    # task_types = {t.task_id: t.task_type for t in config}
    # per_task_metrics = compute_all_task_metrics(
    #     datasets,
    #     cast(dict[str, dict[str, list[dict]]], raw_per_task),
    #     models_config.models,
    #     task_types=task_types,
    # )
    # (run_path / "per_task_metrics.json").write_text(
    #     json.dumps(per_task_metrics, indent=2), encoding="utf-8"
    # )
    return str(run_path)


def step_score_and_rank(run_dir: str | Path) -> str:
    """Load per_task_metrics and config, build result (no weighting), save to run_dir/result.json."""
    run_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
    per_task_metrics = cast(
        dict[str, Any], json.loads((run_path / "per_task_metrics.json").read_text(encoding="utf-8"))
    )
    config = load_dataset_config()
    # (run_path / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return str(run_path)
