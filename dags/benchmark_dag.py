# Main DAG: multi-task pipeline (generate per task -> run benchmarks -> compute metrics -> score -> report)
# generate_dataset >> run_benchmarks >> compute_metrics >> score_and_rank >> generate_report
# Schedule: @weekly. run_benchmarks: retries=3, retry_delay=2min

import asyncio
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from huggingface_hub import HfApi
from llm_benchmark.pipeline import (
    step_compute_metrics,
    step_generate_datasets,
    step_generate_report,
    step_run_benchmarks,
    step_score_and_rank,
)

from dags.utils.dag_helpers import pull_path, push_path

logger = logging.getLogger(__name__)


REPO_ID = "hyeon-expression/subscription-killer-synthetic-emails"


def _generate_and_push_dataset(**context: Any) -> str:
    run_id = context["run_id"]
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp_dir:
        was_changed = asyncio.run(step_generate_datasets(run_id, Path(tmp_dir)))

        if was_changed:
            commit_info = api.upload_folder(
                folder_path=tmp_dir,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"data: update dataset for run {run_id}",
            )

            logger.info(f"New dataset pushed to Hugging Face. Commit: {commit_info.oid}")
            return commit_info.oid

        # 3. If no changes, get the latest commit SHA from the repo
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type="dataset")
        if not repo_info.sha:
            raise ValueError(f"Could not retrieve SHA for {REPO_ID}")

        logger.info("No changes detected. Using existing latest SHA.")
        return repo_info.sha


def _run_benchmarks(**context: Any) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        step_run_benchmarks(tmp_dir)
        return ""


def _compute_metrics(**context: Any) -> None:
    run_dir = pull_path(context, "run_benchmarks", "run_dir")
    if not run_dir:
        raise ValueError("run_dir not found from run_benchmarks")
    step_compute_metrics(run_dir)
    push_path(context, "run_dir", run_dir)


def _score_and_rank(**context: Any) -> None:
    run_dir = pull_path(context, "compute_metrics", "run_dir")
    if not run_dir:
        raise ValueError("run_dir not found from compute_metrics")
    step_score_and_rank(run_dir)
    push_path(context, "run_dir", run_dir)


def _generate_report(**context: Any) -> None:
    run_dir = pull_path(context, "score_and_rank", "run_dir")
    if not run_dir:
        raise ValueError("run_dir not found from score_and_rank")
    step_generate_report(run_dir)
    push_path(context, "run_dir", run_dir)


with DAG(
    dag_id="benchmark_dag",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"retries": 0},
    tags=["benchmark"],
) as dag:
    generate_dataset = PythonOperator(
        task_id="generate_dataset",
        python_callable=_generate_and_push_dataset,
    )
    run_benchmarks = PythonOperator(
        task_id="run_benchmarks",
        python_callable=_run_benchmarks,
        retries=3,
        retry_delay=timedelta(minutes=2),
    )
    compute_metrics = PythonOperator(
        task_id="compute_metrics",
        python_callable=_compute_metrics,
    )
    score_and_rank = PythonOperator(
        task_id="score_and_rank",
        python_callable=_score_and_rank,
    )
    generate_report = PythonOperator(
        task_id="generate_report",
        python_callable=_generate_report,
    )

    chain = (
        generate_dataset >> run_benchmarks >> compute_metrics >> score_and_rank >> generate_report
    )
