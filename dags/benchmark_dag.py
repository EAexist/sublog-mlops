# Main DAG: multi-task pipeline (generate per task -> run benchmarks -> compute metrics -> score -> report)
# generate_dataset >> run_benchmarks >> compute_metrics >> score_and_rank >> generate_report
# Schedule: @weekly. run_benchmarks: retries=3, retry_delay=2min

from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from llm_benchmark.pipeline import (
    step_generate_datasets,
    step_run_benchmarks,
    step_compute_metrics,
    step_score_and_rank,
    step_generate_report,
)
from llm_benchmark.config_loader import load_benchmark_config
from dags.utils.dag_helpers import push_path, pull_path


def _generate_dataset(**context: Any) -> None:
    run_id = context["run_id"]
    config = load_benchmark_config()
    run_dir = step_generate_datasets(run_id, config.output_dir)
    push_path(context, "run_dir", run_dir)


def _run_benchmarks(**context: Any) -> None:
    run_dir = pull_path(context, "generate_dataset", "run_dir")
    if not run_dir:
        raise ValueError("run_dir not found from generate_dataset")
    step_run_benchmarks(run_dir)
    push_path(context, "run_dir", run_dir)


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
        python_callable=_generate_dataset,
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

    generate_dataset >> run_benchmarks >> compute_metrics >> score_and_rank >> generate_report
