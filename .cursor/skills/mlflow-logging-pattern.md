---
name: mlflow-logging-pattern
description: Log benchmark runs to MLflow (params, metrics, artifacts). Use when implementing or modifying the generate_report task or MLflow logging.
---

# MLflow Logging Pattern

```python
import mlflow

with mlflow.start_run(run_name=run_id):
    mlflow.log_params({"oracle_model": ..., "n_samples": ..., "dataset_hash": ...})
    mlflow.log_metrics({"gpt4o_correctness": ..., "gpt4o_latency_p50": ..., "gpt4o_cost": ...})
    mlflow.log_artifact("outputs/runs/<run_id>/report.md")
```

One MLflow run = one DAG run. Log inside the `generate_report` task after all metrics are final.