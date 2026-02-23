---
name: xcom-pattern
description: Airflow task data passing via file paths in XCom. Use when writing or reviewing DAG tasks that pass dataset or result paths between tasks.
---

# XCom Pattern (file path convention)

Tasks pass file paths only — never raw data objects.

**Push:**
```python
context["ti"].xcom_push(key="dataset_path", value=str(output_path))
```

**Pull:**
```python
dataset_path = context["ti"].xcom_pull(task_ids="generate_dataset", key="dataset_path")
```

All output files go to `outputs/runs/<run_id>/`. Use `run_id = context["run_id"]` to construct the path.