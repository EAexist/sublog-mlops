---
name: fine-tuning
description: Implement fine-tuning workflow (Feature 7). Use when implementing FineTunableModel or adding a fine-tune DAG.
---

# Implement Feature 7 (Fine-Tuning)

**Stub location:** `models/base.py → FineTunableModel.fine_tune(dataset: Dataset) -> str`

1. Implement in `openai_client.py` first (OpenAI fine-tune API).
2. Create a separate `dags/fine_tune_dag.py` — do not add to `benchmark_dag.py`.
3. Fine-tune DAG shape: `prepare_dataset >> submit_job >> poll_until_complete >> register_model`
4. `poll_until_complete` should be an Airflow sensor with `poke_interval=300`.
5. On completion, add the returned model ID to `models.yaml` and trigger a new benchmark run.