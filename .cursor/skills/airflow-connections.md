---
name: airflow-connections
description: Airflow Connection IDs and usage for API keys. Use when wiring LLM clients to Airflow or seeding connections.
---

# Airflow Connections Reference

| Connection ID    | Type | Used by             |
|------------------|------|---------------------|
| `openai_default` | HTTP | openai_client.py    |
| `google_default` | HTTP | gemini_client.py    |

Seed via `make airflow-seed` (reads from `.env`). Never hardcode keys in DAG files.