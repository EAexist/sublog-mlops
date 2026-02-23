---
name: airflow-first-time-setup
description: First-time setup for the Airflow benchmark stack. Use when setting up Airflow locally, onboarding, or when the user asks how to run the benchmark stack.
---

# Airflow First-Time Setup

```bash
cp .env.example .env        # fill in API keys
make install                # uv sync
make airflow-up             # wait ~30s
make airflow-seed           # loads .env → Airflow Connections + Variables
# open http://localhost:8080  (admin / admin)
# enable benchmark_dag → trigger manually
```

If containers are unhealthy, run `docker compose logs scheduler` first.