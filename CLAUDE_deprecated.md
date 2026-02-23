# CLAUDE.md

## What This Is
MLOps portfolio project. Multi-task benchmark: each task (= prompt parametrized with multiple data) is benchmarked independently (correctness, cost per data). Synthesizes eval datasets via a fixed oracle LLM. Orchestrated by Apache Airflow. Per-task diagnostics for portfolio and MLOps visibility.

---

## Structure
```
dags/               # Airflow DAGs only — no business logic here
llm_benchmark/      # core library — zero Airflow imports
  dataset/          # generator.py, loader.py, schema.py
  models/           # base.py, registry.py, *_client.py
  benchmark/        # runner.py, metrics.py, scorer.py
  reporting/        # reporter.py, relay.py (stub)
config/             # models.yaml, benchmark.yaml
tests/unit/
tests/integration/
cli.py              # thin Typer CLI for local dev without Airflow
```

---

## Hard Rules
- `llm_benchmark/` never imports from `dags/`. It is plain Python, testable without Docker.
- DAG tasks are thin wrappers — all logic lives in `llm_benchmark/`.
- XCom values are file paths (strings) only — never full data objects.
- API keys go in Airflow Connections, not env vars in DAG code.
- All public functions: type annotations + docstring.
- Log via `logging`, never `print`.

---

## DAG Shape
```
generate_dataset >> run_benchmarks >> compute_metrics >> score_and_rank >> generate_report
```
Schedule: `@weekly`. `run_benchmarks` has `retries=3, retry_delay=2min`.

---

## Config
- Add a model: one block in `config/models.yaml`, no code changes.
- Tasks (prompts) in `config/benchmark.yaml`: list of `tasks` (task_id, n_samples).

---

## Key Commands
```bash
make airflow-up       # start stack
make airflow-seed     # load connections from .env
make benchmark-run    # trigger DAG
make cli-run          # run pipeline locally (no Docker)
make test && make lint
```

---

## Stubs (not implemented — don't refactor away)
- `reporting/relay.py` → `push_to_client_repo(result)` raises `NotImplementedError`
- `models/base.py` → `FineTunableModel` mixin with `fine_tune(dataset)`
- `dataset/schema.py` → `PromptVersion` schema already defined

---

## Stack
Airflow 2.x · Python 3.12 · uv · pydantic v2 · httpx/asyncio · mlflow · typer · ruff · mypy · pytest
