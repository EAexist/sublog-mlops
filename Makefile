.PHONY: install airflow-up airflow-down airflow-seed benchmark-run cli-run test lint typecheck submodule-init

install:
	uv sync

airflow-up:
	docker compose up -d

airflow-down:
	docker compose down

airflow-seed:
	python -c "from dags.utils.seed_connections import seed_from_env; seed_from_env()" || true
	@echo "Load .env into Airflow Connections manually if seed script not yet implemented."

benchmark-run:
	@echo "Trigger benchmark_dag via Airflow UI or API (e.g. airflow dags trigger benchmark_dag)."
	airflow dags trigger benchmark_dag 2>/dev/null || echo "Run from Airflow env or use UI: http://localhost:8080"

cli-run:
	uv run python cli.py run

test:
	uv run pytest tests/ -v -m "not integration" $(MARKER_FILTER) $(ARGS)

test-only:
	uv run pytest -v -k "$(NAME)" --log-cli-level=INFO

test-int:
	uv run pytest tests/ -v -k "test_generator_real" -m integration --log-cli-level=INFO

lint:
	uv run ruff check . && uv run ruff format --check .

lint-fix:
	uv run ruff check . --fix && uv run ruff format .

typecheck:
	uv run mypy llm_benchmark dags cli.py --ignore-missing-imports

submodule-init:
	git submodule update --init --recursive

test-gen:
	docker compose run --rm airflow-worker airflow tasks test benchmark_dag generate_dataset 2026-03-01

local-gen:
	@uv run python -c "\
from dags.utils.logging import setup_litellm_logging; \
from dags.benchmark_dag import _generate_and_push_dataset; \
setup_litellm_logging(); \
_generate_and_push_dataset( \
    run_id='local_$(shell date +%Y%m%d_%H%M%S)', \
    ti=type('MockTI', (), {'xcom_push': lambda self, k, v: print(f'Saved {k}={v}')})() \
)"