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
	uv run pytest tests/ -v

lint:
	uv run ruff check . && uv run ruff format --check .

lint-fix:
	uv run ruff check . --fix && uv run ruff format .

typecheck:
	uv run mypy llm_benchmark dags cli.py --ignore-missing-imports

submodule-init:
	git submodule update --init --recursive
