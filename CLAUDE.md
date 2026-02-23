# CLAUDE.md

## What This Is
MLOps portfolio project. Multi-task benchmark: each task (= prompt parametrized with multiple data) is benchmarked independently (correctness, cost per data). Synthesizes eval datasets via a fixed oracle LLM. Orchestrated by Apache Airflow. Per-task diagnostics for portfolio and MLOps visibility.

---

## Structure
```
dags/               # Airflow DAGs only — no business logic here
llm_benchmark/      # core library — zero Airflow imports
  dataset/          # generator.py, loader.py, schema.py, validator.py, publisher.py
  models/           # base.py, registry.py, *_client.py
  benchmark/        # runner.py, metrics.py, scorer.py
  reporting/        # reporter.py, relay.py (stub)
config/             # models.yaml, benchmark.yaml
datasets/           # git submodule — shared eval dataset repo (see Dataset Submodule)
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
generate_dataset >> publish_dataset >> run_benchmarks >> compute_metrics >> score_and_rank >> generate_report
```
Schedule: `@weekly`. `run_benchmarks` has `retries=3, retry_delay=2min`. `publish_dataset` has `retries=2, retry_delay=1min`.

### Step Responsibilities

**`generate_dataset`** — generate + validate (extended from original)
Generates samples via oracle LLM, then immediately validates before any data leaves this step.
Validation failure aborts the run here — no malformed data ever reaches the submodule or benchmarks.
- `dataset/generator.py` — call oracle LLM, produce raw samples
- `dataset/validator.py` *(new)* — schema validation (via `PromptVersion` in `schema.py`) + content checks (non-empty, expected fields, sample count vs `n_samples`)
- XComs out: path to validated dataset file (local temp path)

**`publish_dataset`** — version + archive + push *(new standalone step)*
Owns all interaction with the `datasets/` git submodule. Receives the validated dataset path via XCom and is solely responsible for making it available to the team.
- `dataset/publisher.py` *(new)* — all logic below; no git CLI calls outside this module
- **Versioned write**: copies dataset into `datasets/versions/YYYY-MM-DD_<dag_run_id>/`
- **Latest pointer**: overwrites `datasets/latest/` symlink (or `latest.json` pointer file if symlinks are fragile on your OS) to point at the new version directory
- **Retention/archive**: enforces `max_versions` from `config/benchmark.yaml`; oldest versions beyond the limit are moved to `datasets/archive/` and noted in `datasets/archive/index.json` — they are never deleted, only demoted from `versions/`
- **Git commit + push**: stages all submodule changes, commits with message `"dataset: <task_id> <version>"`, and pushes to the submodule remote — making the dataset available to any repo that includes this submodule
- XComs out: versioned dataset path inside submodule (string)

---

## Dataset Submodule

### What this project is responsible for
| Concern | Owner |
|---|---|
| Validation logic and schema | This project (`llm_benchmark/dataset/`) |
| Versioned folder structure convention | This project (`publisher.py`) |
| `latest` pointer maintenance | This project (`publisher.py`) |
| Retention / archive policy | This project (config-driven via `benchmark.yaml`) |
| `git commit + push` inside submodule | This project (`publisher.py`) |
| Submodule remote repo hosting | External (GitHub/GitLab — set up once by team) |
| Consuming the submodule in other repos | External (each repo runs `git submodule update`) |

### Submodule layout (inside `datasets/`)
```
datasets/
  versions/
    2025-01-20_run_abc123/     # one folder per published run
      <task_id>.jsonl
      meta.json                # run_id, dag_run_id, n_samples, oracle_model, timestamp
    2025-01-27_run_def456/
      ...
  archive/
    2024-11-04_run_xyz789/     # versions rotated out of active versions/
      ...
    index.json                 # manifest of all archived versions
  latest -> versions/2025-01-27_run_def456/   # symlink (or latest.json pointer)
  README.md
```

### Versioning convention
- Version directory name: `YYYY-MM-DD_<dag_run_id>` — sortable, unique, traceable to Airflow run
- `latest` always reflects the most recently successfully published run
- `max_versions` (default: `10`) controls how many entries stay in `versions/` before the oldest is moved to `archive/`
- Nothing is ever hard-deleted from the submodule

### Config additions (`config/benchmark.yaml`)
```yaml
dataset_publishing:
  max_versions: 10          # versions kept in versions/ before archiving
  submodule_path: datasets  # relative path to submodule root
  remote_branch: main       # branch to push to in submodule remote
```

---

## Config
- Add a model: one block in `config/models.yaml`, no code changes.
- Tasks (prompts) in `config/benchmark.yaml`: list of `tasks` (task_id, n_samples).
- Dataset publishing settings under `dataset_publishing` key in `config/benchmark.yaml` (see above).

---

## Key Commands
```bash
make airflow-up          # start stack
make airflow-seed        # load connections from .env
make benchmark-run       # trigger DAG
make cli-run             # run pipeline locally (no Docker)
make submodule-init      # git submodule update --init --recursive
make test && make lint
```

---

## Stubs (not implemented — don't refactor away)
- `reporting/relay.py` → `push_to_client_repo(result)` raises `NotImplementedError`
- `models/base.py` → `FineTunableModel` mixin with `fine_tune(dataset)`
- `dataset/schema.py` → `PromptVersion` schema already defined

---

## Stack
Airflow 2.x · Python 3.12 · uv · pydantic v2 · httpx/asyncio · mlflow · typer · ruff · mypy · pytest · gitpython (submodule ops)
