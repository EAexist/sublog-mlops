# Makefile Usage Guide

## Setup & Installation

### Dependencies
```bash
make install                    # Install dependencies with uv
make submodule-init            # Initialize git submodules
```

## Development & Testing

### Testing Commands
```bash
make test                      # Run unit tests (default)
make test MARKER_FILTER="-m integration"    # Run only integration tests
make test MARKER_FILTER=""     # Run all tests
make test ARGS="-k test_name"  # Run specific test by name
make test-only NAME="pattern"  # Run specific test pattern
make test-int                  # Legacy integration test command
```

### Code Quality
```bash
make lint                      # Check code formatting and linting
make lint-fix                  # Auto-fix linting issues
make typecheck                 # Run type checking with mypy
```

## Airflow & Benchmarking

### Airflow Management
```bash
make airflow-up                # Start Airflow with Docker Compose
make airflow-down              # Stop Airflow containers
make airflow-seed              # Seed connections from environment
```

### Benchmark Execution
```bash
make benchmark-run             # Trigger benchmark DAG via Airflow
make cli-run                   # Run benchmark via CLI
make test-gen                  # Test dataset generation in Airflow worker
make local-gen                 # Generate dataset locally
```

## Test Structure

- **Unit Tests**: `tests/unit/` - Fast tests, no external dependencies
- **Integration Tests**: `tests/integration/` - Real API calls, slower execution

Integration tests are marked with `@pytest.mark.integration` and can be expensive (API costs).

## Parameters

- `MARKER_FILTER` - Pytest marker filter (e.g., `-m integration`)
- `ARGS` - Additional pytest arguments (e.g., `-k test_name`, `--pdb`)
- `NAME` - Test name pattern for `test-only` command
