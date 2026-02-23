# Unit tests for config_loader

import pytest
from pathlib import Path
from llm_benchmark.config_loader import load_models_config, load_benchmark_config


def test_load_models_config(project_root: Path) -> None:
    path = project_root / "config" / "models.yaml"
    if not path.exists():
        pytest.skip("config/models.yaml not found")
    config = load_models_config(path)
    assert len(config.models) >= 1
    assert config.models[0].id
    assert config.models[0].provider in ("openai", "google", "ollama")


def test_load_benchmark_config(project_root: Path) -> None:
    path = project_root / "config" / "benchmark.yaml"
    if not path.exists():
        pytest.skip("config/benchmark.yaml not found")
    config = load_benchmark_config(path)
    assert config.oracle_model_id
    assert len(config.tasks) >= 1
    assert config.tasks[0].task_id
    assert config.tasks[0].n_samples > 0
    # The 'metric_weights' attribute was removed from BenchmarkConfig, so this assertion is no longer valid.


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent
