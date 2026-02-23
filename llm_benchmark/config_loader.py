# Loads + validates YAML configs via Pydantic

import logging
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelEntry(BaseModel):
    id: str
    provider: str
    model_string: str
    input_price_per_1k_tokens: float
    output_price_per_1k_tokens: float


class ModelsConfig(BaseModel):
    models: list[ModelEntry]


class TaskConfig(BaseModel):
    task_id: str
    n_samples: int
    task_type: str = "task_a"  # "task_a" (categorize) or "task_b" (extract regexes)


class BenchmarkConfig(BaseModel):
    oracle_model_id: str
    output_dir: str
    mlflow_experiment_name: str
    tasks: list[TaskConfig]


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Load and validate config/models.yaml."""
    import yaml
    path = path or Path("config/models.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(data)


def load_benchmark_config(path: Path | None = None) -> BenchmarkConfig:
    """Load and validate config/benchmark.yaml."""
    import yaml
    path = path or Path("config/benchmark.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BenchmarkConfig.model_validate(data)
