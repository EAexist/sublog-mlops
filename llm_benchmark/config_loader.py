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
    task_type: str = "task_a"  # "task_a" (categorize) or "task_b" (extract regexes)


class DatasetConfig(BaseModel):
    oracle_model_id: str
    output_dir: str
    mlflow_experiment_name: str
    n_templates_per_event: int
    n_samples_per_template: int
    tasks: list[TaskConfig]
    locales: list[str]
    do_update_templates: bool
    do_update_parameters: bool
    hf_repo: str


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Load and validate config/models.yaml."""
    import yaml

    path = path or Path("config/models.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(data)


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    """Load and validate config/dataset.yaml."""
    import yaml

    path = path or Path("config/dataset.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetConfig.model_validate(data)
