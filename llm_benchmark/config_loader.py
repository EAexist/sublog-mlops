import logging
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, RootModel, computed_field

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _load_yaml_config(path: Path, config_class: type[T]) -> T:
    """Generic YAML config loader following the same pattern."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config_class.model_validate(data)


class ModelEntry(BaseModel):
    id: str
    provider: str

    @computed_field
    @property
    def full_id(self) -> str:
        return f"{self.provider}/{self.id}"

    # input_price_per_1k_tokens: float
    # output_price_per_1k_tokens: float


class ModelsConfig(BaseModel):
    models: list[ModelEntry]


class TaskEntry(BaseModel):
    name: str
    description: str
    batch_size: int
    response_model: str


class TasksConfig(RootModel[dict[str, TaskEntry]]):
    """Configuration for task definitions."""


class DatasetConfig(BaseModel):
    oracle_model_id: str
    output_dir: str
    mlflow_experiment_name: str
    n_templates_per_event: int
    n_samples_per_template: int
    locales: list[str]
    do_update_templates: bool
    do_update_parameters: bool
    hf_repo: str


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Load and validate config/models.yaml."""
    path = path or Path("config/models.yaml")
    return _load_yaml_config(path, ModelsConfig)


def load_tasks_config(path: Path | None = None) -> TasksConfig:
    """Load and validate config/tasks.yml."""
    path = path or Path("config/tasks.yml")
    return _load_yaml_config(path, TasksConfig)


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    """Load and validate config/dataset.yaml."""
    path = path or Path("config/dataset.yaml")
    return _load_yaml_config(path, DatasetConfig)
