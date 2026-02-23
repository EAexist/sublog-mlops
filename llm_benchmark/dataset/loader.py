# Save/load dataset with content hash

import logging
from pathlib import Path
from llm_benchmark.dataset.schema import Dataset

logger = logging.getLogger(__name__)


def save_dataset(dataset: Dataset, path: Path) -> None:
    """Persist dataset to path (e.g. JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved dataset to %s", path)


def load_dataset(path: Path) -> Dataset:
    """Load dataset from path."""
    data = Path(path).read_text(encoding="utf-8")
    return Dataset.model_validate_json(data)
