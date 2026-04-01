"""
Dataset loader module for accessing HuggingFace datasets with double access pattern.

Follows the pattern: 1. Access latest.json -> 2. Follow path in latest.json to actual data.
"""

import logging
from typing import Any

from datasets import load_dataset
from datasets_shared.schema import EmailTemplate, EmailTextParameterSet, Sample
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HuggingFaceDatasetLoader:
    """Loads datasets from HuggingFace using the double access pattern."""

    def __init__(self, dataset_name: str):
        """
        Initialize the loader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "hyeon-expression/subscription-killer-synthetic-emails")
        """
        self.dataset_name = dataset_name

    def load_latest_file_path(self, component: str) -> str:
        """
        Step 1: Access latest.json to get the path to the latest data file.

        Args:
            component: Component name (e.g., 'templates', 'parameters', 'samples')

        Returns:
            Relative path to the latest data file
        """
        try:
            latest_file = f"data/{component}/latest.json"
            dataset = load_dataset(
                self.dataset_name,
                data_files=latest_file,
                field=None,
            )

            latest_info = dataset["train"][0]
            relative_path = latest_info["relative_path"]

            logger.info(f"Latest {component} path: {relative_path}")
            return relative_path

        except Exception as e:
            logger.error(f"Failed to load latest.json for {component}: {e}")
            raise

    def load_data_file(
        self, relative_path: str, data_model: type[BaseModel] | None = None
    ) -> list[Any] | list[BaseModel]:
        """
        Step 2: Load the actual data file using the path from latest.json.

        Args:
            relative_path: Path to the data file (from latest.json)
            data_model: Optional Pydantic model to validate and convert data

        Returns:
            List of data items (raw dicts or Pydantic models if specified)
        """
        try:
            dataset = load_dataset(
                self.dataset_name,
                data_files=relative_path,
                field=None,
                download_mode="force_redownload",
            )

            raw_data = list(dataset["train"])

            if data_model:
                # Convert to Pydantic models
                validated_data = [data_model.model_validate(item) for item in raw_data]
                logger.info(
                    f"Loaded {len(validated_data)} {data_model.__name__} objects from {relative_path}"
                )
                return validated_data
            else:
                logger.info(f"Loaded {len(raw_data)} raw items from {relative_path}")
                return raw_data

        except Exception as e:
            logger.error(f"Failed to load data file {relative_path}: {e}")
            raise

    def load_latest_component(
        self, component: str, data_model: type[BaseModel] | None = None
    ) -> list[BaseModel]:
        """
        Complete double access: latest.json -> actual data file.

        Args:
            component: Component name (e.g., 'templates', 'parameters', 'samples')
            data_model: Optional Pydantic model to validate and convert data

        Returns:
            List of data items (raw dicts or Pydantic models if specified)
        """
        logger.info(f"Loading latest {component}")

        # Step 1: Get latest file path
        relative_path = self.load_latest_file_path(component)

        latest_file = f"data/{component}/{relative_path}"
        # Step 2: Load actual data
        return self.load_data_file(latest_file, data_model)

    def load_latest_templates(self) -> list[EmailTemplate]:
        """Convenience method to load latest templates."""
        return self.load_latest_component("templates", EmailTemplate)  # type: ignore

    def load_latest_parameters(self) -> list[EmailTextParameterSet]:
        """Convenience method to load latest parameters."""
        return self.load_latest_component("parameters", EmailTextParameterSet)  # type: ignore

    def load_latest_samples(self) -> list[Sample]:
        """Convenience method to load latest samples."""
        return self.load_latest_component("samples", Sample)  # type: ignore

    def get_latest_metadata(self, component: str) -> dict[str, Any]:
        """
        Get metadata from latest.json for a component.

        Args:
            component: Component name

        Returns:
            Metadata dictionary with run_id, relative_path, timestamp
        """
        try:
            latest_file = f"{component}/latest.json"
            dataset = load_dataset(
                self.dataset_name,
                data_files=latest_file,
                field=None,
                download_mode="force_redownload",
            )

            return dataset["train"][0]

        except Exception as e:
            logger.error(f"Failed to load metadata for {component}: {e}")
            raise
