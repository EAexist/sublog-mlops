"""
Factory for creating and caching task instances with YAML configuration.
"""

from pathlib import Path
from typing import Any

from llm_benchmark.benchmark.task.models import RESPONSE_MODEL_REGISTRY
from llm_benchmark.benchmark.task.task import (
    BaseTask,
    EmailCategorizationTask,
    EmailTemplateExtractionTask,
)
from llm_benchmark.config_loader import load_tasks_config


class TaskFactory:
    """
    Factory for creating task instances from YAML configuration.
    Uses registry pattern and instance caching for stateless tasks.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the TaskFactory with configuration.

        Args:
            config_path: Path to tasks.yml file, defaults to config/tasks.yml
        """
        # Load configuration using the same pattern as other configs
        self._config = load_tasks_config(config_path).model_dump()

        # Registry mapping task_id to the Class
        self._registry: dict[str, type[BaseTask]] = {
            "email_categorization": EmailCategorizationTask,
            "email_template_extraction": EmailTemplateExtractionTask,
        }

        # Internal cache for instances (stateless tasks can be safely shared)
        self._instance_cache: dict[str, BaseTask] = {}

    def get_task(self, task_id: str) -> BaseTask:
        """
        Returns a cached instance of the task.
        Since they are stateless, sharing the instance is safe and efficient.

        Args:
            task_id: The task identifier to retrieve

        Returns:
            BaseTask instance for the given task_id

        Raises:
            ValueError: If task_id is not found in registry or config
        """
        if task_id not in self._instance_cache:
            if task_id not in self._registry:
                raise ValueError(f"Task ID '{task_id}' not found in registry.")

            if task_id not in self._config:
                raise ValueError(f"Task ID '{task_id}' not found in configuration.")

            task_class = self._registry[task_id]
            task_cfg = self._config[task_id]

            model_name = task_cfg["response_model"]
            response_model = RESPONSE_MODEL_REGISTRY.get(model_name)
            if response_model is None:
                raise ValueError(f"Response model '{model_name}' not found in registry.")

            # Create the instance once with configuration
            self._instance_cache[task_id] = task_class(
                task_id=task_id,
                name=task_cfg["name"],
                description=task_cfg["description"],
                batch_size=task_cfg["batch_size"],
                response_model=response_model,
            )

        return self._instance_cache[task_id]

    def list_available_tasks(self) -> list[str]:
        """
        List all available task IDs.

        Returns:
            List of task IDs available in the factory
        """
        return list(self._registry.keys())

    def get_all_tasks(self) -> list[BaseTask]:
        """
        Get a list of all task instances from the registry.

        Returns:
            List of all BaseTask instances available in the factory
        """
        return [self.get_task(task_id) for task_id in self.list_available_tasks()]

    def get_task_config(self, task_id: str) -> dict[str, Any]:
        """
        Get the configuration for a specific task.

        Args:
            task_id: The task identifier

        Returns:
            Configuration dictionary for the task

        Raises:
            ValueError: If task_id is not found in configuration
        """
        if task_id not in self._config:
            raise ValueError(f"Task ID '{task_id}' not found in configuration.")
        return self._config[task_id].copy()


# Singleton instance for easy access
task_factory = TaskFactory()
