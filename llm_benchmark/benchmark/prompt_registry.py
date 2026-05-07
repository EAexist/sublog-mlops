"""
Prompt loader module for reading and formatting latest prompts by task_id.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from prompts.schema import PromptTask

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and formats the latest prompts by task_id with parameter insertion.
    """

    def __init__(self, prompts_dir: Path | None = None):
        """
        Initialize the PromptLoader with optional custom prompts directory.

        Args:
            prompts_dir: Path to prompts directory, defaults to ../prompts
        """
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)

        logger.info(f"PromptLoader initialized with prompts directory: {self.prompts_dir}")

    def get_latest_prompt_by_task_id(self, task_id: str, parameters: dict[str, Any]) -> str:
        """
        Load the latest prompt for a given task_id and insert parameters.

        Args:
            task_id: The task identifier to load prompt for
            parameters: Dictionary of parameter values to insert into the prompt

        Returns:
            Formatted prompt string with parameters inserted

        Raises:
            FileNotFoundError: If prompt files are not found
            ValueError: If task_id is not found in latest version
        """
        logger.debug(f"Loading latest prompt for task_id: {task_id}")

        # Load latest version info
        latest_file = self.prompts_dir / "latest.json"
        if not latest_file.exists():
            raise FileNotFoundError(f"Latest version file not found: {latest_file}")

        with open(latest_file) as f:
            latest_data = json.load(f)

        # Get the version directory path
        version_path = self.prompts_dir / latest_data["path"]
        if not version_path.exists():
            raise FileNotFoundError(f"Version directory not found: {version_path}")

        # Load the task file
        task_file = version_path / f"{task_id}.yaml"
        if not task_file.exists():
            raise ValueError(f"Task '{task_id}' not found in version {latest_data['version']}")

        import yaml

        with open(task_file, encoding="utf-8") as f:
            task_data = yaml.safe_load(f)

        # Validate task data
        task = PromptTask(**task_data)
        logger.debug(f"Loaded prompt for task '{task_id}': {task.name}")

        # Format prompt with parameters
        formatted_prompt = self._format_prompt(task.prompt, parameters)

        logger.info(f"Successfully formatted prompt for task '{task_id}'")
        return formatted_prompt

    def _format_prompt(self, prompt_template: str, parameters: dict[str, Any]) -> str:
        """
        Format a prompt template by inserting parameter values.

        Args:
            prompt_template: The prompt template with placeholders
            parameters: Dictionary of parameter values to insert

        Returns:
            Formatted prompt string

        Raises:
            KeyError: If required parameter is missing
        """
        try:
            # Use string formatting with parameter substitution
            formatted_prompt = prompt_template.format(**parameters)
            return formatted_prompt
        except KeyError as e:
            missing_param = str(e).strip("'\"")
            raise KeyError(f"Missing required parameter: {missing_param}") from e

    def get_task_info(self, task_id: str) -> PromptTask:
        """
        Get task information without formatting the prompt.

        Args:
            task_id: The task identifier

        Returns:
            PromptTask instance with task metadata

        Raises:
            FileNotFoundError: If task files are not found
            ValueError: If task_id is not found in latest version
        """
        logger.debug(f"Getting task info for task_id: {task_id}")

        # Load latest version info
        latest_file = self.prompts_dir / "latest.json"
        if not latest_file.exists():
            raise FileNotFoundError(f"Latest version file not found: {latest_file}")

        with open(latest_file) as f:
            latest_data = json.load(f)

        # Get the version directory path
        version_path = self.prompts_dir / latest_data["path"]
        if not version_path.exists():
            raise FileNotFoundError(f"Version directory not found: {version_path}")

        # Load the task file
        task_file = version_path / f"{task_id}.yaml"
        if not task_file.exists():
            raise ValueError(f"Task '{task_id}' not found in version {latest_data['version']}")

        import yaml

        with open(task_file, encoding="utf-8") as f:
            task_data = yaml.safe_load(f)

        return PromptTask(**task_data)

    def list_available_tasks(self) -> list[str]:
        """
        List all available task IDs in the latest version.

        Returns:
            List of task IDs available in the latest version

        Raises:
            FileNotFoundError: If latest version files are not found
        """
        logger.debug("Listing available tasks in latest version")

        # Load latest version info
        latest_file = self.prompts_dir / "latest.json"
        if not latest_file.exists():
            raise FileNotFoundError(f"Latest version file not found: {latest_file}")

        with open(latest_file) as f:
            latest_data = json.load(f)

        # Get the version directory path
        version_path = self.prompts_dir / latest_data["path"]
        if not version_path.exists():
            raise FileNotFoundError(f"Version directory not found: {version_path}")

        # Load meta.json to get task list
        meta_file = version_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")

        with open(meta_file) as f:
            meta_data = json.load(f)

        return meta_data.get("tasks", [])


default_path = os.getenv("PROMPT_REGISTRY_PATH")
prompt_loader = PromptLoader(prompts_dir=Path(default_path) if default_path else None)
