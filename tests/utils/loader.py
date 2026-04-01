import json
from pathlib import Path
from typing import TypeVar

from datasets_shared.schema.models import EmailTemplate, EmailTextParameterSet
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _load_json_objects(
    file_path: Path, object_key: str, model_class: type[T], item_name: str
) -> list[T]:
    """Helper function to load JSON objects from a file (supports both JSON and JSONL)."""
    if not file_path.exists():
        raise FileNotFoundError(f"{item_name} file not found: {file_path}")

    # Handle JSONL format (one JSON object per line)
    if file_path.suffix == ".jsonl":
        data_list = []
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # Skip empty lines
                    try:
                        item_data = json.loads(line)
                        data_list.append(item_data)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON on line {line_num} in {file_path}: {e}"
                        ) from e
    else:
        # Handle regular JSON format
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single object format and array format
        if object_key in data:
            data_list = data[object_key]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError(f"Invalid data format in {file_path}")

    return [model_class(**item_data) for item_data in data_list]


def load_email_templates_from_file(file_path: Path) -> list[EmailTemplate]:
    """Load and parse email templates from a JSON file."""
    return _load_json_objects(file_path, "templates", EmailTemplate, "Template")


def load_email_parameters_from_file(file_path: Path) -> list[EmailTextParameterSet]:
    """Load and parse email text parameter sets from a JSON file."""
    return _load_json_objects(file_path, "parameters", EmailTextParameterSet, "Parameter")
