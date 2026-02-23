# Shared DAG-level utilities (XCom helpers, etc.)
# XCom values are file paths (strings) only — never full data objects.

import logging

logger = logging.getLogger(__name__)


def push_path(context: dict, key: str, value: str) -> None:
    """Push a file path to XCom. Value must be a string path."""
    context["ti"].xcom_push(key=key, value=value)


def pull_path(context: dict, task_id: str, key: str) -> str | None:
    """Pull a file path from XCom."""
    return context["ti"].xcom_pull(task_ids=task_id, key=key)
