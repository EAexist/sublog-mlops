"""
Dataset publisher module.

Handles versioning, archiving, and publishing of datasets to the git submodule.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

logger = logging.getLogger(__name__)


class DatasetPublisher:
    """Handles publishing datasets with versioning."""

    def __init__(
        self,
        base_path: Path,
        component_name: str,
        max_versions: int = 10,
    ):
        """
        Initialize the publisher.

        Args:
            base_path: Base path for publishing
            component_name: Name of the component (e.g., 'templates', 'parameters', 'samples')
            max_versions: Maximum versions to keep in versions/ directory
        """
        self.base_path = base_path
        self.component_name = component_name
        self.max_versions = max_versions
        self.versions_dir = base_path / "versions"
        self.latest_pointer = base_path / "latest.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_path(self) -> Path:
        """Returns the path to the current latest file."""
        if not self.latest_pointer.exists():
            raise FileNotFoundError("No latest parameters found")
        with open(self.latest_pointer) as f:
            data = json.load(f)
            return self.base_path / data["relative_path"]

    def publish(self, data: Any, run_id: str) -> Path:
        """Saves a new version and updates the latest pointer."""
        version_dir = self.versions_dir / run_id
        file_path = version_dir / f"{self.component_name}.jsonl"

        self.save_jsonl(data, file_path)

        # Update latest.json (The "Promotion" step)
        latest_info = {
            "run_id": run_id,
            "relative_path": file_path.relative_to(self.base_path).as_posix(),
            "timestamp": datetime.now().isoformat(),
        }
        self.latest_pointer.write_text(json.dumps(latest_info, indent=2))

        self.enforce_retention()
        return file_path

    def enforce_retention(self):
        """Simple cleanup of old version directories."""
        dirs = sorted([d for d in self.versions_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        while len(dirs) > self.max_versions:
            old_dir = dirs.pop(0)
            shutil.rmtree(old_dir)
            logger.info(f"Deleted old version: {old_dir.name}")

    def save_json(self, data: Any, path: Path) -> None:
        """Persist dataset to path (e.g. JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable_data = TypeAdapter(type(data)).dump_python(data, mode="json")

        path.write_text(json.dumps(serializable_data, indent=2), encoding="utf-8")
        logger.info("Saved dataset to %s", path)

    def save_jsonl(self, data: Any, path: Path) -> None:
        """Persist dataset to path as JSONL format (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable_data = TypeAdapter(type(data)).dump_python(data, mode="json")

        # Handle different data types for JSONL format
        if isinstance(serializable_data, list):
            # Write each item as a separate JSON line
            lines = [json.dumps(item, separators=(",", ":")) for item in serializable_data]
            content = "\n".join(lines)
        else:
            # For single objects, write as one JSON line
            content = json.dumps(serializable_data, separators=(",", ":"))

        path.write_text(content, encoding="utf-8")
        logger.info("Saved dataset to %s as JSONL", path)

    def save_manifest(
        self,
        output_path: Path,
        template_path: Path,
        param_path: Path,
        dataset_path: Path,
        run_id: str,
    ):

        manifest = {
            "run_id": run_id,
            "components": {
                "templates": str(template_path),
                "parameters": str(param_path),
                "dataset": str(dataset_path),
            },
        }

        # Save the manifest so the next step knows exactly what this "virtual run" consists of
        # Extract output_path from dataset_path (go up 3 levels: emails/versions/run_id/samples.json -> data)
        manifest_path = output_path / "runs" / run_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_json(manifest, manifest_path)

        return manifest_path
