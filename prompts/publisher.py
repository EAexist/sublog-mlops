"""
Prompt version management and publishing logic.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import ArchiveIndex, LatestPointer, PromptVersion

logger = logging.getLogger(__name__)


class PromptPublisher:
    """Manages prompt versioning, archiving, and publishing."""

    def __init__(self, prompts_root: Path):
        self.prompts_root = Path(prompts_root)
        self.versions_dir = self.prompts_root / "versions"
        self.archive_dir = self.prompts_root / "archive"
        self.latest_file = self.prompts_root / "latest.json"
        self.archive_index_file = self.archive_dir / "index.json"

    def create_new_version(
        self,
        version: str,
        tasks: dict[str, dict[str, Any]],
        changelog: str,
        author: str = "prompt_engineer",
    ) -> Path:
        """Create a new prompt version."""
        version_dir = self.versions_dir / f"{datetime.now().strftime('%Y-%m-%d')}_v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save individual task files
        for task_id, task_data in tasks.items():
            task_file = version_dir / f"{task_id}.yaml"
            with open(task_file, "w") as f:
                import yaml

                yaml.dump(task_data, f, default_flow_style=False)

        # Create metadata
        metadata = PromptVersion(
            version=version,
            created_at=datetime.now(),
            tasks=list(tasks.keys()),
            changelog=changelog,
            author=author,
            total_prompts=len(tasks),
        )

        meta_file = version_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

        logger.info(f"Created new prompt version: {version_dir}")
        return version_dir

    def update_latest_pointer(self, version_dir: Path) -> None:
        """Update the latest pointer to point to the new version."""
        relative_path = version_dir.relative_to(self.prompts_root)
        version = version_dir.name.split("_v")[-1]

        latest = LatestPointer(version=version, path=str(relative_path), created_at=datetime.now())

        with open(self.latest_file, "w") as f:
            json.dump(latest.model_dump(), f, indent=2, default=str)

        logger.info(f"Updated latest pointer to: {relative_path}")

    def archive_old_versions(self, max_versions: int = 10) -> None:
        """Archive old versions, keeping only max_versions in the active directory."""
        version_dirs = sorted(
            [d for d in self.versions_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        if len(version_dirs) <= max_versions:
            return

        # Load archive index
        try:
            with open(self.archive_index_file) as f:
                archive_index = ArchiveIndex(**json.load(f))
        except FileNotFoundError:
            archive_index = ArchiveIndex()

        # Archive old versions
        versions_to_archive = version_dirs[max_versions:]
        for version_dir in versions_to_archive:
            archive_path = self.archive_dir / version_dir.name
            shutil.move(str(version_dir), str(archive_path))
            archive_index.archived_versions.append(version_dir.name)
            archive_index.total_archived += 1
            archive_index.last_archived = datetime.now()

            logger.info(f"Archived version: {version_dir.name}")

        # Save updated archive index
        with open(self.archive_index_file, "w") as f:
            json.dump(archive_index.model_dump(), f, indent=2, default=str)

    def get_latest_version(self) -> Path | None:
        """Get the path to the latest version."""
        try:
            with open(self.latest_file) as f:
                latest = LatestPointer(**json.load(f))
            if latest.path:
                return self.prompts_root / latest.path
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return None

    def publish_version(
        self,
        version: str,
        tasks: dict[str, dict[str, Any]],
        changelog: str,
        max_versions: int = 10,
        author: str = "prompt_engineer",
    ) -> Path:
        """Create and publish a new prompt version."""
        # Create new version
        version_dir = self.create_new_version(version, tasks, changelog, author)

        # Update latest pointer
        self.update_latest_pointer(version_dir)

        # Archive old versions
        self.archive_old_versions(max_versions)

        return version_dir
