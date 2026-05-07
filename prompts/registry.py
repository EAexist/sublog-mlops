"""
HuggingFace prompt registry integration.
"""

import json
import logging
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, Repository

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("huggingface_hub not installed. Registry functionality will be limited.")

from .schema import PromptVersion

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Manages prompt registry operations with HuggingFace."""

    def __init__(self, repo_id: str, local_path: Path | None = None):
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required for registry functionality")

        self.repo_id = repo_id
        self.local_path = local_path or Path("./prompt_registry")
        self.api = HfApi()

    def push_version(self, version_dir: Path, commit_message: str | None = None) -> str:
        """Push a prompt version to the HuggingFace registry."""
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")

        # Load version metadata
        meta_file = version_dir / "meta.json"
        with open(meta_file) as f:
            metadata = PromptVersion(**json.load(f))

        # Prepare commit message
        if not commit_message:
            commit_message = f"Add prompt version {metadata.version}: {metadata.changelog}"

        # Clone or update repository
        if not self.local_path.exists():
            repo = Repository(local_dir=str(self.local_path), clone_from=self.repo_id, token=True)
        else:
            repo = Repository(local_dir=str(self.local_path), token=True)
            repo.git_pull()

        # Copy version files to registry
        target_dir = self.local_path / f"versions/{metadata.version}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from version directory
        import shutil

        for file_path in version_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, target_dir / file_path.name)

        # Update latest pointer in registry
        latest_file = self.local_path / "latest.json"
        latest_data = {
            "version": metadata.version,
            "path": f"versions/{metadata.version}",
            "created_at": metadata.created_at.isoformat(),
        }
        with open(latest_file, "w") as f:
            json.dump(latest_data, f, indent=2)

        # Commit and push changes
        repo.git_add(pattern=".")
        repo.git_commit(commit_message)
        repo.git_push()

        logger.info(f"Pushed prompt version {metadata.version} to registry")
        return f"https://huggingface.co/{self.repo_id}/tree/main/versions/{metadata.version}"

    def pull_latest_version(self, target_dir: Path) -> Path:
        """Pull the latest version from the registry."""
        if not self.local_path.exists():
            repo = Repository(local_dir=str(self.local_path), clone_from=self.repo_id, token=True)
        else:
            repo = Repository(local_dir=str(self.local_path), token=True)
            repo.git_pull()

        # Read latest pointer
        latest_file = self.local_path / "latest.json"
        with open(latest_file) as f:
            latest_data = json.load(f)

        # Copy latest version to target directory
        source_dir = self.local_path / latest_data["path"]
        target_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        for file_path in source_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, target_dir / file_path.name)

        logger.info(f"Pulled latest version {latest_data['version']} to {target_dir}")
        return target_dir

    def list_versions(self) -> dict[str, Any]:
        """List all available versions in the registry."""
        try:
            repo_info = self.api.repo_info(repo_id=self.repo_id)
            files = [
                f.rfilename for f in repo_info.repo_files if f.rfilename.startswith("versions/")
            ]

            versions = []
            for file_path in files:
                parts = file_path.split("/")
                if len(parts) >= 3 and parts[2] == "meta.json":
                    version = parts[1]
                    versions.append(version)

            return {
                "repo_id": self.repo_id,
                "versions": sorted(versions),
                "total_versions": len(versions),
            }
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return {"repo_id": self.repo_id, "versions": [], "total_versions": 0}
