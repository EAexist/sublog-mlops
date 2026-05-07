"""
Prompt schema and validation logic.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PromptParameter(BaseModel):
    """Parameter definition for prompt templates."""

    name: str = Field(..., description="Parameter name")
    description: str = Field("", description="Parameter description")
    example: str = Field(..., description="Example value for the parameter")


class PromptTask(BaseModel):
    """Individual prompt task definition."""

    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Task description")
    prompt: str = Field(..., description="Prompt template with placeholders")
    parameters: list[PromptParameter] = Field(
        default_factory=list, description="Parameter definitions"
    )
    version: str = Field(..., description="Prompt version")
    created_at: datetime = Field(..., description="Creation timestamp")
    author: str = Field(..., description="Author of the prompt")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v):
        if not v or not v.replace("_", "").isalnum():
            raise ValueError("task_id must be alphanumeric with underscores only")
        return v


class PromptVersion(BaseModel):
    """Complete prompt version metadata."""

    version: str = Field(..., description="Semantic version")
    created_at: datetime = Field(..., description="Creation timestamp")
    tasks: list[str] = Field(..., description="List of task IDs in this version")
    changelog: str = Field(..., description="Version changelog")
    author: str = Field(..., description="Author of the version")
    total_prompts: int = Field(..., description="Total number of prompts")

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v):
        import re

        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return v


class LatestPointer(BaseModel):
    """Latest version pointer."""

    version: str | None = Field(None, description="Latest version")
    path: str | None = Field(None, description="Path to latest version directory")
    created_at: datetime | None = Field(None, description="Creation timestamp")


class ArchiveIndex(BaseModel):
    """Archive manifest."""

    archived_versions: list[str] = Field(
        default_factory=list, description="List of archived versions"
    )
    last_archived: datetime | None = Field(None, description="Last archival timestamp")
    total_archived: int = Field(default=0, description="Total archived versions")
