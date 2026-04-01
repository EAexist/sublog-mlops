"""
Prompt schema and validation logic.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class PromptTask(BaseModel):
    """Individual prompt task definition."""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Task description")
    prompt: str = Field(..., description="Prompt template with placeholders")
    version: str = Field(..., description="Prompt version")
    created_at: datetime = Field(..., description="Creation timestamp")
    author: str = Field(..., description="Author of the prompt")

    @validator('task_id')
    def validate_task_id(cls, v):
        if not v or not v.replace('_', '').isalnum():
            raise ValueError('task_id must be alphanumeric with underscores only')
        return v


class PromptVersion(BaseModel):
    """Complete prompt version metadata."""
    version: str = Field(..., description="Semantic version")
    created_at: datetime = Field(..., description="Creation timestamp")
    tasks: List[str] = Field(..., description="List of task IDs in this version")
    changelog: str = Field(..., description="Version changelog")
    author: str = Field(..., description="Author of the version")
    total_prompts: int = Field(..., description="Total number of prompts")

    @validator('version')
    def validate_semver(cls, v):
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError('Version must follow semantic versioning (X.Y.Z)')
        return v


class LatestPointer(BaseModel):
    """Latest version pointer."""
    version: Optional[str] = Field(None, description="Latest version")
    path: Optional[str] = Field(None, description="Path to latest version directory")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class ArchiveIndex(BaseModel):
    """Archive manifest."""
    archived_versions: List[str] = Field(default_factory=list, description="List of archived versions")
    last_archived: Optional[datetime] = Field(None, description="Last archival timestamp")
    total_archived: int = Field(default=0, description="Total archived versions")
