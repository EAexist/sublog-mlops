"""
Tests for dataset validator module.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets_shared.schema import PromptVersion
from llm_benchmark.dataset.validator import (
    DatasetValidationError,
    validate_dataset,
    validate_sample_format,
)


class TestDatasetValidator:
    """Test cases for dataset validation."""

    def test_validate_dataset_success(self, tmp_path):
        """Test successful dataset validation."""
        # Create a sample dataset file
        dataset_path = tmp_path / "test_dataset.jsonl"
        dataset_path.write_text('{"prompt": "test", "completion": "test"}\n')

        # Mock prompt version
        prompt_version = Mock(spec=PromptVersion)

        result = validate_dataset(dataset_path, 1, prompt_version)
        assert result is True

    def test_validate_dataset_file_not_found(self):
        """Test validation with non-existent file."""
        non_existent_path = Path("/non/existent/file.jsonl")
        prompt_version = Mock(spec=PromptVersion)

        with pytest.raises(DatasetValidationError):
            validate_dataset(non_existent_path, 1, prompt_version)

    def test_validate_dataset_wrong_sample_count(self, tmp_path):
        """Test validation with wrong sample count."""
        dataset_path = tmp_path / "test_dataset.jsonl"
        dataset_path.write_text('{"prompt": "test", "completion": "test"}\n')

        prompt_version = Mock(spec=PromptVersion)

        # TODO: Implement this test when validation logic is complete
        # with pytest.raises(DatasetValidationError, match="Expected 2 samples, got 1"):
        #     validate_dataset(dataset_path, 2, prompt_version)
        pass

    def test_validate_sample_format_valid(self):
        """Test valid sample format validation."""
        valid_sample = {
            "prompt": "What is 2+2?",
            "completion": "4",
            "metadata": {"source": "test"}
        }

        assert validate_sample_format(valid_sample) is True

    def test_validate_sample_format_invalid(self):
        """Test invalid sample format validation."""
        invalid_sample = {"invalid": "structure"}

        # TODO: Implement this test when validation logic is complete
        # assert validate_sample_format(invalid_sample) is False
        pass

    def test_validate_dataset_schema_validation(self, tmp_path):
        """Test schema validation against PromptVersion."""
        dataset_path = tmp_path / "test_dataset.jsonl"
        # Write invalid schema data
        dataset_path.write_text('{"wrong": "schema"}\n')

        prompt_version = Mock(spec=PromptVersion)

        # TODO: Implement this test when schema validation is complete
        pass
