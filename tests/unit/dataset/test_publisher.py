"""
Tests for dataset publisher module.
"""

import json
from unittest.mock import patch

import pytest
from llm_benchmark.dataset.publisher import DatasetPublisher


class TestDatasetPublisher:
    """Test cases for dataset publisher."""

    def test_initialization_default_values(self, tmp_path):
        """Test publisher initialization with default values."""
        base_path = tmp_path / "test_component"

        publisher = DatasetPublisher(base_path, "test_component")

        assert publisher.base_path == base_path
        assert publisher.component_name == "test_component"
        assert publisher.max_versions == 10
        assert publisher.versions_dir == base_path / "versions"
        assert publisher.latest_pointer == base_path / "latest.json"

    def test_initialization_custom_values(self, tmp_path):
        """Test publisher initialization with custom values."""
        base_path = tmp_path / "test_component"

        publisher = DatasetPublisher(base_path, "custom_component", max_versions=5)

        assert publisher.base_path == base_path
        assert publisher.component_name == "custom_component"
        assert publisher.max_versions == 5

    @patch("llm_benchmark.dataset.publisher.datetime")
    def test_publish_creates_version_dir_and_pointer(self, mock_datetime, tmp_path):
        """Test publish creates version directory and updates pointer."""
        # Mock datetime to return a proper ISO format string
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-20T12:00:00"

        base_path = tmp_path / "test_component"
        publisher = DatasetPublisher(base_path, "test_component")

        test_data = {"test": "data"}
        run_id = "test_run_123"

        # Execute
        result_path = publisher.publish(test_data, run_id)

        # Verify version directory created
        expected_version_dir = base_path / "versions" / "test_run_123"
        assert expected_version_dir.exists()

        # Verify file created
        expected_file = expected_version_dir / "test_component.jsonl"
        assert expected_file.exists()

        # Verify file content is valid JSONL format
        with open(expected_file) as f:
            lines = f.readlines()
            assert len(lines) == 1  # Single object should be one line
            # Parse the JSON line to verify it's valid JSON
            parsed_data = json.loads(lines[0].strip())
            assert parsed_data == test_data

        # Verify latest pointer updated
        assert publisher.latest_pointer.exists()
        with open(publisher.latest_pointer) as f:
            pointer_data = json.load(f)
            assert pointer_data["run_id"] == run_id
            assert "test_run_123" in pointer_data["relative_path"]

        # Verify return path
        assert result_path == expected_file

    def test_get_latest_path_success(self, tmp_path):
        """Test get_latest_path returns correct path when latest exists."""
        base_path = tmp_path / "test_component"
        publisher = DatasetPublisher(base_path, "test_component")

        # Create latest pointer
        version_dir = base_path / "versions" / "test_run"
        version_dir.mkdir(parents=True)
        test_file = version_dir / "test_component.jsonl"
        test_file.write_text('{"test": "data"}\n')

        latest_info = {
            "run_id": "test_run",
            "relative_path": "versions/test_run/test_component.jsonl",
        }
        publisher.latest_pointer.write_text(json.dumps(latest_info))

        # Execute
        result = publisher.get_latest_path()

        assert result == test_file

    def test_get_latest_path_not_found(self, tmp_path):
        """Test get_latest_path raises FileNotFoundError when no latest exists."""
        base_path = tmp_path / "test_component"
        publisher = DatasetPublisher(base_path, "test_component")

        # Execute and verify exception
        with pytest.raises(FileNotFoundError, match="No latest parameters found"):
            publisher.get_latest_path()

    def test_enforce_retention_removes_old_versions(self, tmp_path):
        """Test retention policy removes old versions when exceeded."""
        base_path = tmp_path / "test_component"
        publisher = DatasetPublisher(base_path, "test_component", max_versions=3)

        # Create 3 version directories
        for i in range(3):
            version_dir = base_path / "versions" / f"run_{i}"
            version_dir.mkdir(parents=True)

        # Execute publish (should trigger retention)
        publisher.publish({"test": "data"}, "run_3")

        # Should only keep 3 most recent versions
        remaining_dirs = [d for d in (base_path / "versions").iterdir() if d.is_dir()]
        assert len(remaining_dirs) == 3

        # Verify oldest was removed
        dir_names = [d.name for d in remaining_dirs]
        assert "run_0" not in dir_names  # Oldest should be removed
        assert "run_3" in dir_names  # Latest should exist
