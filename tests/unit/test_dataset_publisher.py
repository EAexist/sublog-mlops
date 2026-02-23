"""
Tests for dataset publisher module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llm_benchmark.dataset.publisher import DatasetPublisher


class TestDatasetPublisher:
    """Test cases for dataset publisher."""
    
    def test_publisher_initialization(self, tmp_path):
        """Test publisher initialization with default values."""
        submodule_path = tmp_path / "datasets"
        
        publisher = DatasetPublisher(submodule_path)
        
        assert publisher.submodule_path == submodule_path
        assert publisher.max_versions == 10
        assert publisher.remote_branch == "main"
    
    def test_publisher_initialization_custom_values(self, tmp_path):
        """Test publisher initialization with custom values."""
        submodule_path = tmp_path / "datasets"
        
        publisher = DatasetPublisher(
            submodule_path,
            max_versions=5,
            remote_branch="develop"
        )
        
        assert publisher.max_versions == 5
        assert publisher.remote_branch == "develop"
    
    @patch('llm_benchmark.dataset.publisher.datetime')
    def test_create_version_dir(self, mock_datetime, tmp_path):
        """Test version directory creation."""
        mock_datetime.now.return_value.strftime.return_value = "2025-01-20"
        
        submodule_path = tmp_path / "datasets"
        publisher = DatasetPublisher(submodule_path)
        
        version_dir = publisher._create_version_dir("run_abc123")
        
        expected_path = submodule_path / "versions" / "2025-01-20_run_abc123"
        assert version_dir == expected_path
        assert version_dir.exists()
    
    def test_publish_dataset_success(self, tmp_path):
        """Test successful dataset publishing."""
        submodule_path = tmp_path / "datasets"
        submodule_path.mkdir()
        
        # Create source dataset
        source_dataset = tmp_path / "source.jsonl"
        source_dataset.write_text('{"prompt": "test", "completion": "test"}\n')
        
        publisher = DatasetPublisher(submodule_path)
        
        with patch.object(publisher, '_create_version_dir') as mock_create_dir:
            mock_version_dir = submodule_path / "versions" / "2025-01-20_run_abc123"
            mock_version_dir.mkdir(parents=True, exist_ok=True)
            mock_create_dir.return_value = mock_version_dir
            
            published_path = publisher.publish_dataset(
                source_dataset,
                task_id="test_task",
                dag_run_id="run_abc123",
                n_samples=1,
                oracle_model="gpt-4"
            )
            
            expected_path = mock_version_dir / "test_task.jsonl"
            assert published_path == expected_path
    
    def test_update_latest_pointer_symlink(self, tmp_path):
        """Test updating latest pointer with symlink."""
        submodule_path = tmp_path / "datasets"
        version_dir = submodule_path / "versions" / "2025-01-20_run_abc123"
        version_dir.mkdir(parents=True)
        
        publisher = DatasetPublisher(submodule_path)
        
        # TODO: Implement this test when symlink logic is complete
        # publisher._update_latest_pointer(version_dir)
        # latest_link = submodule_path / "latest"
        # assert latest_link.is_symlink()
        # assert latest_link.resolve() == version_dir
        pass
    
    def test_update_latest_pointer_json(self, tmp_path):
        """Test updating latest pointer with JSON file."""
        submodule_path = tmp_path / "datasets"
        version_dir = submodule_path / "versions" / "2025-01-20_run_abc123"
        version_dir.mkdir(parents=True)
        
        publisher = DatasetPublisher(submodule_path)
        
        # TODO: Implement this test when JSON pointer logic is complete
        pass
    
    def test_enforce_retention(self, tmp_path):
        """Test retention policy enforcement."""
        submodule_path = tmp_path / "datasets"
        versions_dir = submodule_path / "versions"
        versions_dir.mkdir(parents=True)
        
        # Create old versions
        for i in range(12):
            version_dir = versions_dir / f"2025-01-{i:02d}_run_{i}"
            version_dir.mkdir()
        
        archive_dir = submodule_path / "archive"
        archive_dir.mkdir()
        
        publisher = DatasetPublisher(submodule_path, max_versions=10)
        
        # TODO: Implement this test when retention logic is complete
        # publisher._enforce_retention()
        # 
        # # Should have 10 versions in versions/, 2 in archive/
        # assert len(list(versions_dir.iterdir())) == 10
        # assert len(list(archive_dir.iterdir())) == 2
        pass
    
    @patch('subprocess.run')
    def test_commit_and_push(self, mock_subprocess, tmp_path):
        """Test git commit and push operations."""
        submodule_path = tmp_path / "datasets"
        version_dir = submodule_path / "versions" / "2025-01-20_run_abc123"
        version_dir.mkdir(parents=True)
        
        publisher = DatasetPublisher(submodule_path)
        
        # TODO: Implement this test when git operations are complete
        # publisher._commit_and_push(version_dir, "test_task")
        # 
        # # Verify git commands were called
        # mock_subprocess.assert_any_call(["git", "add", "."], cwd=submodule_path)
        # mock_subprocess.assert_any_call(
        #     ["git", "commit", "-m", "dataset: test_task 2025-01-20_run_abc123"],
        #     cwd=submodule_path
        # )
        # mock_subprocess.assert_any_call(
        #     ["git", "push", "origin", "main"],
        #     cwd=submodule_path
        # )
        pass
