import time
from unittest.mock import MagicMock, patch

import pytest
from datasets_shared.loader.loader import load_samples_from_file
from datasets_shared.schema import SubscriptionEventType
from llm_benchmark.dataset.constants import CompanyInfo
from llm_benchmark.pipeline import step_generate_datasets
from tests.utils import load_email_parameters_from_file, load_email_templates_from_file


@pytest.mark.integration
# @pytest.mark.skip(reason="Manual skip to avoid API costs. Uncomment this decorator to run.")
class TestStepGenerateDatasets:
    """Integration tests for step_generate_datasets pipeline using real oracle calls."""

    @pytest.fixture(autouse=True)
    def setup_mock_config(self):
        """Shared mock configuration for all tests."""
        self.mock_config = MagicMock()
        self.mock_config.oracle_model_id = "groq/llama-3.1-8b-instant"
        self.mock_config.n_templates_per_event = 1
        self.mock_config.n_samples_per_template = 1
        self.mock_config.locales = ["en_US"]

    @pytest.mark.asyncio
    @patch("llm_benchmark.pipeline.load_dataset_config")
    @patch("datasets_shared.config.DATA_ROOT")
    async def test_complete_dataset_generation(self, mock_data_root, mock_load_config, tmp_path):
        """Test complete dataset generation with real oracle calls and comprehensive validation."""
        import llm_benchmark.dataset.generator.template_generator as template_generator_module
        import llm_benchmark.pipeline as pipeline_module

        # Setup mock companies as proper CompanyInfo objects (direct assignment like unit test)
        original_companies = pipeline_module.COMPANIES
        original_template_companies = template_generator_module.COMPANIES

        mock_companies = [
            CompanyInfo(
                id="netflix",
                name="Netflix",
                email="support@netflix.com",
                industry="Streaming",
                require_shared_email_discriminator=False,
            ),
        ]

        # Update both modules that import COMPANIES
        pipeline_module.COMPANIES = mock_companies
        template_generator_module.COMPANIES = mock_companies

        try:
            # Setup configuration with real oracle model
            mock_load_config.return_value = self.mock_config
            # Setup DATA_ROOT to use temporary directory
            mock_data_root.return_value = tmp_path / "data"

            # Execute function with real oracle calls
            run_id = "integration_test_run"
            result = await step_generate_datasets(run_id, tmp_path)
            output_path = tmp_path / "data"

            # Verify manifest file was created
            # manifest_file = output_path / "runs" / run_id / "manifest.json"
            # assert manifest_file.exists()
            # manifest_content = json.loads(manifest_file.read_text(encoding="utf-8"))

            # assert manifest_content["run_id"] == run_id
            # assert "components" in manifest_content
            # assert "templates" in manifest_content["components"]
            # assert "parameters" in manifest_content["components"]
            # assert "dataset" in manifest_content["components"]

            # Verify the component paths are strings and point to existing files

            # template_path_str = manifest_content["components"]["templates"]
            # parameter_path_str = manifest_content["components"]["parameters"]
            # dataset_path_str = manifest_content["components"]["dataset"]

            # assert isinstance(template_path_str, str)
            # assert isinstance(parameter_path_str, str)
            # assert isinstance(dataset_path_str, str)

            # # Verify files actually exist
            # assert pathlib.Path(template_path_str).exists()
            # assert pathlib.Path(parameter_path_str).exists()
            # assert pathlib.Path(dataset_path_str).exists()

            # Verify template files were created and have content
            template_file = output_path / "templates" / "versions" / run_id / "templates.jsonl"
            assert template_file.exists()
            templates = load_email_templates_from_file(template_file)
            assert len(templates) > 0

            # Verify parameter files were created and have content
            parameter_file = output_path / "parameters" / "versions" / run_id / "parameters.jsonl"
            assert parameter_file.exists()
            parameters = load_email_parameters_from_file(parameter_file)
            assert len(parameters) > 0

            # Verify dataset file was created with proper structure
            dataset_file = output_path / "emails" / "versions" / run_id / "samples.jsonl"
            assert dataset_file.exists()
            samples = load_samples_from_file(dataset_file)
            assert len(samples) > 0

            # Verify return value
            assert result

            # Comprehensive sample validation
            sample = samples[0]
            required_fields = ["id", "subject", "snippet", "subscription_event_type", "company_id"]
            for field in required_fields:
                assert hasattr(sample, field), f"Missing required field: {field}"

            # Validate content looks realistic (generated by real LLM)
            assert len(sample.subject) > 0
            assert len(sample.snippet) > 0
            assert sample.subscription_event_type in [e.value for e in SubscriptionEventType]

            assert "{{client_name}}" not in sample.snippet  # Should be substituted
            assert "{{date}}" not in sample.snippet  # Should be substituted
            assert "{{payment_amount}}" not in sample.snippet  # Should be substituted

        finally:
            # Restore original companies in both modules
            pipeline_module.COMPANIES = original_companies
            template_generator_module.COMPANIES = original_template_companies

    @pytest.mark.skip(reason="Manual skip to avoid API costs. Uncomment this decorator to run.")
    @patch("llm_benchmark.pipeline.load_dataset_config")
    async def test_oracle_error_handling(self, mock_load_config, tmp_path):
        """Test behavior when real oracle calls fail with invalid model."""
        # Setup configuration with invalid model to trigger error
        self.mock_config.oracle_model_id = "invalid/model-id"
        mock_load_config.return_value = self.mock_config

        # Execute and verify exception is raised
        run_id = "oracle_error_test"
        with pytest.raises(Exception):  # Should fail with invalid model
            await step_generate_datasets(run_id, tmp_path)

    @pytest.mark.skip(reason="Manual skip to avoid API costs. Uncomment this decorator to run.")
    @patch("llm_benchmark.pipeline.load_dataset_config")
    async def test_oracle_performance(self, mock_load_config, tmp_path):
        """Test real oracle performance and throttling."""
        # Setup configuration with slightly larger dataset for performance testing
        self.mock_config.n_templates_per_event = 2
        mock_load_config.return_value = self.mock_config

        # Track execution time
        start_time = time.time()

        # Execute with real oracle calls
        run_id = "performance_test"
        await step_generate_datasets(run_id, tmp_path)

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify it completes in reasonable time (should be < 60 seconds for small dataset)
        assert execution_time < 60, f"Test took too long: {execution_time:.2f} seconds"

        # Verify files were created
        expected_run_dir = tmp_path / "data" / run_id
        assert expected_run_dir.exists()

        dataset_file = tmp_path / "data" / "emails" / "versions" / run_id / "samples.jsonl"
        assert dataset_file.exists()


# class TestStepGenerateDatasetsUnit:
#     """Unit tests for step_generate_datasets without real API calls."""

#     def test_file_structure(self, tmp_path):
#         """Test that expected file structure is created."""
#         # This test verifies the directory structure logic
#         run_id = "structure_test"

#         expected_paths = {
#             "run_dir": tmp_path / "data" / run_id,
#             "datasets_dir": tmp_path / "data" / run_id / "datasets",
#             "templates_dir": tmp_path / "data" / "templates" / "versions" / run_id,
#             "parameters_dir": tmp_path / "data" / "parameters" / "versions" / run_id,
#             "emails_dir": tmp_path / "data" / "emails" / "versions" / run_id,
#         }

#         # Verify path construction logic
#         for path_name, path_obj in expected_paths.items():
#             assert str(path_obj).endswith(run_id), f"{path_name} should end with run_id"

#         # Verify relative structure
#         assert expected_paths["datasets_dir"].parent == expected_paths["run_dir"]
#         assert expected_paths["templates_dir"].parent.parent.parent == tmp_path / "data" / "templates"
