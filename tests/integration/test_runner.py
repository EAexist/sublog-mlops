# Integration tests for benchmark runner

from unittest.mock import MagicMock, patch

import pytest
from llm_benchmark.benchmark.task.task_factory import task_factory
from llm_benchmark.langfuse.langfuse_client import LangfuseClient
from llm_benchmark.pipeline import step_run_benchmarks
from tests.utils.langfuse_test_utils import LangfuseTestHelper, create_test_langfuse_config


@pytest.fixture
def langfuse_helper():
    """Fixture providing Langfuse test helper."""
    return LangfuseTestHelper()


@pytest.fixture
def mock_langfuse_config():
    """Fixture providing mock Langfuse configuration."""
    return create_test_langfuse_config()


@pytest.fixture
def mock_model_config():
    """Fixture providing mock model configuration."""
    from llm_benchmark.config_loader import ModelEntry, ModelsConfig

    return ModelsConfig(
        models=[
            ModelEntry(
                id="test-model-1",
                provider="test_provider",
            ),
            ModelEntry(
                id="test-model-2",
                provider="test_provider",
            ),
        ]
    )


class MockModelClient:
    """Mock model client for testing."""

    async def complete(self, prompt: str, response_type):
        """Mock completion returning predictable response."""
        response = MagicMock()
        response.content = f"Mock response for: {prompt[:50]}..."
        response.prompt_tokens = 10
        response.completion_tokens = 5
        response.latency_ms = 100
        return response


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="Work in progress / unstable")
async def test_step_run_benchmarks_integration(mock_dataset, langfuse_helper, tmp_path):
    """Test step_run_benchmarks with all external dependencies properly mocked."""

    # Create real dataset file in temporary directory
    from llm_benchmark.utils.loader import save_jsonl

    dataset_file = tmp_path / "latest_samples.jsonl"
    save_jsonl(mock_dataset.samples, dataset_file)

    mock_langfuse = MagicMock()

    mock_root_span = MagicMock()
    mock_observation = MagicMock()
    mock_observation.id = "test_obs_id"
    mock_langfuse.start_as_current_observation.return_value.__enter__.return_value = mock_root_span
    mock_root_span.start_as_current_observation.return_value.__enter__.return_value = (
        mock_observation
    )

    mock_get_langfuse_client = MagicMock()
    mock_get_langfuse_client.return_value = LangfuseClient(create_test_langfuse_config())

    with (
        patch(
            "llm_benchmark.benchmark.runner.get_langfuse_client",
            return_value=mock_get_langfuse_client,
        ),
        patch("langfuse.get_client", return_value=mock_langfuse),
        patch("llm_benchmark.models.litellm_factory.LLMClientFactory") as mock_llmclient_factory,
    ):
        from llm_benchmark.config_loader import load_models_config

        model_config = load_models_config()
        models = model_config.models
        tasks = task_factory.get_all_tasks()

        mock_llmclient_factory.get_client.return_value = MockModelClient()

        mock_trace = MagicMock()
        mock_trace.id = "test_trace_id"
        mock_langfuse.api.trace.list.return_value.data = [mock_trace] * (len(models) * len(tasks))
        mock_langfuse.api.observations.get_many.return_value = [mock_trace] * (
            len(models)
            * sum([(len(mock_dataset.samples) + t.batch_size - 1) // t.batch_size for t in tasks])
        )

        # benchmark_runner = BenchmarkRunner(langfuse=mock_langfuse)

        def mock_update(input=None, output=None, usage_details=None):
            # Store trace data for verification using the real execution flow
            if hasattr(mock_observation, "_current_context"):
                model_id, task_id, experiment_id, sample_id = mock_observation._current_context
                task_name = task_id  # Use actual task_id from real execution

                langfuse_helper.created_traces[mock_observation.id] = {
                    "model_id": model_id,
                    "task_id": task_name,
                    "experiment_id": experiment_id,
                    "sample_id": sample_id,
                    "prompt": input,
                    "completion": output,
                    "prompt_tokens": usage_details.get("input", 10) if usage_details else 10,
                    "completion_tokens": usage_details.get("output", 5) if usage_details else 5,
                    "latency_ms": 100,
                }

        def mock_score(name, value, comment):
            langfuse_helper.created_scores[mock_observation.id] = {
                name: {"value": value, "comment": comment}
            }

        mock_observation.update = mock_update
        mock_observation.score = mock_score

        # Store context for trace creation
        def capture_context(**kwargs):
            mock_observation._current_context = (
                kwargs.get("model", "mock_model"),
                kwargs.get("metadata", {}).get("task_id", "unknown_task"),
                "test_experiment",
                "mock_sample",
            )

        mock_root_span.start_as_current_observation.side_effect = capture_context

        # Execute the function under test
        result = await step_run_benchmarks(tmp_path, use_dataset_cache=True)

        # Verify return value
        assert result == "success"

        # Verify model clients were created for each model
        assert mock_llmclient_factory.get_client.call_count == len(models)

        # Verify Langfuse client was initialized
        mock_langfuse.assert_called_once()

        # Verify traces were created for each sample and model combination
        expected_trace_count = len(mock_dataset.samples) * len(models) * 2  # 2 tasks
        assert langfuse_helper.get_trace_count() == expected_trace_count

        # Verify traces have correct metadata
        for model in models:
            for sample in mock_dataset.samples:
                for task in tasks:
                    # Find the corresponding trace (order may vary)
                    matching_traces = [
                        trace_id
                        for trace_id, metadata in langfuse_helper.created_traces.items()
                        if (
                            metadata["model_id"] == model.id
                            and metadata["sample_id"] == sample.id
                            and metadata["task_id"] == task.task_id
                        )
                    ]

                    assert len(matching_traces) == 1, (
                        f"Expected 1 trace for model={model.id}, sample={sample.id}, task={task.task_id}"
                    )

                    trace_id = matching_traces[0]
                    trace_metadata = langfuse_helper.get_trace_metadata(trace_id)

                    # Verify trace metadata
                    assert trace_metadata["experiment_id"] == "test-experiment-123"
                    assert trace_metadata["task_id"] == task.task_id
                    assert trace_metadata["model_id"] == model.id

                    # Verify scores were created
                    assert langfuse_helper.verify_score_exists(trace_id, "accuracy")
                    accuracy_score = langfuse_helper.get_score_value(trace_id, "accuracy")
                    assert isinstance(accuracy_score, float)
                    assert 0.0 <= accuracy_score <= 1.0

        # Verify Langfuse flush was called
        mock_langfuse.flush.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_step_run_benchmarks_with_empty_dataset(langfuse_helper, tmp_path):
    """Test step_run_benchmarks behavior with empty dataset."""

    # Create empty dataset file in temporary directory
    from llm_benchmark.utils.loader import save_jsonl

    dataset_file = tmp_path / "latest_samples.jsonl"
    save_jsonl([], dataset_file)

    # Use same clean mocking approach as first test
    mock_langfuse = MagicMock()

    mock_root_span = MagicMock()
    mock_observation = MagicMock()
    mock_observation.id = "empty_test_obs_id"
    mock_langfuse.start_as_current_observation.return_value.__enter__.return_value = mock_root_span
    mock_root_span.start_as_current_observation.return_value.__enter__.return_value = (
        mock_observation
    )

    mock_get_langfuse_client = MagicMock()
    mock_get_langfuse_client.return_value = LangfuseClient(create_test_langfuse_config())

    with (
        patch(
            "llm_benchmark.benchmark.runner.get_langfuse_client",
            return_value=mock_get_langfuse_client,
        ),
        patch("langfuse.get_client", return_value=mock_langfuse),
        patch("llm_benchmark.models.litellm_factory.LLMClientFactory") as mock_llmclient_factory,
        # Mock load_latest_samples to use created sample list instead of real HF repo
        patch(
            "llm_benchmark.pipeline.dataset_loader.load_latest_samples"
        ) as mock_load_latest_samples,
    ):
        # Configure load_latest_samples to return empty list for empty dataset test
        mock_load_latest_samples.return_value = []
        mock_llmclient_factory.get_client.return_value = MockModelClient()

        # Mock Langfuse.api.trace.list() to return empty traces for empty dataset test
        mock_langfuse.api.trace.list.return_value.data = []

        # Execute the function
        result = await step_run_benchmarks(tmp_path)

        # Verify successful execution
        assert result == "success"

        # Verify no traces were created for empty dataset
        assert langfuse_helper.get_trace_count() == 0

        # Verify flush was called (BenchmarkRunner always calls flush when it has a LangfuseClient)
        mock_langfuse.flush.assert_called_once()
