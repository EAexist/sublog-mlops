import pytest
import json
from unittest.mock import MagicMock, patch
from typing import List

from llm_benchmark.dataset.generator import generate_dataset, _call_oracle
from llm_benchmark.dataset.schema import Dataset, Sample
from llm_benchmark.config_loader import BenchmarkConfig, TaskConfig
from tests.utils.mock_dataset import create_mock_dataset


@patch("llm_benchmark.dataset.generator.litellm")
def test_call_oracle(mock_litellm):
    """Test the _call_oracle wrapper ensures litellm is called correctly."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Generated Content"
    mock_litellm.completion.return_value = mock_response

    # Execute
    result = _call_oracle("test prompt", "model-id", {"temperature": 0.5})

    # Assert
    assert result == "Generated Content"
    mock_litellm.completion.assert_called_once()
    _, kwargs = mock_litellm.completion.call_args
    assert kwargs["model"] == "model-id"
    assert kwargs["temperature"] == 0.5
    assert kwargs["messages"] == [{"role": "user", "content": "test prompt"}]


@patch("llm_benchmark.dataset.generator._call_oracle")
def test_generate_dataset_structure(mock_call_oracle):
    """
    Test generate_dataset returns a valid Dataset object.
    We mock _call_oracle to return valid JSON sequences for the 3-step generation process.
    """
    # Define mock responses for the 3 steps
    # 1. Initial samples
    initial_samples = [{"subject": "Init Subj", "snippet": "Init Snip"}]
    # 2. Regex extraction
    regex_data = {"subjectRegex": ".*", "snippetRegex": ".*"}
    # 3. Variations
    variations = [{"subject": "Var Subj 1", "snippet": "Var Snip 1"}, {"subject": "Var Subj 2", "snippet": "Var Snip 2"}]
    
    # We need to return these in order. 
    # The loop runs for each company (2) * each event type (5).
    # Inside the loop:
    #   1 call for initial samples
    #   For each initial sample (1 in this mock):
    #     1 call for regex
    #     1 call for variations
    
    # So for one iteration of the outer loops: 3 calls.
    # We have 2 companies * 5 event types = 10 iterations.
    # Total calls = 30.
    
    # We can use a side_effect function to return based on prompt content, which is more robust.
    def side_effect(prompt, model_id, config):
        if "Generate" in prompt and "diverse, complete" in prompt: # Step 1
            return json.dumps(initial_samples)
        elif "Analyze the following single email" in prompt: # Step 2
            return json.dumps(regex_data)
        elif "Generate" in prompt and "variations" in prompt: # Step 3
            return json.dumps(variations)
        return "{}"

    mock_call_oracle.side_effect = side_effect

    dataset = generate_dataset(
        oracle_model_id="test-oracle",
        n_templates=1,
        n_samples_per_template=2,
        generation_config={"temperature": 0.7}
    )

    assert isinstance(dataset, Dataset)
    assert dataset.content_hash is not None
    # 2 companies * 5 events * 1 template * 2 variations = 20 samples
    assert len(dataset.samples) == 20
    
    # Verify sample structure
    sample = dataset.samples[0]
    assert isinstance(sample, Sample)
    assert sample.message.id is not None
    assert sample.metadata["company"] is not None
    assert len(sample.message.payload.headers) > 0
