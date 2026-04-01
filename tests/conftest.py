# Pytest fixtures and config

import logging
import os

import litellm
import pytest
from tests.utils.test_llm_logger import TestLLMLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

litellm.success_callback = ["input_output_logging"]
litellm.failure_callback = ["input_output_logging"]
litellm.log_raw_request_response = True
litellm.callbacks = [TestLLMLogger()]

os.environ['LITELLM_LOG'] = 'DEBUG'

@pytest.fixture
def sample_models_config_path(tmp_path):
    """Path to a minimal models.yaml."""
    p = tmp_path / "models.yaml"
    p.write_text("""
models:
  - id: stub
    provider: openai
    model_string: gpt-4o
    input_price_per_1k_tokens: 0.0
    output_price_per_1k_tokens: 0.0
""")
    return p
