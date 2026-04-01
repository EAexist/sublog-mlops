import os

import litellm
from tests.utils.test_llm_logger import TestLLMLogger


def setup_litellm_logging():
    """Configures LiteLLM with custom callbacks and verbose logging."""

    custom_logger = TestLLMLogger()
    litellm.callbacks = [custom_logger]

    # 4. Force Environment Variable for the internal logic
    os.environ["LITELLM_LOG"] = "DEBUG"

    print("🚀 LiteLLM Debugging & Custom Callbacks Enabled")
