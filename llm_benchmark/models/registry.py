# Loads models.yaml → instantiates correct client

import logging
from typing import Any

from llm_benchmark.models.base import LLMClient

logger = logging.getLogger(__name__)


def get_client(provider: str, model_string: str, **kwargs: Any) -> LLMClient:
    """Return LLMClient for provider (openai | google | ollama)."""
    if provider == "openai":
        from llm_benchmark.models.openai_client import OpenAIClient
        return OpenAIClient(model_string=model_string, **kwargs)
    if provider == "google":
        from llm_benchmark.models.gemini_client import GeminiClient
        return GeminiClient(model_string=model_string, **kwargs)
    if provider == "ollama":
        from llm_benchmark.models.ollama_client import OllamaClient
        return OllamaClient(model_string=model_string, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")
