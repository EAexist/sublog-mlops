"""Factory for creating and caching LLM clients."""

from typing import ClassVar

from llm_benchmark.models.litellm_client import LiteLLMClient


class LLMClientFactory:
    """Factory for creating and caching LiteLLM clients."""
    _clients: ClassVar[dict[str, "LiteLLMClient"]] = {}

    @classmethod
    def get_client(cls, model_id: str, api_key: str | None = None) -> "LiteLLMClient":
        """Get or create a cached client for the given model_id."""
        if model_id not in cls._clients:
            cls._clients[model_id] = LiteLLMClient(model_id, api_key)
        return cls._clients[model_id]
