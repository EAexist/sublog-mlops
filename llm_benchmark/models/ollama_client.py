# Local models via Ollama

import logging
from llm_benchmark.models.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Ollama local API client."""

    def __init__(self, model_string: str, base_url: str = "http://localhost:11434", **kwargs: object) -> None:
        self.model_string = model_string
        self._base_url = base_url

    async def complete(self, prompt: str) -> LLMResponse:
        """Call Ollama completion API."""
        # TODO: use httpx to call Ollama, return LLMResponse
        return LLMResponse(
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
        )
