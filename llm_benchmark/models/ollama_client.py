# Local models via Ollama

import logging
from typing import Any

from llm_benchmark.models.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Ollama local API client."""

    def __init__(
        self, model_id: str, base_url: str = "http://localhost:11434", **kwargs: object
    ) -> None:
        self.model_id = model_id
        self._base_url = base_url

    async def complete(
        self, prompt: str, response_model: type | None = None, config: dict[str, Any] | None = None
    ) -> LLMResponse[Any]:
        """Call Ollama completion API."""
        # TODO: use httpx to call Ollama, return LLMResponse
        return LLMResponse[Any](
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
        )
