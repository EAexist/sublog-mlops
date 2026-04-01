# Google Gemini LLM client

import logging
from typing import Any

from llm_benchmark.models.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model_string: str, api_key: str | None = None, **kwargs: object) -> None:
        self.model_string = model_string
        self._api_key = api_key

    async def complete(self, prompt: str, response_model: type | None = None, config: dict[str, Any] | None = None) -> LLMResponse[Any]:
        """Call Gemini completion API."""
        # TODO: use google-generativeai, return LLMResponse
        return LLMResponse[Any](
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
        )
