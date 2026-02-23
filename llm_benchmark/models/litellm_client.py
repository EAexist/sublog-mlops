import time
from typing import Any, Dict, Optional

import litellm  # type: ignore
from llm_benchmark.models.base import LLMClient, LLMResponse


class LiteLLMClient(LLMClient):
    """
    Unified client using LiteLLM to support OpenAI, Anthropic, Gemini, Ollama, etc.
    """

    def __init__(self, provider: str, model_string: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model_string = model_string
        self.api_key = api_key

    async def complete(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Wraps litellm.acompletion to return the standardized LLMResponse.
        """
        config = config or {}
        start_time = time.perf_counter()

        # Map 'json_mode' from config to LiteLLM specific format if needed
        response_format = {"type": "json_object"} if config.get("json_mode") else None
        
        # LiteLLM handles the provider logic based on the model string usually,
        # but we can pass specific args if needed.
        response = await litellm.acompletion(
            model=self.model_string,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.api_key,
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens"),
            response_format=response_format,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract usage
        usage = response.usage or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        content = response.choices[0].message.content

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )