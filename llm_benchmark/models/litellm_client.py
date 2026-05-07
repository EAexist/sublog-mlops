import asyncio
import json
import time
from typing import Any, TypeVar

import litellm
from litellm.exceptions import APIConnectionError, APIError, BadRequestError, RateLimitError
from pydantic import (
    BaseModel,
    TypeAdapter,  # type: ignore
)

from llm_benchmark.models.base import LLMClient, LLMResponse

T = TypeVar("T", bound=BaseModel)


class LiteLLMClient(LLMClient):
    """
    Unified client using LiteLLM to support OpenAI, Anthropic, Gemini, Ollama, etc.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    async def enforce_rate_limit(self) -> None:
        """
        Enforce provider-specific rate limits before making API calls.

        For Groq: 30 requests/minute = 2 seconds between requests
        Future implementations can add other providers with their specific limits.
        """
        if "groq" in self.model_id:
            # Groq rate limit: 30 requests/minute
            # 60 seconds / 30 requests = 2 seconds per request
            await asyncio.sleep(2.0)

        # Add other provider rate limits here as needed
        # Example:
        # elif "openai" in self.model_id:
        #     await asyncio.sleep(0.1)  # OpenAI has higher limits

    async def complete(
        self, prompt: str, response_model: type[T], config: dict[str, Any] | None = None
    ) -> LLMResponse[T]:
        """
        Wraps litellm.acompletion to return the standardized LLMResponse.
        """
        config = config or {}
        start_time = time.perf_counter()

        is_groq = "groq" in self.model_id
        actual_format = {"type": "json_object"} if (is_groq) else response_model
        # actual_format = response_model

        try:
            response = await litellm.acompletion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get(
                    "temperature", 1.0 if "gemini-3.1" in self.model_id else 0.0
                ),
                max_tokens=config.get("max_tokens"),
                response_format=actual_format,  # <--- Native Pydantic support
                num_retries=10,
            )
        except BadRequestError as e:
            print(f"\n❌ [400] Logic error on {self.model_id}: {e}")
            try:
                err_json = json.loads(str(e))
                failed = err_json.get("error", {}).get("failed_generation")
                print("\n🧨 FAILED GENERATION:")
                print(failed)
            except Exception:
                pass
                raise
        except RateLimitError as e:
            print(f"\n❌ [429] Rate Limit Hit on {self.model_id}: {e}")
            raise
        except APIConnectionError as e:
            print(f"\n❌ [Connection] Failed to reach provider: {e}")
            raise
        except APIError as e:
            print(f"\n❌ [API Error] Status {getattr(e, 'status_code', 'unknown')}: {e}")
            raise

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract usage
        usage = getattr(response, "usage", None) or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)

        # Handle both streaming and non-streaming responses
        choices = getattr(response, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", "")
        else:
            content = ""

        if response_model:
            try:
                message = choices[0].message if choices else None
                parsed_data = getattr(message, "parsed", None)

                if parsed_data is None and content:
                    # TypeAdapter validates the dict against your response_model
                    parsed_data = TypeAdapter(response_model).validate_json(content)

                return LLMResponse[T](
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    parsed_data=parsed_data,
                )
            except (json.JSONDecodeError, AttributeError) as e:
                raise ValueError(f"Failed to parse structured response: {e}")

        return LLMResponse[Any](
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
