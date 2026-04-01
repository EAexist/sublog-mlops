# Abstract LLMClient + FineTunableModel mixin (stub)

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar

from datasets_shared.schema import Dataset
from pydantic import BaseModel


class ResponseFormat(Enum):
    """Response format types for LLM calls."""
    TEXT = "text"
    JSON = "json"


T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel, Generic[T]):
    """Response from an LLM call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    parsed_data: T | None = None  # Parsed data when response_model is used


class LLMClient(ABC):
    """Abstract base for all LLM clients."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        response_model: type[T],
        config: dict[str, Any] | None = None
    ) -> LLMResponse[T]:
        """Return completion for the given prompt."""
        ...


class FineTunableModel:
    """Stub mixin: fine_tune(dataset) not implemented."""

    def fine_tune(self, dataset: Dataset) -> str:
        """Return new model id after fine-tuning. Stub — do not refactor away."""
        raise NotImplementedError("FineTunableModel.fine_tune is not implemented")
