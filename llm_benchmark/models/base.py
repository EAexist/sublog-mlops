# Abstract LLMClient + FineTunableModel mixin (stub)

from abc import ABC, abstractmethod
from pydantic import BaseModel

from llm_benchmark.dataset.schema import Dataset


class LLMResponse(BaseModel):
    """Response from an LLM call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient(ABC):
    """Abstract base for all LLM clients."""

    @abstractmethod
    async def complete(self, prompt: str) -> LLMResponse:
        """Return completion for the given prompt."""
        ...


class FineTunableModel:
    """Stub mixin: fine_tune(dataset) not implemented."""

    def fine_tune(self, dataset: Dataset) -> str:
        """Return new model id after fine-tuning. Stub — do not refactor away."""
        raise NotImplementedError("FineTunableModel.fine_tune is not implemented")
