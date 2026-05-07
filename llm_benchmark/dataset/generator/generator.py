# Calls oracle model, returns Dataset per task

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from datasets_shared.schema import (
    Dataset,
    Sample,
)
from datasets_shared.schema.models import EmailTemplate, EmailTextParameterSet
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _create_version_dir(base_path: Path) -> Path:
    """Create versioned directory for this run."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    version_dir = base_path / "versions" / f"{date_str}"
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


async def _call_oracle(
    prompt: str,
    model_id: str,
    response_model: type[T],
    generation_config: dict[str, Any] | None = None,
) -> T:
    """
    Calls the oracle model using LiteLLMClient.

    Args:
        generation_config: Dict containing params like temperature, max_tokens, json_mode.
        response_model: Pydantic model class for structured response parsing.
    """
    from llm_benchmark.models.litellm_factory import LLMClientFactory

    config = generation_config or {}
    logger.info(f"Calling oracle {model_id} (config={config}) with prompt length {len(prompt)}")
    logger.debug(f"{prompt}")

    # Get cached client and make synchronous call
    client = LLMClientFactory.get_client(model_id)
    response = await client.complete(prompt, response_model, config)

    if response.parsed_data is None:
        raise ValueError(f"Oracle {model_id} failed to return structured data.")

    data = response.parsed_data

    if isinstance(data, dict):
        return response_model(**data)

    return data


def assemble_dataset(
    templates: list[EmailTemplate],
    parameters: list[EmailTextParameterSet],
    n_samples_per_template: int,
) -> Dataset:

    param_iter = iter(parameters)
    samples = []

    for tpl in templates:
        # Generate n_samples_per_template samples for each template
        for _ in range(n_samples_per_template):
            try:
                params = next(param_iter)
                subject, snippet = tpl.subject, tpl.snippet
                param_dict = params.model_dump(by_alias=False)

                # Merge placeholders
                for key, val in param_dict.items():
                    p = f"{{{{{key}}}}}"
                    subject = subject.replace(p, str(val))
                    snippet = snippet.replace(p, str(val))

                sample = Sample(
                    id=str(uuid.uuid4()),
                    subject=subject,
                    snippet=snippet[:200],
                    subscription_event_type=tpl.subscription_event_type,
                    company_id=tpl.company_id,
                    template_id=tpl.id,
                )

                if _validate_sample(sample):
                    samples.append(sample)
            except StopIteration:
                break

    hash_str = json.dumps(sorted([s.id for s in samples if s.id]))
    content_hash = hashlib.md5(hash_str.encode()).hexdigest()
    return Dataset(samples=samples, content_hash=content_hash)


def _validate_sample(sample: Sample) -> bool:
    """
    Basic validation for a generated sample.
    """
    # 1. Must have non-empty subject and snippet
    if not sample.subject or not sample.snippet:
        return False
    # 2. Future: Check for hallucinations (e.g. placeholders like [Your Name] left in text)
    return True
