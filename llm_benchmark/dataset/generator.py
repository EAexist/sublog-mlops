# Calls oracle model, returns Dataset per task

import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from llm_benchmark.config_loader import BenchmarkConfig, load_benchmark_config
from llm_benchmark.dataset.constants import COMPANIES
from llm_benchmark.dataset.schema import Dataset, Header, Message, Payload, Sample, SubscriptionEventType

logger = logging.getLogger(__name__)

# Try to import litellm, handle if missing (though it should be in requirements)
try:
    import litellm
except ImportError:
    litellm = None

# --- Prompt Templates ---
# Best Practice: Externalize prompts to facilitate prompt engineering without changing logic.

# Step i: Generate a few diverse, initial examples for a scenario.
PROMPT_GENERATE_INITIAL_SAMPLES = """
Generate {n_initial_samples} diverse, complete, and realistic email examples for a "{event_type}" event from {company_name}, a company in the {industry} industry.
The examples should be different from each other in phrasing and structure to represent different possible templates.
Ensure the tone is professional yet persuasive.

Output a JSON list of {n_initial_samples} objects, each with 'subject' and 'snippet' keys.
"""

# Step ii: Extract a regex template from a single concrete example.
PROMPT_EXTRACT_REGEX_FROM_SINGLE_SAMPLE = """
Analyze the following single email sample and generate a Python regular expression (regex) for its Subject and Snippet.
The regex should be general enough to allow for variations in details like dates, amounts, or IDs, but specific enough to capture the core structure of this one sample.
Use non-capturing groups `(?:...)` where appropriate.

Email Sample:
{sample_json}

Output a single JSON object with 'subjectRegex' and 'snippetRegex'.
"""

# Step iii: Generate many variations from a single regex template.
PROMPT_GENERATE_VARIATIONS_FROM_REGEX = """
Generate {n_variations} new email subjects and snippets that strictly match the following regex patterns.
The variations should be realistic and diverse in their variable parts (like dates, names, amounts).

Subject Regex: `{subject_regex}`
Snippet Regex: `{snippet_regex}`

Context:
- Company: {company_name}
- Industry: {industry}
- Event Type: {event_type}

Output a JSON list of {n_variations} objects, each with 'subject' and 'snippet' keys.
"""


def _call_oracle(
    prompt: str,
    model_id: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Calls the oracle model using LiteLLM.
    
    Args:
        generation_config: Dict containing params like temperature, max_tokens, json_mode.
    """
    if not litellm:
        raise ImportError("LiteLLM is required for dataset generation.")

    config = generation_config or {}
    logger.info(f"Calling oracle {model_id} (config={config}) with prompt length {len(prompt)}")
    
    # Synchronous call for the generator (Airflow tasks are often sync)
    response = litellm.completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.get("temperature", 0.7),
        response_format={"type": "json_object"} if config.get("json_mode", False) else None,
        # Note: API keys must be set in os.environ by the Airflow Connection/Operator before this runs
    )
    
    return response.choices[0].message.content


def generate_dataset(
    oracle_model_id: str,
    n_templates: int,
    n_samples_per_template: int,
    generation_config: Optional[Dict[str, Any]] = None
) -> Dataset:
    """
    Generate eval dataset for one task using oracle model.
    
    Args:
        oracle_model_id: The ID of the oracle model to use for generation.
        n_templates: Number of initial samples to generate to create templates from (N).
        n_samples_per_template: Number of variations to generate for each extracted template (M).
        generation_config: Configuration for the oracle model (e.g. temperature, json_mode).
        
    Returns:
        Dataset: A collection of generated samples.
    """
    logger.info(
        "Generating dataset with oracle=%s n_templates=%s n_samples_per_template=%s",
        oracle_model_id, n_templates, n_samples_per_template
    )
    
    samples: List[Sample] = []
    # Best Practice: Use higher temperature for dataset generation to ensure diversity
    gen_config = generation_config or {"temperature": 0.8, "json_mode": True} # Default to json_mode
    
    for company in COMPANIES:
        for event_type in SubscriptionEventType:
            # Loop N times to create N distinct "templates" (scenarios)
            # STEP i: Generate a few initial, diverse samples to serve as a base for templates.
            initial_samples_prompt = PROMPT_GENERATE_INITIAL_SAMPLES.format(
                n_initial_samples=n_templates,
                company_name=company.name,
                industry=company.industry,
                event_type=event_type.value
            )
            initial_samples_data = []
            try:
                response = _call_oracle(initial_samples_prompt, oracle_model_id, gen_config)
                initial_samples_data = json.loads(response)
                if not isinstance(initial_samples_data, list):
                    raise ValueError("Expected a list of initial samples from LLM.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse initial samples: {e}. Skipping this batch.")
                continue

            # This loop will perform steps ii and iii for each initial sample
            for initial_sample in initial_samples_data:
                if not isinstance(initial_sample, dict) or not initial_sample.get("subject"):
                    continue

                # STEP ii: For each initial sample, extract a regex template.
                regex_extraction_prompt = PROMPT_EXTRACT_REGEX_FROM_SINGLE_SAMPLE.format(
                    sample_json=json.dumps(initial_sample, indent=2)
                )
                subject_regex = ".*"
                snippet_regex = ".*"
                try:
                    response_regex = _call_oracle(regex_extraction_prompt, oracle_model_id, gen_config)
                    regex_data = json.loads(response_regex)
                    subject_regex = regex_data.get("subjectRegex", ".*")
                    snippet_regex = regex_data.get("snippetRegex", ".*")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse extracted regex: {e}. Using fallback '.*'.")

                # STEP iii: Using the extracted template, regenerate many variations.
                variations_prompt = PROMPT_GENERATE_VARIATIONS_FROM_REGEX.format(
                    n_variations=n_samples_per_template,
                    subject_regex=subject_regex,
                    snippet_regex=snippet_regex,
                    company_name=company.name,
                    industry=company.industry,
                    event_type=event_type.value
                )
                generated_variations_data = []
                try:
                    response_variations = _call_oracle(variations_prompt, oracle_model_id, gen_config)
                    generated_variations_data = json.loads(response_variations)
                    if not isinstance(generated_variations_data, list):
                        raise ValueError("Expected a list of variations from LLM.")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse generated variations: {e}. Skipping template.")
                    continue

                # 4. Validate and Create Samples from the generated variations
                for email_data in generated_variations_data:
                    subject = email_data.get("subject")
                    snippet = email_data.get("snippet")

                    if not subject or not snippet:
                        continue

                    # Validate generated variation against its source regex
                    if not re.fullmatch(subject_regex, subject):
                        logger.warning(f"Generated subject '{subject}' does not match regex '{subject_regex}'. Skipping.")
                        continue
                    if not re.fullmatch(snippet_regex, snippet):
                        logger.warning(f"Generated snippet '{snippet}' does not match regex '{snippet_regex}'. Skipping.")
                        continue

                    message_id = str(uuid.uuid4())

                    # Create the Message object
                    message = Message(
                        id=message_id,
                        snippet=snippet,
                        payload=Payload(
                            headers=[
                                Header(name="From", value=company.email),
                                Header(name="Subject", value=subject),
                            ]
                        )
                    )
                    
                    # Create the Sample object
                    sample = Sample(
                        message=message,
                        subscription_event_type=event_type,
                        subject_regex=subject_regex,
                        snippet_regex=snippet_regex,
                        metadata={
                            "company": company.name,
                            "emailTemplate": {
                                "subjectRegex": subject_regex,
                                "snippetRegex": snippet_regex
                            },
                        }
                    )
                    
                    # Best Practice: Validate immediately.
                    if _validate_sample(sample):
                        samples.append(sample)
                    else:
                        logger.warning(f"Skipping invalid sample {message_id}")

    # Compute content hash
    # Using a hash of the sorted list of message IDs to ensure determinism for the same set of samples
    sample_ids = sorted([s.message.id for s in samples if s.message.id])
    content_hash = hashlib.md5(json.dumps(sample_ids).encode()).hexdigest()
    
    return Dataset(samples=samples, content_hash=content_hash)


def _validate_sample(sample: Sample) -> bool:
    """
    Validates a synthesized sample against quality rules.
    Returns True if valid, False otherwise.
    """
    # 1. Check for empty content
    if not sample.message.snippet or not sample.message.payload.headers:
        return False
    
    # 2. Future: Check for hallucinations (e.g. placeholders like [Your Name] left in text)
    return True


def generate_datasets_for_tasks(
    config: BenchmarkConfig | None = None,
) -> dict[str, Dataset]:
    """Generate one dataset per task from benchmark config. Returns dict[task_id, Dataset]."""
    config = config or load_benchmark_config()
    out: dict[str, Dataset] = {}
    
    # Default N (templates)
    DEFAULT_N_TEMPLATES = 2
    
    for task in config.tasks:
        out[task.task_id] = generate_dataset(
            oracle_model_id=config.oracle_model_id,
            n_templates=DEFAULT_N_TEMPLATES,
            n_samples_per_template=task.n_samples,
            generation_config={"temperature": 0.7, "json_mode": True} # Example config
        )
    return out
