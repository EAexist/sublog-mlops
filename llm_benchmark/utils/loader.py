import json
import logging
from pathlib import Path
from typing import Any

from datasets_shared.schema import Sample
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)


def save_jsonl(data: Any, path: Path) -> None:
    """Persist dataset to path as JSONL format (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = TypeAdapter(type(data)).dump_python(data, mode="json")

    size = 0

    # Handle different data types for JSONL format
    if isinstance(serializable_data, list):
        # Write each item as a separate JSON line
        lines = [json.dumps(item, separators=(",", ":")) for item in serializable_data]
        content = "\n".join(lines)
        size = len(lines)
    else:
        # For single objects, write as one JSON line
        content = json.dumps(serializable_data, separators=(",", ":"))
        size = 1

    path.write_text(content, encoding="utf-8")
    logger.info("Saved dataset to %s as JSONL (%d lines)", path, size)


def save_json(data: Any, path: Path) -> None:
    """Persist dataset to path (e.g. JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = TypeAdapter(type(data)).dump_python(data, mode="json")

    path.write_text(json.dumps(serializable_data, indent=2), encoding="utf-8")
    logger.info("Saved dataset to %s", path)


def select_test_samples(
    samples: list[Sample],
    n_samples_per_template: int = 10,
    n_companies: int = 8,
    n_templates: int = 2,
) -> list[Sample]:
    """Select samples using specific criteria.

    Args:
        samples: List of Sample objects
        n_samples_per_template: Number of samples to select per template (default: 10, must be < 40)
        n_companies: Number of unique companies to sample from (default: 8, must be < 40)
        n_templates: Number of templates to select per group (default: 4)

    Returns:
        List of samples meeting criteria
    """
    import random
    from collections import defaultdict

    # Set constant random seed for reproducibility
    random.seed(42)

    # Get unique company IDs and select n_companies of them
    unique_companies = list({s.company_id for s in samples})
    if len(unique_companies) > n_companies:
        selected_companies = random.sample(unique_companies, n_companies)
    else:
        selected_companies = unique_companies

    # Filter samples to only include selected companies
    filtered_samples = [s for s in samples if s.company_id in selected_companies]

    # Group samples by company and subscription event type
    groups = defaultdict(list)
    for sample in filtered_samples:
        groups[(sample.company_id, sample.subscription_event_type.value)].append(sample)

    selected_samples = []

    # For each group, select template_ids and samples
    for (_, _), group_samples in groups.items():
        if len(group_samples) >= 100:  # We have at least 100 samples for this combination
            # Select template_ids from this group
            available_template_ids = {s.template_id for s in group_samples}
            selected_template_ids = list(available_template_ids)[
                :n_templates
            ]  # Select up to n_templates template_ids

            for template_id in selected_template_ids:
                matching_samples = [s for s in group_samples if s.template_id == template_id]
                if matching_samples:
                    selected_samples.extend(
                        random.sample(
                            matching_samples, min(n_samples_per_template, len(matching_samples))
                        )
                    )

    logger.info(
        "Selected %d samples from %d samples (using %d companies)",
        len(selected_samples),
        len(filtered_samples),
        len(selected_companies),
    )

    # Shuffle the selected samples for randomness
    random.shuffle(selected_samples)

    return selected_samples


def read_jsonl(path: Path, return_type: type = list[dict[str, Any]]) -> Any:
    """Read JSONL file and return list of specified type.

    Args:
        path: Path to JSONL file
        return_type: Type to return (e.g., list[dict], Sample, etc.)

    Returns:
        List of objects of specified type, one per line
    """
    path = Path(path)

    if not path.exists():
        logger.warning("JSONL file does not exist: %s", path)
        return []

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    data = []

    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                parsed_data = json.loads(line)
                # Convert to specified return type
                if return_type == list[dict[str, Any]]:
                    data.append(parsed_data)
                else:
                    # For non-dict types, use the type as a constructor
                    data.append(return_type(**parsed_data))
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON line: %s. Error: %s", line, e)
                continue

    logger.info("Loaded %d records from %s", len(data), path)
    return data
