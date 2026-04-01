"""
Temporary patch utilities for fixing data issues.

These methods are intended for one-time data migration/patching tasks
and should be removed after the patching is complete.
"""

import logging
import uuid

from datasets_shared.schema import EmailTemplate

logger = logging.getLogger(__name__)


def patch_template_ids_with_uuid(templates: list[EmailTemplate]) -> list[EmailTemplate]:
    """
    Temporary patch method to convert all template IDs to new random UUIDs.

    This replaces the problematic hash-based IDs (which can be large negative integers)
    with proper UUID strings to avoid JSON overflow issues.

    Args:
        templates: list of EmailTemplate objects with potentially problematic IDs

    Returns:
        list of EmailTemplate objects with new UUID-based IDs
    """
    logger.info(f"Patching {len(templates)} template IDs with UUIDs...")

    patched_templates = []
    for template in templates:
        # Create a new template with the same data but a new UUID ID
        patched_template = EmailTemplate(
            id=str(uuid.uuid4()),  # Generate new UUID
            subject=template.subject,
            snippet=template.snippet,
            company_id=template.company_id,
            subscription_event_type=template.subscription_event_type,
        )
        patched_templates.append(patched_template)

    logger.info(f"Successfully patched {len(patched_templates)} templates")
    return patched_templates


def patch_template_ids_deterministic(templates: list[EmailTemplate]) -> list[EmailTemplate]:
    """
    Alternative patch method using deterministic UUIDs based on content.

    This generates the same UUID for the same content, which can be useful
    for debugging or when you want reproducible IDs.

    Args:
        templates: list of EmailTemplate objects with potentially problematic IDs

    Returns:
        list of EmailTemplate objects with deterministic UUID-based IDs
    """
    logger.info(f"Patching {len(templates)} template IDs with deterministic UUIDs...")

    patched_templates = []
    for template in templates:
        # Create deterministic UUID based on template content
        content_string = f"{template.subject}|{template.snippet}|{template.company_id}|{template.subscription_event_type.value}"
        deterministic_uuid = uuid.uuid5(uuid.NAMESPACE_URL, content_string)

        patched_template = EmailTemplate(
            id=str(deterministic_uuid),
            subject=template.subject,
            snippet=template.snippet,
            company_id=template.company_id,
            subscription_event_type=template.subscription_event_type,
        )
        patched_templates.append(patched_template)

    logger.info(f"Successfully patched {len(patched_templates)} templates with deterministic UUIDs")
    return patched_templates


# Example usage function for one-time patching
def patch_and_save_templates(
    templates: list[EmailTemplate], output_path: str, use_deterministic: bool = False
) -> None:
    """
    Patch template IDs and save to file.

    Args:
        templates: Original templates with problematic IDs
        output_path: Path to save the patched templates
        use_deterministic: If True, use deterministic UUIDs; otherwise use random UUIDs
    """
    if use_deterministic:
        patched_templates = patch_template_ids_deterministic(templates)
    else:
        patched_templates = patch_template_ids_with_uuid(templates)

    # Save the patched templates
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSONL format (one JSON object per line)
    lines = [template.model_dump_json() for template in patched_templates]
    output_file.write_text("\n".join(lines))

    logger.info(f"Saved {len(patched_templates)} patched templates to {output_path}")
