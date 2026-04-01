import asyncio
import uuid

from datasets_shared.schema import EmailTemplate, EmailTextParameterSet, SubscriptionEventType
from llm_benchmark.dataset.constants import COMPANIES, CompanyInfo
from pydantic import BaseModel, TypeAdapter


class EmailTemplatePayload(BaseModel):
    subject: str
    snippet: str


class EmailTemplateListPayload(BaseModel):
    """Represents a list of email templates."""

    templates: list[EmailTemplatePayload]


PROMPT_GENERATE_TEMPLATES = """
Generate {count} unique JSON objects for a "{event_type}" email for {company} ({industry}).

### Constraints:
- **Placeholders**: Use ONLY following exact placeholders: {placeholders}.
- **Integrity**: Keep placeholders as single, unbroken words.
- **Style**: No acrostics or using placeholder letters to start words. Each template must be significanlty distinct in pharase and tone.
- **Length**: Subject and Snippet must be at most one sentence.
- **Output Format**: Return ONLY the raw JSON content.
{constraints}

Example:
{{
  "templates": [
    {{"subject": "Welcome", "snippet": "Hello {{client_name}}, your {company} plan started on {{date}} ({{payment_amount}})."}}
  ]
}}
"""

PROMPT_GENERATE_NON_SUBSCRIPTION_TEMPLATES = """
Generate {count} unique and realistic email templates for a email from {company}, a company in the {industry} industry.

Requirements:
- Email MUST NOT BE related to a user's subscription state.
- Wrong content: Subscription Start, Subscription Cancel, Monthly Payment, Annual Payment
- Correct content: Advertisement or Newsletter or Alert not about subscription state or Notice not about subscription state.
- Create exactly {count} templates.
- **Placeholders**: Use ONLY following exact placeholder: {{{{client_name}}}}.
- **Style**: Each template must be significanlty distinct in pharase and tone.
- **Length**: Subject and Snippet must be at most one sentence.
- **Output Format**: Return ONLY the raw JSON content.
{constraints}

Example:
{{
  "templates": [
    {{"subject": "Our New Summer Collection", "snippet": "Hello {{{{client_name}}}}, have you seen our latest updates for this season?"}}
  ]
}}
"""

template_list_adapter = TypeAdapter(list[EmailTemplate])

# company_id = "netflix"
company_id = False


def get_shared_email_discriminator_constraint(company: CompanyInfo) -> str:
    return (
        company.require_shared_email_discriminator
        and f"Subject must contain the word {company.name}"
        or ""
    )


async def generate_templates_deprecated(n_templates: int, oracle_fn) -> list[EmailTemplate]:
    """
    DEPRECATED: Generate templates using parallel asyncio.gather approach.
    Kept for reference - use generate_templates() instead.

    This approach uses asyncio.gather() which can cause API rate limit issues
    due to burst request patterns.
    """
    placeholders = EmailTextParameterSet.placeholders()
    companies = [c for c in COMPANIES if c.id == company_id] if company_id else COMPANIES

    # --- PHASE 1: Generate Templates and Convert to EmailTemplate ---
    async def get_tpl_and_convert(c, et):

        constraints = get_shared_email_discriminator_constraint(c)

        if et == SubscriptionEventType.NOT_A_SUBSCRIPTION_EMAIL:
            prompt = PROMPT_GENERATE_NON_SUBSCRIPTION_TEMPLATES.format(
                count=n_templates, company=c.name, industry=c.industry, constraints=constraints
            )
        else:
            prompt = PROMPT_GENERATE_TEMPLATES.format(
                count=n_templates,
                company=c.name,
                industry=c.industry,
                event_type=et.value,
                placeholders=placeholders,
                constraints=constraints,
            )

        try:
            converted_templates = []
            max_attempts = 3  # Prevent infinite loops
            attempt = 0

            while len(converted_templates) < n_templates and attempt < max_attempts:
                attempt += 1
                print(
                    f"Attempt {attempt} for {c.name} - {et.value} (need {n_templates - len(converted_templates)} more)"
                )

                res = await oracle_fn(prompt, EmailTemplateListPayload)
                if not res or not res.templates:
                    print(f"No templates returned for {c.name} - {et.value} on attempt {attempt}")
                    continue

                seen_subjects = set()  # Track seen subjects for uniqueness
                for tpl in res.templates:
                    try:
                        # Check for duplicate subject
                        if tpl.subject in seen_subjects:
                            print(
                                f"Skipping duplicate subject '{tpl.subject}' for {c.name} - {et.value}"
                            )
                            continue

                        email_template = EmailTemplate(
                            **tpl.model_dump(),
                            id=str(hash(tpl.snippet)),
                            company_id=c.id,
                            subscription_event_type=et,
                        )
                        converted_templates.append(email_template)
                        seen_subjects.add(tpl.subject)  # Add to seen set

                        # Stop if we have enough templates
                        if len(converted_templates) >= n_templates:
                            break
                    except Exception as e:
                        # Invalid template in the templates -> Omit from result.
                        print(f"Failed to convert template for {c.name} - {et.value}: {e}")
                        continue

            print(
                f"Event type: {et.value}, Generated {len(converted_templates)} / {n_templates} templates after {attempt} attempts"
            )
            return converted_templates[:n_templates]  # Return exactly n_templates

        except Exception as e:
            print(f"Template gen failed for {c.name} - {et.value}: {e}")
            return []

    tpl_tasks = [
        get_tpl_and_convert(c, et)
        for c in companies
        for et in SubscriptionEventType
        # if et == SubscriptionEventType.NOT_A_SUBSCRIPTION_EMAIL
    ]

    template_results = await asyncio.gather(*tpl_tasks, return_exceptions=True)

    # Flatten the results
    templates: list[EmailTemplate] = []
    for result in template_results:
        if isinstance(result, BaseException):
            print(f"Unexpected error: {result}")
            continue
        templates.extend(result)

    # Publish with versioning (auto-generates run_id)
    metadata = {
        "n_templates": n_templates,
        "companies": [c.id for c in companies],
        "total_templates": len(templates),
        "event_types": [et.value for et in SubscriptionEventType],
    }

    return templates


async def generate_templates(n_templates: int, oracle_fn) -> list[EmailTemplate]:
    """
    Generate and save templates with versioning using sequential approach.

    This approach processes companies and event types sequentially to avoid
    API rate limit issues caused by burst request patterns.

    Args:
        n_templates: Number of templates per event type
        oracle_fn: Oracle function to call

    Returns:
        List of generated templates
    """
    placeholders = EmailTextParameterSet.placeholders()
    companies = [c for c in COMPANIES if c.id == company_id] if company_id else COMPANIES

    templates: list[EmailTemplate] = []

    # Process companies and events sequentially to avoid rate limits
    for c in companies:
        for et in SubscriptionEventType:
            try:
                print(f"Processing {c.name} - {et.value}...")

                constraints = get_shared_email_discriminator_constraint(c)

                if et == SubscriptionEventType.NOT_A_SUBSCRIPTION_EMAIL:
                    prompt = PROMPT_GENERATE_NON_SUBSCRIPTION_TEMPLATES.format(
                        count=n_templates,
                        company=c.name,
                        industry=c.industry,
                        constraints=constraints,
                    )
                else:
                    prompt = PROMPT_GENERATE_TEMPLATES.format(
                        count=n_templates,
                        company=c.name,
                        industry=c.industry,
                        event_type=et.value,
                        placeholders=placeholders,
                        constraints=constraints,
                    )

                # Generate templates with retry logic
                converted_templates = []
                max_attempts = 3
                attempt = 0

                while len(converted_templates) < n_templates and attempt < max_attempts:
                    attempt += 1
                    print(
                        f"Attempt {attempt} for {c.name} - {et.value} (need {n_templates - len(converted_templates)} more)"
                    )

                    res = await oracle_fn(prompt, EmailTemplateListPayload)
                    if not res or not res.templates:
                        print(
                            f"No templates returned for {c.name} - {et.value} on attempt {attempt}"
                        )
                        continue

                    seen_subjects = set()
                    for tpl in res.templates:
                        try:
                            # Check for duplicate subject
                            if tpl.subject in seen_subjects:
                                print(
                                    f"Skipping duplicate subject '{tpl.subject}' for {c.name} - {et.value}"
                                )
                                continue

                            email_template = EmailTemplate(
                                **tpl.model_dump(),
                                id=str(uuid.uuid4()),
                                company_id=c.id,
                                subscription_event_type=et,
                            )
                            converted_templates.append(email_template)
                            seen_subjects.add(tpl.subject)

                            # Stop if we have enough templates
                            if len(converted_templates) >= n_templates:
                                break
                        except Exception as e:
                            print(f"Failed to convert template for {c.name} - {et.value}: {e}")
                            continue

                    print(
                        f"Event type: {et.value}, Generated {len(converted_templates)} / {n_templates} templates after {attempt} attempts"
                    )

                    # Break if we got enough templates
                    if len(converted_templates) >= n_templates:
                        break

                templates.extend(converted_templates[:n_templates])

                # Rate limiting: simple delay between requests
                # await asyncio.sleep(0.5)  # Adjust based on API rate limits

            except Exception as e:
                print(f"Template gen failed for {c.name} - {et.value}: {e}")
                continue

    # Publish with versioning (auto-generates run_id)
    metadata = {
        "n_templates": n_templates,
        "companies": [c.id for c in companies],
        "total_templates": len(templates),
        "event_types": [et.value for et in SubscriptionEventType],
    }

    return templates


# def load_templates() -> List[EmailTemplate]:
#     """Load templates from the latest version."""
#     try:
#         latest_path = publisher.get_latest_path()
#         templates_file = latest_path / "templates.json"

#         if not templates_file.exists():
#             return []

#         return template_list_adapter.validate_json(templates_file.read_text())
#     except FileNotFoundError:
#         return []
