from pathlib import Path
from typing import Any, cast

from datasets import load_dataset
from pydantic import BaseModel, model_validator

# Centralized path definitions
PROJECT_ROOT = Path(__file__).parent.parent.parent


class CompanyInfo(BaseModel):
    id: str
    name: str
    email: str
    industry: str
    require_shared_email_discriminator: bool = False

    @model_validator(mode="before")
    @classmethod
    def map_nested_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Map 'aliasNames' dict to 'name'
            if "aliasNames" in data:
                # Prioritize EN, fallback to any available, then "Unknown"
                names = data["aliasNames"]
                data["name"] = names.get("EN") or names.get("KO") or "Unknown"

            # 2. Map 'emailAddresses' list to 'email'
            if "emailAddresses" in data and isinstance(data["emailAddresses"], list):
                data["email"] = (
                    data["emailAddresses"][0] if data["emailAddresses"] else "no-reply@example.com"
                )

            # 3. Ensure 'id' and 'industry' exist (even if null in JSON)
            data.setdefault("id", "unknown_id")
            data.setdefault("industry", "General")

        return data


def load_companies() -> list[CompanyInfo]:
    """Load companies from dataset/data/reference/companies.json and convert to CompanyInfo objects."""

    dataset = load_dataset(
        "hyeon-expression/subscription-killer-synthetic-emails",
        data_files="data/reference/companies.json",
        field=None,
        download_mode="force_redownload",
    )

    companies = [CompanyInfo(**cast(dict[str, Any], item)) for item in dataset["train"]]

    # Count email frequencies
    email_counts = {}
    for company in companies:
        email_counts[company.email] = email_counts.get(company.email, 0) + 1

    # Set require_shared_email_discriminator=True for companies with shared emails
    for company in companies:
        if email_counts[company.email] > 1:
            company.require_shared_email_discriminator = True

    return companies


# Load companies from JSON file
COMPANIES = load_companies()
