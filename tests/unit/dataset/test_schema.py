# Unit tests for dataset schema (subscription-email Sample, Message, SubscriptionEventType)

from datasets_shared.schema import (
    TASK_A_CATEGORIZE,
    TASK_B_EXTRACT,
    Dataset,
    Header,
    Payload,
    RawGmailMessage,
    Sample,
    SubscriptionEventType,
)
from tests.utils.mock_dataset import create_mock_dataset


def test_sample_from_example_json() -> None:
    """Parse user example JSON (camelCase) and check get_prompt / get_expected."""
    raw = {
        "id": "sample-id",
        "companyId": "netflix",
        "templateId": "netflix-monthly-payment",
        "snippet": "Your Netflix subscription has been renewed for $15.99. Next billing date is Feb 15, 2024.",
        "subject": "Your monthly subscription has been renewed",
        "subscription_event_type": "MONTHLY_PAYMENT",
    }
    sample = Sample.model_validate(raw)
    assert sample.subscription_event_type == SubscriptionEventType.MONTHLY_PAYMENT
    assert sample.snippet.startswith("Your Netflix")
    assert sample.subject == "Your monthly subscription has been renewed"
    assert sample.company_id == "netflix"

    # Task A: categorize
    prompt_a = sample.get_prompt(TASK_A_CATEGORIZE)
    assert "Netflix" in prompt_a and "subscription has been renewed" in prompt_a
    assert sample.get_expected(TASK_A_CATEGORIZE) == "MONTHLY_PAYMENT"

    # Task B: extract
    prompt_b = sample.get_prompt(TASK_B_EXTRACT)
    assert "Subject:" in prompt_b and "Snippet:" in prompt_b


def test_message_get_header() -> None:
    msg = RawGmailMessage(
        # internal_date="1705319400000",
        snippet="Test",
        payload=Payload(
            headers=[
                Header(name="From", value="a@b.com"),
                Header(name="Subject", value="Hello"),
            ]
        ),
    )
    assert msg.get_header("Subject") == "Hello"
    assert msg.get_header("subject") == "Hello"
    assert msg.subject == "Hello"
    assert msg.from_address == "a@b.com"


def test_dataset_roundtrip() -> None:
    """Tests that a mock Dataset can be serialized and deserialized."""
    dataset = create_mock_dataset(num_samples=1)
    js = dataset.model_dump_json()
    loaded = Dataset.model_validate_json(js)
    assert len(loaded.samples) == 1
    assert loaded.samples[0].subscription_event_type == SubscriptionEventType.MONTHLY_PAYMENT
    assert loaded.content_hash == "mock_hash"
