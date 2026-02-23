# Unit tests for dataset schema (subscription-email Sample, Message, SubscriptionEventType)

import json
import pytest
from llm_benchmark.dataset.schema import (
    Sample,
    Message,
    Payload,
    Header,
    SubscriptionEventType,
    Dataset,
    TASK_A_CATEGORIZE,
    TASK_B_EXTRACT,
)
from tests.utils.mock_dataset import create_mock_dataset


def test_sample_from_example_json() -> None:
    """Parse user example JSON (camelCase) and check get_prompt / get_expected."""
    raw = {
        "message": {
            "id": None,
            "internalDate": "1705319400000",
            "snippet": "Your Netflix subscription has been renewed for $15.99. Next billing date is Feb 15, 2024.",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Netflix <noreply@netflix.com>"},
                    {"name": "Subject", "value": "Your monthly subscription has been renewed"},
                ]
            },
        },
        "subscriptionEventType": "MONTHLY_PAYMENT",
        "subjectRegex": ".*monthly.*subscription.*renewed.*",
        "snippetRegex": ".*subscription.*renewed.*\\$.*",
    }
    sample = Sample.model_validate(raw)
    assert sample.subscription_event_type == SubscriptionEventType.MONTHLY_PAYMENT
    assert sample.message.snippet.startswith("Your Netflix")
    assert sample.message.subject == "Your monthly subscription has been renewed"
    assert sample.message.from_address == "Netflix <noreply@netflix.com>"
    assert sample.subject_regex == ".*monthly.*subscription.*renewed.*"

    # Task A: categorize
    assert sample.get_expected(TASK_A_CATEGORIZE) == "MONTHLY_PAYMENT"
    prompt_a = sample.get_prompt(TASK_A_CATEGORIZE)
    assert "Netflix" in prompt_a and "subscription has been renewed" in prompt_a

    # Task B: extract
    assert sample.get_expected(TASK_B_EXTRACT) == json.dumps(
        {"subject_regex": sample.subject_regex, "snippet_regex": sample.snippet_regex},
        sort_keys=True,
    )
    prompt_b = sample.get_prompt(TASK_B_EXTRACT)
    assert "Subject:" in prompt_b and "Snippet:" in prompt_b

    # Backward compat
    assert sample.prompt == prompt_a
    assert sample.expected == "MONTHLY_PAYMENT"


def test_message_get_header() -> None:
    msg = Message(
        # internal_date="1705319400000",
        snippet="Test",
        payload=Payload(headers=[
            Header(name="From", value="a@b.com"),
            Header(name="Subject", value="Hello"),
        ]),
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
