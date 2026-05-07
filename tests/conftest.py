# Pytest fixtures and config

import logging
import os
import uuid

import litellm
import pytest
from datasets_shared.schema import Dataset, Sample, SubscriptionEventType
from tests.utils.test_llm_logger import TestLLMLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

litellm.success_callback = ["input_output_logging"]
litellm.failure_callback = ["input_output_logging"]
litellm.log_raw_request_response = True
litellm.callbacks = [TestLLMLogger()]

os.environ["LITELLM_LOG"] = "DEBUG"


@pytest.fixture
def sample_models_config_path(tmp_path):
    """Path to a minimal models.yaml."""
    p = tmp_path / "models.yaml"
    p.write_text("""
models:
  - id: stub
    provider: openai
    model_id: gpt-4o
    input_price_per_1k_tokens: 0.0
    output_price_per_1k_tokens: 0.0
""")
    return p


@pytest.fixture
def mock_single_sample():
    """Create a single mock Sample for testing."""
    return Sample(
        id=str(uuid.uuid4()),
        company_id="test_company",
        template_id="test_template",
        subject="Test Subject",
        snippet="Test Snippet",
        subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
    )


@pytest.fixture
def mock_dataset_single_sample(mock_single_sample):
    """Create a mock Dataset with one sample for testing."""
    return Dataset(samples=[mock_single_sample], content_hash="mock_hash")


@pytest.fixture
def mock_samples_dataset():
    """Create a mock Dataset with multiple samples for testing."""
    samples = []
    for i in range(3):
        sample = Sample(
            id=str(uuid.uuid4()),
            company_id="test_company",
            template_id="test_template",
            subject=f"Test Subject {i}",
            snippet=f"Test Snippet {i}",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
        )
        samples.append(sample)

    return Dataset(samples=samples, content_hash="mock_hash")


@pytest.fixture
def mock_samples():
    """Create realistic mock samples with different SubscriptionEventType values."""
    samples = [
        Sample(
            id=str(uuid.uuid4()),
            company_id="netflix",
            template_id="netflix-monthly-payment",
            subject="Your Netflix subscription has been renewed",
            snippet="Your Netflix subscription has been renewed for $15.99. Next billing date is Feb 15, 2024.",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
        ),
        Sample(
            id=str(uuid.uuid4()),
            company_id="spotify",
            template_id="spotify-annual-payment",
            subject="Spotify Premium - Annual Payment Processed",
            snippet="Your Spotify Premium annual subscription of $99.90 has been successfully charged to your Visa card ending in 4242.",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
        ),
        Sample(
            id=str(uuid.uuid4()),
            company_id="adobe",
            template_id="adobe-subscription-start",
            subject="Welcome to Adobe Creative Cloud!",
            snippet="Your Adobe Creative Cloud subscription has started. You now have access to all Adobe apps and services.",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
        ),
        Sample(
            id=str(uuid.uuid4()),
            company_id="gym",
            template_id="gym-membership-cancel",
            subject="Gym Membership Cancellation Confirmed",
            snippet="Your gym membership has been cancelled as requested. Your access will end on March 31, 2024.",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_CANCEL,
        ),
        Sample(
            id=str(uuid.uuid4()),
            company_id="amazon",
            template_id="amazon-prime-monthly",
            subject="Amazon Prime membership renewal",
            snippet="Your Amazon Prime membership has been automatically renewed for $14.99. Enjoy free shipping, Prime Video, and more benefits.",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
        ),
    ]
    return samples


@pytest.fixture
def mock_dataset(mock_samples):
    """Create a realistic mock Dataset with diverse subscription event types."""
    return Dataset(samples=mock_samples, content_hash="realistic_mock_hash")


def create_mock_dataset(num_samples: int = 1) -> Dataset:
    """
    Creates a mock Dataset object for testing purposes.

    Args:
        num_samples: Number of samples to create in the dataset

    Returns:
        Dataset with mock samples
    """
    samples = []
    for i in range(num_samples):
        sample = Sample(
            id=str(uuid.uuid4()),
            company_id="test_company",
            template_id="test_template",
            subject=f"Test Subject {i}",
            snippet=f"Test Snippet {i}",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
        )
        samples.append(sample)

    return Dataset(samples=samples, content_hash="mock_hash")


def create_mock_samples_for_dataset_generation(company_id: str = "netflix") -> list[Sample]:
    """
    Creates mock samples covering all SubscriptionEventType values for dataset generation testing.

    Args:
        company_id: Company ID to use for all samples (default: "netflix")

    Returns:
        List of Sample objects with all subscription event types
    """
    samples = []
    sample_id = 1

    for event_type in SubscriptionEventType:
        sample = Sample(
            id=f"mock_sample_{sample_id:03d}",
            company_id=company_id,
            template_id=f"template_{event_type.value}",
            subject=f"Mock subject for {event_type.value}",
            snippet=f"Mock snippet for {event_type.value}",
            subscription_event_type=event_type,
            metadata={"test": True},
        )
        samples.append(sample)
        sample_id += 1

    return samples


def create_mock_oracle_responses() -> dict[str, str]:
    """
    Creates mock oracle responses for template and sample generation.

    Returns:
        Dictionary with mock templates and samples for each subscription event type
    """
    return {
        "templates": {
            "SUBSCRIPTION_START_OR_PAYMENT": "Welcome {{client_name}}! Your subscription to {{service_name}} has started on {{date}} for ${{payment_amount}}.",
            "SUBSCRIPTION_END": "Goodbye {{client_name}}! Your subscription to {{service_name}} ended on {{date}}.",
            "PAYMENT_FAILED": "Payment failed for {{client_name}}! Amount ${{payment_amount}} due on {{date}}.",
            "PLAN_CHANGE": "Plan updated for {{client_name}}! New {{service_name}} plan active from {{date}}.",
            "SUBSCRIPTION_RENEWAL": "Your {{service_name}} subscription renewed on {{date}} for ${{payment_amount}}.",
        },
        "samples": {
            "SUBSCRIPTION_START_OR_PAYMENT": "Welcome John Smith! Your subscription to Netflix Premium has started on April 12, 2026 for $15.99.",
            "SUBSCRIPTION_END": "Goodbye Mike Wilson! Your subscription to Netflix Basic ended on April 12, 2026.",
            "PAYMENT_FAILED": "Payment failed for Alex Brown! Amount $15.99 due on April 12, 2026.",
            "PLAN_CHANGE": "Plan updated for Chris Lee! New Netflix Premium plan active from April 12, 2026.",
            "SUBSCRIPTION_RENEWAL": "Your Netflix subscription renewed on April 12, 2026 for $15.99.",
        },
    }
