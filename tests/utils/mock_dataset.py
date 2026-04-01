import uuid

from datasets_shared.schema import (
    Dataset,
    Sample,
    SubscriptionEventType,
)


def create_mock_dataset(num_samples: int = 1) -> Dataset:
    """
    Creates a mock Dataset object for testing purposes.
    """
    samples: list[Sample] = []
    for i in range(num_samples):
        subject = f"Test Subject {i}"
        snippet = f"Test Snippet {i}"

        sample = Sample(
            id=str(uuid.uuid4()),
            company_id="test_company",
            template_id="test_template",
            subject=subject,
            snippet=snippet,
            subscription_event_type=SubscriptionEventType.MONTHLY_PAYMENT,
            metadata={
                "test_run": True,
                "subject_regex": f"^{subject}$",
                "snippet_regex": f"^{snippet}$",
                "from_address": "test@example.com",
            },
        )
        samples.append(sample)

    return Dataset(samples=samples, content_hash="mock_hash")
