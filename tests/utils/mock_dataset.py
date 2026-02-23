import uuid
from typing import List
from llm_benchmark.dataset.schema import Dataset, Sample, Message, Payload, Header, SubscriptionEventType

def create_mock_dataset(num_samples: int = 1) -> Dataset:
    """
    Creates a mock Dataset object for testing purposes.
    """
    samples: List[Sample] = []
    for i in range(num_samples):
        message_id = str(uuid.uuid4())
        subject = f"Test Subject {i}"
        snippet = f"Test Snippet {i}"
        
        message = Message(
            id=message_id,
            snippet=snippet,
            payload=Payload(
                headers=[
                    Header(name="From", value="test@example.com"),
                    Header(name="Subject", value=subject),
                ]
            )
        )
        
        sample = Sample(
            message=message,
            subscription_event_type=SubscriptionEventType.MONTHLY_PAYMENT,
            subject_regex=f"^{subject}$",
            snippet_regex=f"^{snippet}$",
            metadata={"test_run": True}
        )
        samples.append(sample)
        
    return Dataset(samples=samples, content_hash="mock_hash")
