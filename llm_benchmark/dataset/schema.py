# Pydantic: subscription-email Sample (message + truth labels), Dataset, Task, PromptVersion (stub)

import json
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# --- Task type identifiers (used by get_prompt / get_expected) ---
TASK_A_CATEGORIZE = "task_a"  # categorize message -> SubscriptionEventType
TASK_B_EXTRACT = "task_b"     # extract subject_regex, snippet_regex from long text


class SubscriptionEventType(str, Enum):
    """Ground-truth label for Task A: model categorizes each sample into one of these."""
    MONTHLY_PAYMENT = "MONTHLY_PAYMENT"
    ANNUAL_PAYMENT = "ANNUAL_PAYMENT"
    SUBSCRIPTION_START = "SUBSCRIPTION_START"
    SUBSCRIPTION_CANCEL = "SUBSCRIPTION_CANCEL"
    NOT_A_SUBSCRIPTION_EMAIL = "NOT_A_SUBSCRIPTION_EMAIL"


class Header(BaseModel):
    """Single email header."""
    name: str
    value: str


class Payload(BaseModel):
    """Email payload with headers (e.g. From, Subject)."""
    headers: list[Header] = Field(default_factory=list)


class Message(BaseModel):
    """Email-like message: input to the model. id can be set later (e.g. from index)."""
    model_config = {"populate_by_name": True}
    id: str | None = None
    # internal_date: str = Field(default="", alias="internalDate")  # e.g. "1705319400000"
    snippet: str = ""
    payload: Payload = Field(default_factory=Payload)

    def get_header(self, name: str) -> str | None:
        """Return first header value for name (case-insensitive)."""
        for h in self.payload.headers:
            if h.name.lower() == name.lower():
                return h.value
        return None

    @property
    def subject(self) -> str:
        return self.get_header("Subject") or ""

    @property
    def from_address(self) -> str:
        return self.get_header("From") or ""


class Sample(BaseModel):
    """
    Single eval sample: message (model input) + truth labels for Task A and Task B.
    - subscription_event_type: ground truth for Task A (categorization).
    - subject_regex, snippet_regex: ground-truth templates for Task B (extraction).
    """

    model_config = {"populate_by_name": True}
    message: Message
    subscription_event_type: SubscriptionEventType = Field(alias="subscriptionEventType")
    subject_regex: str = Field(default="", alias="subjectRegex")
    snippet_regex: str = Field(default="", alias="snippetRegex")
    metadata: dict = Field(default_factory=dict)

    def get_prompt(self, task_id: str) -> str:
        """
        Prompt string for the given task. Used by runner to call the model.
        - task_a: serialized message (snippet + subject) for categorization.
        - task_b: long text (subject + snippet) for regex extraction.
        """
        if task_id == TASK_B_EXTRACT:
            return f"Subject: {self.message.subject}\n\nSnippet: {self.message.snippet}"
        # task_a or default: full message context for categorization
        return (
            f"From: {self.message.from_address}\n"
            f"Subject: {self.message.subject}\n\n"
            f"Snippet: {self.message.snippet}"
        )

    def get_expected(self, task_id: str) -> str:
        """
        Expected answer as string for comparison with model output.
        - task_a: subscription_event_type value.
        - task_b: JSON with subject_regex and snippet_regex.
        """
        if task_id == TASK_B_EXTRACT:
            return json.dumps({
                "subject_regex": self.subject_regex,
                "snippet_regex": self.snippet_regex,
            }, sort_keys=True)
        return self.subscription_event_type.value

    # Backward compatibility: default to task_a for .prompt / .expected
    @property
    def prompt(self) -> str:
        return self.get_prompt(TASK_A_CATEGORIZE)

    @property
    def expected(self) -> str:
        return self.get_expected(TASK_A_CATEGORIZE)


class Dataset(BaseModel):
    """Collection of samples with content hash (one dataset per task)."""
    samples: list[Sample]
    content_hash: str = ""


class Task(BaseModel):
    """Benchmark task = one prompt/domain. Each task is benchmarked independently."""
    task_id: str
    n_samples: int
    # Optional: "task_a" | "task_b" to override which prompt/expected view to use
    task_type: str = Field(default=TASK_A_CATEGORIZE, description="task_a (categorize) or task_b (extract)")


class PromptVersion(BaseModel):
    """Stub: versioned prompt schema. Only one is_active at a time."""
    id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = False
