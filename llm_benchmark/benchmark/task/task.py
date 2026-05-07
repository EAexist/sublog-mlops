"""
Task definitions for benchmarking email categorization and template extraction.
Provides base class and specific task implementations with configurable parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets_shared.schema import EmailTemplate, Sample, SubscriptionEventType
from llm_benchmark.benchmark.prompt_registry import prompt_loader
from llm_benchmark.benchmark.task.models import RESPONSE_MODEL_REGISTRY
from pydantic import BaseModel, field_validator


@dataclass
class Score:
    """Score object for task evaluation results."""

    name: str
    value: float
    comment: str


class BaseTask(ABC, BaseModel):
    """
    Base class for all benchmark tasks.

    Attributes:
        name: Human-readable task name
        task_id: Unique identifier for the task
        description: Detailed description of what the task does
        batch_size: Number of samples to process in each batch
    """

    name: str
    task_id: str
    description: str
    batch_size: int

    response_model: type[BaseModel]

    @field_validator("response_model", mode="before")
    @classmethod
    def map_string_to_class(cls, v):
        if isinstance(v, str):
            if v not in RESPONSE_MODEL_REGISTRY:
                raise ValueError(f"Unknown response_model: {v}")
            return RESPONSE_MODEL_REGISTRY[v]
        return v

    @abstractmethod
    def get_prompt(self, samples: list[Sample]) -> str:
        """
        Generate prompt for the given samples.

        Args:
            samples: List of samples to generate prompt for

        Returns:
            Generated prompt string
        """
        pass

    @abstractmethod
    def _get_scores(self, ai_output: str, samples: list[Sample], **kwargs) -> list[Score]:
        """
        Compute accuracy scores for the AI output against the expected result.

        Args:
            ai_output: The model's response
            sample: The original sample with expected answer/context

        Returns:
            List of Score objects with name, value, and comment
        """
        pass

    @property
    @abstractmethod
    def evaluation_comment(self) -> str:
        """
        Get the comment to use for Langfuse evaluation.

        Returns:
            Evaluation comment string
        """
        pass


class EmailCategorizationTask(BaseTask):
    """
    Task for categorizing subscription emails into event types.
    """

    def get_prompt(self, samples: list[Sample]) -> str:
        """
        Generate prompt using the samples formatted as ID|Subject|Snippet|Dates for categorization.

        Args:
            samples: List of samples to generate prompt for

        Returns:
            Generated prompt string
        """
        # Format samples as ID|Subject|Snippet|Dates (dates only if recurring)
        formatted_emails = []
        for idx, sample in enumerate(samples):
            # Format: ID|Subject|Snippet|Dates(only if the message is recurring)
            # For now, we'll use the sample index as ID and empty dates since we don't have date info
            email_line = f"{idx}|{sample.subject}|{sample.snippet}"
            formatted_emails.append(email_line)

        emails = "\n".join(formatted_emails)

        return prompt_loader.get_latest_prompt_by_task_id(self.task_id, {"emails": emails})

    def _get_scores(self, ai_output: str, samples: list[Sample], **kwargs) -> list[Score]:
        """
        Compute score for email categorization task.

        Args:
            ai_output: The model's response in JSON format with categorized indices
            samples: List of samples with expected subscription event types

        Returns:
            List of Score objects with accuracy score
        """
        import json

        # Create mapping from category key to SubscriptionEventType
        key_to_type = {
            "S": SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            "C": SubscriptionEventType.SUBSCRIPTION_CANCEL,
        }

        try:
            ai_result = json.loads(ai_output)

            correct_count = 0
            # Iterate through samples and check correctness of each
            for i, sample in enumerate(samples):
                expected_type = sample.subscription_event_type
                predicted_types = []

                # Check if this sample's index appears in the correct category
                for key, index_list in ai_result.items():
                    if key not in key_to_type:
                        continue
                    if i in index_list:
                        predicted_types.append(key_to_type[key])

                if len(predicted_types) > 1:
                    continue

                predicted_type = (
                    predicted_types[0]
                    if len(predicted_types) > 0
                    else SubscriptionEventType.NOT_A_SUBSCRIPTION_EMAIL
                )
                if predicted_type == expected_type:
                    correct_count += 1

            return [Score(name="accuracy", value=float(correct_count), comment="accuracy")]

        except (json.JSONDecodeError, KeyError, TypeError):
            # If AI output is malformed, return 0
            return [Score(name="accuracy", value=0.0, comment="accuracy")]

    @property
    def evaluation_comment(self) -> str:
        """Get evaluation comment for Langfuse."""
        return "Email categorization evaluation logic"


class EmailTemplateExtractionTask(BaseTask):
    """
    Task for extracting template information from subscription emails.
    """

    def get_prompt(self, samples: list[Sample]) -> str:
        """
        Generate prompt using the samples formatted as ID|Subject|Snippet for template extraction.

        Args:
            samples: List of samples to generate prompt for

        Returns:
            Generated prompt string
        """
        # Format samples as ID|Subject|Snippet (no dates for template extraction)
        formatted_emails = []
        for i, sample in enumerate(samples):
            # Format: ID|Subject|Snippet
            email_line = f"{i}|{sample.subject}|{sample.snippet}"
            formatted_emails.append(email_line)

        emails = "\n".join(formatted_emails)
        return prompt_loader.get_latest_prompt_by_task_id(self.task_id, {"emails": emails})

    def _matches(self, text: str, anchors: list[str]) -> bool:
        """
        Check if all anchor strings appear in order within the text.

        This is equivalent to the Kotlin matches method that verifies all anchors
        are found in the text in the correct order (case-insensitive).

        Args:
            text: The text to search within
            anchors: List of anchor strings that must appear in order

        Returns:
            True if all anchors are found in order, False otherwise
        """
        # Validate anchors must be trimmed
        if not all(anchor == anchor.strip() for anchor in anchors):
            raise ValueError("anchors contains untrimmed strings")

        # Normalize text by replacing multiple spaces with single space and trimming
        import re

        normalized_text = re.sub(r"\s+", " ", text).strip()

        current_pos = 0
        for anchor in anchors:
            # Case-insensitive search
            index = normalized_text.lower().find(anchor.lower(), current_pos)

            if index == -1:
                return False

            current_pos = index + len(anchor)

        return True

    def _match_message(
        self, subject: str, snippet: str, subject_anchors: list[str], snippet_anchors: list[str]
    ) -> bool:
        """
        Check if both subject and snippet match their respective anchor patterns.

        Args:
            subject: Email subject text
            snippet: Email snippet text
            subject_anchors: List of anchor strings for subject
            snippet_anchors: List of anchor strings for snippet

        Returns:
            True if both subject and snippet match their anchors
        """
        subject_match = self._matches(subject, subject_anchors)
        snippet_match = self._matches(snippet, snippet_anchors)
        return subject_match and snippet_match

    def _score_single_match(
        self, subject: str, snippet: str, subject_anchors: list[str], snippet_anchors: list[str]
    ) -> float:
        """
        Score a single template match result.

        Args:
            subject: Actual email subject
            snippet: Actual email snippet
            subject_anchors: Predicted subject anchors
            snippet_anchors: Predicted snippet anchors

        Returns:
            Score: 1.0 (both match), 0.5 (one matches), 0.0 (neither matches)
        """
        subject_match = self._matches(subject, subject_anchors) if subject_anchors else False
        snippet_match = self._matches(snippet, snippet_anchors) if snippet_anchors else False

        if subject_match and snippet_match:
            return 1.0  # Perfect match
        elif subject_match or snippet_match:
            return 0.5  # Partial match
        else:
            return 0.0  # No match

    def _get_score_accuracy(
        self, subject: str, snippet: str, subject_anchors: list[str], snippet_anchors: list[str]
    ) -> float:
        """
        Score a single template match result.

        Args:
            subject: Actual email subject
            snippet: Actual email snippet
            subject_anchors: Predicted subject anchors
            snippet_anchors: Predicted snippet anchors

        Returns:
            Score: 1.0 (both match), 0.5 (one matches), 0.0 (neither matches)
        """
        subject_match = self._matches(subject, subject_anchors) if subject_anchors else False
        snippet_match = self._matches(snippet, snippet_anchors) if snippet_anchors else False

        if subject_match and snippet_match:
            return 1.0  # Perfect match
        elif subject_match or snippet_match:
            return 0.5  # Partial match
        else:
            return 0.0  # No match

    def _get_score_specificity(
        self,
        subject_anchors: list[str],
        snippet_anchors: list[str],
        false_templates: list[EmailTemplate],
    ) -> float:

        fp = sum(
            1
            for template in false_templates
            if self._matches(template.subject, subject_anchors)
            and self._matches(template.snippet, snippet_anchors)
        )
        tn = len(false_templates) - fp

        specificity = tn / (tn + fp)

        return specificity

    def _get_scores(
        self, ai_output: str, samples: list[Sample], batch_size: int, **kwargs
    ) -> list[Score]:
        """
        Compute score for email template extraction task.

        Args:
            ai_output: The model's response in JSON format with extracted templates
            samples: List of samples with expected answers

        Returns:
            List of Score objects with accuracy score
        """
        import json

        from llm_benchmark.dataset.loader import dataset_loader

        try:
            all_templates: list[EmailTemplate] = kwargs.get("all_templates", [])

            if not all_templates:
                raise ValueError("all_templates is required for EmailTemplateExtractionTask")

            ai_result = json.loads(ai_output)

            if not ai_result.get("result"):
                return [Score(name="accuracy", value=0.0, comment="accuracy")]

            # Use 0-based indexing like EmailCategorizationTask
            index_to_sample = dict(enumerate(samples))

            templates = dataset_loader.load_latest_templates()
            template_id_to_template = {template.id: template for template in templates}

            total_accuracy_score = 0.0
            total_specificity_score = 0.0
            skip_specificity = False

            if len({t.subscription_event_type for t in all_templates}) == 1:
                total_specificity_score = -1.0
                skip_specificity = True

            results = ai_result["result"]

            for result_item in results:
                message_index = result_item.get("m")
                subject_anchors = result_item.get("j", [])
                snippet_anchors = result_item.get("p", [])

                # Ensure anchors are lists (in case they come as strings)
                if isinstance(subject_anchors, str):
                    subject_anchors = [subject_anchors]
                if isinstance(snippet_anchors, str):
                    snippet_anchors = [snippet_anchors]

                sample = index_to_sample.get(message_index)
                if not sample:
                    # If sample not found, skip this result
                    continue

                template = template_id_to_template.get(sample.template_id)
                if not template:
                    # If template not found, skip this result
                    continue

                accuracy_score = self._get_score_accuracy(
                    template.subject, template.snippet, subject_anchors, snippet_anchors
                )
                total_accuracy_score += accuracy_score

                if skip_specificity:
                    continue

                false_templates = [
                    t
                    for t in all_templates
                    if t.subscription_event_type != template.subscription_event_type
                ]

                if false_templates:
                    specificity_score = self._get_score_specificity(
                        subject_anchors, snippet_anchors, false_templates
                    )
                    total_specificity_score += specificity_score

            return [
                Score(
                    name="accuracy",
                    value=batch_size - (len(results) - total_accuracy_score),
                    comment="accuracy",
                ),
                Score(
                    name="specificity",
                    value=batch_size - (len(results) - total_specificity_score),
                    comment="specificity",
                ),
            ]

        except (json.JSONDecodeError, KeyError, TypeError):
            # If AI output is malformed, return 0
            return [Score(name="accuracy", value=0.0, comment="accuracy")]

    @property
    def evaluation_comment(self) -> str:
        """Get evaluation comment for Langfuse."""
        return "Email template extraction evaluation logic"


def get_task(task_name: str) -> BaseTask:
    """
    Get a task instance by name.

    Args:
        task_name: Name of the task to retrieve

    Returns:
        Task instance

    Raises:
        ValueError: If task name is not found
    """
    from llm_benchmark.benchmark.task.task_factory import task_factory

    try:
        return task_factory.get_task(task_name)
    except ValueError as e:
        available_tasks = ", ".join(task_factory.list_available_tasks())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available_tasks}") from e
