# Unit tests for benchmark/task

import json
from unittest.mock import Mock, patch

import pytest
from datasets_shared.schema import SubscriptionEventType
from llm_benchmark.benchmark.task.task import (
    EmailCategorizationTask,
    EmailTemplateExtractionTask,
    get_task,
)


class TestEmailCategorizationTask:
    """Test cases for EmailCategorizationTask."""

    @pytest.fixture
    def email_categorization_task(self):
        """Create EmailCategorizationTask instance for testing."""
        return EmailCategorizationTask(
            name="Email Categorization",
            task_id="email_categorization",
            description="Test categorization task",
            batch_size=10,
            response_model="email_categorization_v1",
        )

    def test_get_scores_perfect_match(self, email_categorization_task, mock_dataset):
        """Test _get_scores with perfect categorization."""
        test_samples = mock_dataset.samples[:4]

        subscription_start_or_payment_indices = []
        cancel_indices = []

        for i, sample in enumerate(test_samples):
            if (
                sample.subscription_event_type
                == SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT
            ):
                subscription_start_or_payment_indices.append(i)
            elif sample.subscription_event_type == SubscriptionEventType.SUBSCRIPTION_CANCEL:
                cancel_indices.append(i)

        ai_output = json.dumps(
            {
                "S": subscription_start_or_payment_indices,
                "C": cancel_indices,
            }
        )

        result = email_categorization_task._get_scores(ai_output, test_samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 4.0

    def test_get_scores_partial_match(self, email_categorization_task, mock_dataset):
        """Test _get_scores with partial categorization."""
        test_samples = mock_dataset.samples[:4]
        subscription_start_or_payment_indices = []
        cancel_indices = []

        for i, sample in enumerate(test_samples):
            if (
                sample.subscription_event_type
                == SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT
            ):
                subscription_start_or_payment_indices.append(i)
            elif sample.subscription_event_type == SubscriptionEventType.SUBSCRIPTION_CANCEL:
                cancel_indices.append(i)

        ai_output = json.dumps(
            {
                "S": subscription_start_or_payment_indices + cancel_indices[:1],
                "C": cancel_indices,
            }
        )

        result = email_categorization_task._get_scores(ai_output, test_samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 3.0

    def test_get_scores_no_match(self, email_categorization_task, mock_dataset):
        """Test _get_scores with no correct categorizations."""
        test_samples = mock_dataset.samples[:4]

        ai_output = json.dumps(
            {
                "M": [1],  # Wrong assignment - index 1
                "C": [0],  # Wrong assignment - index 0
            }
        )

        result = email_categorization_task._get_scores(ai_output, test_samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    def test_get_scores_invalid_json(self, email_categorization_task, mock_dataset):
        """Test _get_scores with malformed JSON output."""
        test_samples = mock_dataset.samples[:4]

        ai_output = "invalid json"

        result = email_categorization_task._get_scores(ai_output, test_samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    def test_get_scores_unknown_categories(self, email_categorization_task, mock_dataset):
        """Test _get_scores with unknown category keys."""
        test_samples = mock_dataset.samples[:4]

        ai_output = json.dumps({"X": [0], "Y": [1]})  # Unknown category keys

        result = email_categorization_task._get_scores(ai_output, test_samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    def test_get_scores_empty_samples(self, email_categorization_task, mock_dataset):
        """Test _get_scores with empty samples list."""
        samples = []
        ai_output = json.dumps({"M": [], "A": [], "S": [], "C": []})
        result = email_categorization_task._get_scores(ai_output, samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    def test_get_scores_duplicate_ids(self, email_categorization_task, mock_dataset):
        """Test _get_scores when AI output contains duplicate indices."""
        test_samples = mock_dataset.samples[:4]

        ai_output = json.dumps(
            {
                "M": [0],
                "A": [0],  # Duplicate index in different category
            }
        )

        result = email_categorization_task._get_scores(ai_output, test_samples)
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    def test_evaluation_comment(self, email_categorization_task):
        """Test evaluation comment property."""
        assert (
            email_categorization_task.evaluation_comment == "Email categorization evaluation logic"
        )


class TestEmailTemplateExtractionTask:
    """Test cases for EmailTemplateExtractionTask."""

    @pytest.fixture
    def email_template_extraction_task(self):
        """Create EmailTemplateExtractionTask instance for testing."""
        return EmailTemplateExtractionTask(
            name="Email Template Extraction",
            task_id="email_template_extraction",
            description="Test template extraction task",
            batch_size=10,
            response_model="email_template_extraction_v1",
        )

    @pytest.fixture
    def mock_template(self, mock_dataset):
        """Create mock template for testing."""
        t = Mock()
        t.id = mock_dataset.samples[0].template_id
        t.subject = "Payment Confirmation"
        t.snippet = "Your monthly payment has been processed"
        return t

    @pytest.fixture
    def setup_loader(self, mock_template):
        """Create setup function for mock loader."""

        def _setup(mock_loader):
            mock_loader.load_latest_templates.return_value = [mock_template]

        return _setup

    @pytest.fixture
    def basic_anchors(self):
        """Basic anchor list for matching tests."""
        return ["Hello", "World", "Test"]

    @pytest.fixture
    def payment_subject(self):
        """Payment subject for message matching tests."""
        return "Payment Confirmation"

    @pytest.fixture
    def payment_snippet(self):
        """Payment snippet for message matching tests."""
        return "Your monthly payment has been processed"

    @pytest.fixture
    def all_templates(self, mock_dataset):
        """Mock all templates for specificity calculation."""
        templates = []
        for sample in mock_dataset.samples:
            t = Mock()
            t.id = sample.template_id
            t.subject = sample.subject
            t.snippet = sample.snippet
            t.subscription_event_type = sample.subscription_event_type
            templates.append(t)
        return templates

    def test_matches_basic(self, email_template_extraction_task, basic_anchors):
        """Test _matches method with basic functionality."""
        text = "Hello World Test"

        result = email_template_extraction_task._matches(text, basic_anchors)
        assert result is True

    def test_matches_case_insensitive(self, email_template_extraction_task, basic_anchors):
        """Test _matches method is case insensitive."""
        text = "hello WORLD test"

        result = email_template_extraction_task._matches(text, basic_anchors)
        assert result is True

    def test_matches_wrong_order(self, email_template_extraction_task):
        """Test _matches method with wrong anchor order."""
        text = "Hello World Test"
        anchors = ["World", "Hello", "Test"]

        result = email_template_extraction_task._matches(text, anchors)
        assert result is False

    def test_matches_missing_anchor(self, email_template_extraction_task, basic_anchors):
        """Test _matches method with missing anchor."""
        text = "Hello World"

        result = email_template_extraction_task._matches(text, basic_anchors)
        assert result is False

    def test_matches_untrimmed_anchors(self, email_template_extraction_task):
        """Test _matches method raises error for untrimmed anchors."""
        text = "Hello World Test"
        anchors = ["Hello ", "World", "Test"]

        with pytest.raises(ValueError, match="anchors contains untrimmed strings"):
            email_template_extraction_task._matches(text, anchors)

    def test_matches_normalizes_spaces(self, email_template_extraction_task, basic_anchors):
        """Test _matches method normalizes multiple spaces."""
        text = "Hello    World     Test"

        result = email_template_extraction_task._matches(text, basic_anchors)
        assert result is True

    def test_match_message_both_match(
        self, email_template_extraction_task, payment_subject, payment_snippet
    ):
        """Test _match_message when both subject and snippet match."""
        subject_anchors = ["Payment", "Confirmation"]
        snippet_anchors = ["monthly", "payment"]

        result = email_template_extraction_task._match_message(
            payment_subject, payment_snippet, subject_anchors, snippet_anchors
        )
        assert result is True

    def test_match_message_subject_only(self, email_template_extraction_task, payment_subject):
        """Test _match_message when only subject matches."""
        snippet = "Your payment has been processed"
        subject_anchors = ["Payment", "Confirmation"]
        snippet_anchors = ["monthly", "payment"]  # "monthly" not in snippet

        result = email_template_extraction_task._match_message(
            payment_subject, snippet, subject_anchors, snippet_anchors
        )
        assert result is False  # _match_message requires BOTH to match

    def test_match_message_snippet_only(self, email_template_extraction_task):
        """Test _match_message when only snippet matches."""
        subject = "Payment Info"
        snippet = "Your monthly payment has been processed"
        subject_anchors = ["Payment", "Confirmation"]  # "Confirmation" not in subject
        snippet_anchors = ["monthly", "payment"]

        result = email_template_extraction_task._match_message(
            subject, snippet, subject_anchors, snippet_anchors
        )
        assert result is False  # _match_message requires BOTH to match

    @pytest.mark.parametrize(
        "subject_anchors,snippet_anchors,expected",
        [
            (["Payment", "Confirmation"], ["monthly", "payment"], 1.0),
            (["Payment", "Confirmation"], ["missing", "anchor"], 0.5),
            (["missing", "anchor"], ["missing", "anchor"], 0.0),
        ],
    )
    def test_get_score_accuracy(
        self,
        email_template_extraction_task,
        payment_subject,
        payment_snippet,
        subject_anchors,
        snippet_anchors,
        expected,
    ):
        """Test _get_score_accuracy with various match scenarios."""
        result = email_template_extraction_task._get_score_accuracy(
            payment_subject, payment_snippet, subject_anchors, snippet_anchors
        )
        assert result == expected

    def test_get_score_specificity(self, email_template_extraction_task):
        """Test _get_score_specificity with false templates."""
        false_templates = [
            Mock(subject="Cancellation Notice", snippet="Your subscription has been cancelled"),
            Mock(subject="Refund Processed", snippet="Your refund is complete"),
        ]

        subject_anchors = ["Payment", "Confirmation"]
        snippet_anchors = ["monthly", "payment"]

        result = email_template_extraction_task._get_score_specificity(
            subject_anchors, snippet_anchors, false_templates
        )
        assert result == 1.0  # No false templates match

    @pytest.mark.parametrize(
        "j,p,expected_accuracy",
        [
            (["Payment", "Confirmation"], ["monthly", "payment"], 1.0),
            (["Payment", "Confirmation"], ["missing", "anchor"], 0.5),
            (["missing", "anchor"], ["missing", "anchor"], 0.0),
        ],
    )
    @patch("llm_benchmark.dataset.loader.dataset_loader")
    def test_get_scores(
        self,
        mock_loader,
        setup_loader,
        email_template_extraction_task,
        mock_dataset,
        all_templates,
        j,
        p,
        expected_accuracy,
    ):
        """Test _get_scores with various match scenarios."""
        setup_loader(mock_loader)

        ai_output = json.dumps({"result": [{"m": 0, "j": j, "p": p}]})

        result = email_template_extraction_task._get_scores(
            ai_output, [mock_dataset.samples[0]], all_templates=all_templates
        )
        assert len(result) == 2
        assert result[0].name == "accuracy"
        assert result[0].value == expected_accuracy
        assert result[1].name == "specificity"

    @patch("llm_benchmark.dataset.loader.dataset_loader")
    def test_get_scores_multiple_results(
        self, mock_loader, email_template_extraction_task, mock_dataset, all_templates
    ):
        """Test _get_scores with multiple results."""
        mock_template1 = Mock()
        mock_template1.id = mock_dataset.samples[0].template_id
        mock_template1.subject = "Payment Confirmation"
        mock_template1.snippet = "Your monthly payment has been processed"

        mock_template2 = Mock()
        mock_template2.id = mock_dataset.samples[1].template_id
        mock_template2.subject = "Cancellation Notice"
        mock_template2.snippet = "Your subscription has been cancelled"

        mock_loader.load_latest_templates.return_value = [mock_template1, mock_template2]

        test_samples = mock_dataset.samples[:2]

        ai_output = json.dumps(
            {
                "result": [
                    {
                        "m": 0,
                        "j": ["Payment", "Confirmation"],
                        "p": ["monthly", "payment"],
                    },
                    {
                        "m": 1,
                        "j": ["Cancellation", "Notice"],
                        "p": ["subscription", "cancelled"],
                    },
                ]
            }
        )

        result = email_template_extraction_task._get_scores(
            ai_output, test_samples, all_templates=all_templates
        )
        assert len(result) == 2
        assert result[0].name == "accuracy"
        assert result[0].value == 2.0
        assert result[1].name == "specificity"

    def test_get_scores_invalid_json(
        self, email_template_extraction_task, mock_dataset, all_templates
    ):
        """Test _get_scores with malformed JSON."""
        test_samples = mock_dataset.samples[:1]
        ai_output = "invalid json"

        result = email_template_extraction_task._get_scores(
            ai_output, test_samples, all_templates=all_templates
        )
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    @patch("llm_benchmark.dataset.loader.dataset_loader")
    def test_get_scores_missing_result_key(
        self, mock_loader, email_template_extraction_task, mock_dataset, all_templates
    ):
        """Test _get_scores when result key is missing."""
        mock_loader.load_latest_templates.return_value = []

        test_samples = mock_dataset.samples[:1]
        ai_output = json.dumps({"no_result": []})

        result = email_template_extraction_task._get_scores(
            ai_output, test_samples, all_templates=all_templates
        )
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0

    @patch("llm_benchmark.dataset.loader.dataset_loader")
    def test_get_scores_string_anchors(
        self, mock_loader, email_template_extraction_task, mock_dataset, all_templates
    ):
        """Test _get_scores when anchors come as strings instead of lists."""
        mock_template = Mock()
        mock_template.id = mock_dataset.samples[0].template_id
        mock_template.subject = "Payment Confirmation"
        mock_template.snippet = "Your monthly payment has been processed"
        mock_loader.load_latest_templates.return_value = [mock_template]

        test_samples = mock_dataset.samples[:1]

        ai_output = json.dumps(
            {
                "result": [
                    {
                        "m": 0,
                        "j": "Payment",
                        "p": "monthly",
                    }
                ]
            }
        )

        result = email_template_extraction_task._get_scores(
            ai_output, test_samples, all_templates=all_templates
        )
        assert len(result) == 2
        assert result[0].name == "accuracy"
        assert result[0].value == 1.0
        assert result[1].name == "specificity"

    @patch("llm_benchmark.dataset.loader.dataset_loader")
    def test_get_scores_missing_sample(
        self, mock_loader, email_template_extraction_task, mock_dataset, all_templates
    ):
        """Test _get_scores when sample index is not found."""
        mock_loader.load_latest_templates.return_value = []

        test_samples = mock_dataset.samples[:1]

        ai_output = json.dumps(
            {
                "result": [
                    {
                        "m": 999,
                        "j": ["Payment"],
                        "p": ["monthly"],
                    }
                ]
            }
        )

        result = email_template_extraction_task._get_scores(
            ai_output, test_samples, all_templates=all_templates
        )
        assert len(result) == 2
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0
        assert result[1].name == "specificity"

    @patch("llm_benchmark.dataset.loader.dataset_loader")
    def test_get_scores_missing_template(
        self, mock_loader, email_template_extraction_task, mock_dataset, all_templates
    ):
        """Test _get_scores when template is not found."""
        mock_loader.load_latest_templates.return_value = []

        test_samples = mock_dataset.samples[:1]

        ai_output = json.dumps({"result": [{"m": 0, "j": ["Payment"], "p": ["monthly"]}]})

        result = email_template_extraction_task._get_scores(
            ai_output, test_samples, all_templates=all_templates
        )
        assert len(result) == 2
        assert result[0].name == "accuracy"
        assert result[0].value == 0.0
        assert result[1].name == "specificity"

    def test_evaluation_comment(self, email_template_extraction_task):
        """Test evaluation comment property."""
        assert (
            email_template_extraction_task.evaluation_comment
            == "Email template extraction evaluation logic"
        )


class TestGetTaskFunction:
    """Test cases for get_task function."""

    @patch("llm_benchmark.benchmark.task.task_factory.task_factory")
    def test_get_task_success(self, mock_factory):
        """Test get_task with valid task name."""
        mock_task = Mock()
        mock_factory.get_task.return_value = mock_task
        mock_factory.list_available_tasks.return_value = ["task1", "task2"]

        result = get_task("task1")

        assert result == mock_task
        mock_factory.get_task.assert_called_once_with("task1")

    @patch("llm_benchmark.benchmark.task.task_factory.task_factory")
    def test_get_task_not_found(self, mock_factory):
        """Test get_task with invalid task name."""
        mock_factory.get_task.side_effect = ValueError("Task not found")
        mock_factory.list_available_tasks.return_value = ["task1", "task2"]

        with pytest.raises(
            ValueError, match="Unknown task 'invalid_task'. Available tasks: task1, task2"
        ):
            get_task("invalid_task")

    @patch("llm_benchmark.benchmark.task.task_factory.task_factory")
    def test_get_task_no_available_tasks(self, mock_factory):
        """Test get_task when no tasks are available."""
        mock_factory.get_task.side_effect = ValueError("Task not found")
        mock_factory.list_available_tasks.return_value = []

        with pytest.raises(ValueError, match="Unknown task 'invalid_task'. Available tasks: "):
            get_task("invalid_task")
