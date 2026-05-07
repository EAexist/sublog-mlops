"""
Tests for dataset generator modules.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets_shared.schema import (
    Dataset,
    EmailTemplate,
    EmailTextParameterSet,
    Sample,
    SubscriptionEventType,
)
from llm_benchmark.dataset.generator.generator import (
    _call_oracle,
    _validate_sample,
    assemble_dataset,
)
from llm_benchmark.dataset.generator.parameter_generator import generate_parameters
from llm_benchmark.dataset.generator.template_generator import (
    EmailTemplateListPayload,
    EmailTemplatePayload,
    generate_templates,
)
from pydantic import BaseModel


class TestResponseModel(BaseModel):
    __test__ = False
    content: str


class TestDatasetGenerator:
    """Test cases for dataset generator modules."""

    @pytest.mark.asyncio
    @patch("llm_benchmark.models.litellm_factory.LLMClientFactory")
    async def test_call_oracle(self, mock_factory):
        """Test the _call_oracle wrapper ensures client.complete is called correctly."""
        # Setup mock client and response
        mock_client = AsyncMock()
        mock_factory.get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.parsed_data = {"content": "Generated Content"}
        mock_response.content = "Generated Content"
        mock_client.complete.return_value = mock_response

        # Execute
        result = await _call_oracle(
            "test prompt", "model-id", TestResponseModel, {"temperature": 0.5}
        )

        # Assert
        assert isinstance(result, TestResponseModel)
        assert result.content == "Generated Content"
        mock_factory.get_client.assert_called_once_with("model-id")
        mock_client.complete.assert_called_once_with(
            "test prompt", TestResponseModel, {"temperature": 0.5}
        )

    @pytest.mark.asyncio
    async def test_generate_templates(self):
        """Test generate_templates function with mocked oracle."""
        import llm_benchmark.dataset.generator.template_generator as tg
        from llm_benchmark.dataset.constants import CompanyInfo

        # Setup mock companies
        original_companies = tg.COMPANIES
        mock_company = CompanyInfo(
            id="netflix",
            name="Netflix",
            email="support@netflix.com",
            industry="Streaming",
            require_shared_email_discriminator=False,
        )
        tg.COMPANIES = [mock_company]

        # Mock oracle function
        mock_oracle = AsyncMock()
        mock_template_payload = EmailTemplatePayload(
            subject="Test Subject",
            snippet="Hello {{client_name}}, your subscription has started at {{date}} (Billing: {{payment_amount}}).",
        )
        mock_response = EmailTemplateListPayload(templates=[mock_template_payload])
        mock_oracle.return_value = mock_response

        try:
            # Execute
            templates = await generate_templates(n_templates=1, oracle_fn=mock_oracle)

            # Assert
            assert len(templates) > 0
            assert all(isinstance(t, EmailTemplate) for t in templates)

            # Check template structure
            template = templates[0]
            assert template.subject == "Test Subject"
            assert "{{client_name}}" in template.snippet
            assert template.company_id == "netflix"  # from mock companies
            # The event type will be one of the SubscriptionEventType values
            assert isinstance(template.subscription_event_type, SubscriptionEventType)

            # Verify oracle was called correctly
            mock_oracle.assert_called()

        finally:
            # Restore original companies
            tg.COMPANIES = original_companies

    @pytest.mark.asyncio
    async def test_generate_parameters(self):
        """Test generate_parameters function using Faker."""
        # Mock oracle function (should not be used in current implementation)
        mock_oracle = AsyncMock()

        # Execute
        parameters = generate_parameters(count=3, locales=["en_US"])

        # Assert
        assert len(parameters) == 3
        assert all(isinstance(p, EmailTextParameterSet) for p in parameters)

        # Check uniqueness
        client_names = [p.client_name for p in parameters]
        dates = [p.date for p in parameters]
        amounts = [p.payment_amount for p in parameters]

        assert len(set(client_names)) == 3  # All names unique
        assert len(set(dates)) == 3  # All dates unique
        assert len(set(amounts)) == 3  # All amounts unique

        # Check parameter structure
        param = parameters[0]
        assert isinstance(param.client_name, str)
        assert isinstance(param.date, str)
        assert isinstance(param.payment_amount, str)
        # Check for any supported currency symbol
        assert any(currency in param.payment_amount for currency in ["$", "€", "£", "¥", "₩"])

        # Verify oracle was NOT called (Faker-based implementation)
        mock_oracle.assert_not_called()

        # Note: publisher.publish is no longer called in generate_parameters
        # The function now just returns the parameters without publishing

    def test_assemble_dataset(self):
        """Test assemble_dataset function with mock data."""
        # Create mock templates
        templates = [
            EmailTemplate(
                id="test-id-1",
                subject="Subscription Started",
                snippet="Hello {{client_name}}, your subscription started on {{date}}.",
                company_id="netflix",
                subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            ),
            EmailTemplate(
                id="test-id-2",
                subject="Payment Received",
                snippet="Payment of {{payment_amount}} received from {{client_name}} on {{date}}.",
                company_id="netflix",
                subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            ),
        ]

        # Create mock parameters
        parameters = [
            EmailTextParameterSet(
                client_name="John Doe", date="2025-01-15", payment_amount="$15.99"
            ),
            EmailTextParameterSet(
                client_name="Jane Smith", date="2025-01-16", payment_amount="$25.99"
            ),
        ]

        # Execute
        dataset = assemble_dataset(templates, parameters, n_samples_per_template=1)

        # Assert
        assert isinstance(dataset, Dataset)
        assert len(dataset.samples) == 2  # Limited by min(templates, parameters)
        assert dataset.content_hash is not None
        assert len(dataset.content_hash) == 32  # MD5 hash length

        # Check first sample
        sample = dataset.samples[0]
        assert sample.subject == "Subscription Started"
        assert "John Doe" in sample.snippet
        assert "2025-01-15" in sample.snippet
        assert sample.subscription_event_type == SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT
        assert sample.company_id == "netflix"
        assert sample.template_id == "test-id-1"

        # Check second sample
        sample2 = dataset.samples[1]
        assert sample2.subject == "Payment Received"
        assert "Jane Smith" in sample2.snippet
        assert "$25.99" in sample2.snippet
        assert (
            sample2.subscription_event_type == SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT
        )

    def test_assemble_dataset_unequal_lengths(self):
        """Test assemble_dataset when templates and parameters have different lengths."""
        templates = [
            EmailTemplate(
                id="test-id-1",
                subject="Test 1",
                snippet="Hello {{client_name}}",
                company_id="netflix",
                subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            ),
            EmailTemplate(
                id="test-id-2",
                subject="Test 2",
                snippet="Hi {{client_name}}",
                company_id="netflix",
                subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            ),
            EmailTemplate(
                id="test-id-3",
                subject="Test 3",
                snippet="Hey {{client_name}}",
                company_id="netflix",
                subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            ),
        ]

        parameters = [
            EmailTextParameterSet(client_name="John", date="2025-01-15", payment_amount="$15.99")
        ]

        dataset = assemble_dataset(templates, parameters, n_samples_per_template=1)

        # Should only create 1 sample (limited by parameters)
        assert len(dataset.samples) == 1
        assert dataset.samples[0].subject == "Test 1"

    def test_validate_sample(self):
        """Test _validate_sample function."""
        # Valid sample
        valid_sample = Sample(
            id="test-1",
            subject="Test Subject",
            snippet="Test snippet with content",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            company_id="netflix",
            template_id="template-1",
        )
        assert _validate_sample(valid_sample) is True

        # Invalid sample - empty subject
        invalid_subject = Sample(
            id="test-2",
            subject="",
            snippet="Test snippet",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            company_id="netflix",
            template_id="template-2",
        )
        assert _validate_sample(invalid_subject) is False

        # Invalid sample - empty snippet
        invalid_snippet = Sample(
            id="test-3",
            subject="Test Subject",
            snippet="",
            subscription_event_type=SubscriptionEventType.SUBSCRIPTION_START_OR_PAYMENT,
            company_id="netflix",
            template_id="template-3",
        )
        assert _validate_sample(invalid_snippet) is False

    @pytest.mark.asyncio
    async def test_complete_generation_workflow(self):
        """Test complete workflow from templates to dataset assembly."""
        # Mock template generation
        mock_oracle = AsyncMock()
        mock_template_payload = EmailTemplatePayload(
            subject="Test Subject",
            snippet="Hello {{client_name}}, test snippet with {{date}} and {{payment_amount}}.",
        )
        mock_template_response = EmailTemplateListPayload(templates=[mock_template_payload])
        mock_oracle.return_value = mock_template_response

        # Generate templates
        templates = await generate_templates(n_templates=1, oracle_fn=mock_oracle)

        # Generate parameters (Faker-based, no oracle calls)
        parameters = generate_parameters(count=1, locales=["en_US"])

        # Assemble dataset
        dataset = assemble_dataset(templates, parameters, n_samples_per_template=1)

        # Verify complete workflow
        assert len(templates) > 0
        assert len(parameters) > 0
        assert len(dataset.samples) > 0
        assert dataset.content_hash is not None

        # Verify sample content is properly substituted
        sample = dataset.samples[0]
        assert "{{client_name}}" not in sample.snippet
        assert "{{date}}" not in sample.snippet
        assert "{{payment_amount}}" not in sample.snippet
