"""
Utilities for testing Langfuse integration.
"""

from typing import Any

from llm_benchmark.langfuse.langfuse_client import LangfuseClient, LangfuseConfig


def create_test_langfuse_config() -> LangfuseConfig:
    """Create a test configuration for Langfuse client."""
    return LangfuseConfig(
        secret_key="test-secret-key",
        public_key="test-public-key",
        host="http://localhost:3000",  # Local Langfuse instance for testing
        enabled=True,
    )


def create_mock_langfuse_client() -> LangfuseClient:
    """Create a mock Langfuse client for testing."""
    config = create_test_langfuse_config()
    return LangfuseClient(config)


class LangfuseTestHelper:
    """Helper class for testing Langfuse integration."""

    def __init__(self, client: LangfuseClient | None = None):
        self.client = client or create_mock_langfuse_client()
        self.created_traces: dict[str, dict[str, Any]] = {}
        self.created_scores: dict[str, dict[str, Any]] = {}

    def get_trace_count(self) -> int:
        """Get the number of traces created during testing."""
        return len(self.created_traces)

    def get_score_count(self) -> int:
        """Get the number of scores created during testing."""
        return len(self.created_scores)

    def verify_trace_exists(self, trace_id: str) -> bool:
        """Verify a trace was created with the given ID."""
        return trace_id in self.created_traces

    def verify_score_exists(self, trace_id: str, score_name: str) -> bool:
        """Verify a score was created for the given trace and score name."""
        return trace_id in self.created_scores and score_name in self.created_scores[trace_id]

    def get_trace_metadata(self, trace_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific trace."""
        return self.created_traces.get(trace_id)

    def get_score_value(self, trace_id: str, score_name: str) -> float | None:
        """Get the value of a specific score."""
        if trace_id in self.created_scores and score_name in self.created_scores[trace_id]:
            return self.created_scores[trace_id][score_name].get("value")
        return None

    def reset(self) -> None:
        """Reset all tracked traces and scores."""
        self.created_traces.clear()
        self.created_scores.clear()
