import logging
from functools import lru_cache
from typing import Any

from langfuse.api import GetScoresResponseData, ObservationV2, TraceWithDetails
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from langfuse import Langfuse, LangfuseGeneration

logger = logging.getLogger(__name__)


class LangfuseConfig(BaseSettings):
    """Configuration for Langfuse integration using pydantic-settings."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LANGFUSE_", extra="ignore")
    secret_key: SecretStr | None = Field(default=None)
    public_key: str | None = Field(default=None)
    host: str | None = Field(default="https://cloud.langfuse.com")
    enabled: bool = Field(default=True)


@lru_cache
def get_langfuse_config():
    return LangfuseConfig()


langfuse_config = get_langfuse_config()


class LangfuseClient:
    """
    Langfuse client for logging benchmark results.

    This implements the "HF + Langfuse SDK" pattern:
    - HuggingFace: Source of truth for datasets
    - Langfuse: Result dashboard for traces and scores
    """

    def __init__(self, config: LangfuseConfig):
        self.config = config
        # Fail immediately if enabled but no keys
        if config.enabled and (not config.public_key or not config.secret_key):
            raise RuntimeError("Langfuse is enabled but keys are missing from environment.")

        self._client = (
            Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key.get_secret_value() if config.secret_key else None,
                base_url=config.host,
            )
            # if config.enabled
            # else None
        )
        self._enabled = config.enabled

    def is_enabled(self) -> bool:
        """Check if Langfuse integration is enabled and available."""
        return self._enabled

    @property
    def client(self):
        """
        Internal accessor that guarantees a non-None client.
        Raises an error if accessed while disabled.
        """
        # if not self._enabled or self._client is None:
        #     raise RuntimeError(
        #         "Attempted to access Langfuse client while it is disabled or uninitialized."
        #     )
        return self._client

    def create_generation_span(
        self,
        parent: Any,
        model_id: str,
        sample_ids: list[str],
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        score_name: str | None = None,
        score_value: float | None = None,
        score_comment: str | None = None,
    ) -> str | None:
        """
        Create a trace for a single benchmark execution.

        Args:
            model_id: Model being benchmarked
            task_id: Task being evaluated
            experiment_id: Unique experiment identifier (e.g., HF commit hash)
            sample_ids: Sample identifier list from dataset
            prompt: Input prompt sent to model
            completion: Model's response
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency_ms: Response latency in milliseconds
            score_name: Optional score name (e.g., "accuracy")
            score_value: Optional score value (0.0 to 1.0 for normalized scores)
            score_comment: Optional comment describing the evaluation logic
            parent_trace_id: Optional parent trace ID to create this observation as a child span

        Returns:
            Trace ID if successful, None otherwise
        """

        try:
            observation_name = "prompt_call"
            with parent.start_as_current_observation(
                as_type="generation",
                name=observation_name,
                model=model_id,
                latency_ms=latency_ms,
            ) as observation:
                self._update_observation(
                    observation,
                    sample_ids,
                    prompt,
                    completion,
                    prompt_tokens,
                    completion_tokens,
                    score_name,
                    score_value,
                    score_comment,
                )

            logger.debug(f"Created generation span: {observation.id}")
            return observation.id
        except Exception as e:
            logger.error(f"Failed to create generation span: {e}")
            return None

    def _update_observation(
        self,
        observation: LangfuseGeneration,
        sample_ids: list[str],
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        score_name: str | None,
        score_value: float | None,
        score_comment: str | None,
    ) -> None:
        """Helper method to update observation with data and optional score."""
        observation.update(
            input=prompt,
            output=completion,
            metadata={
                "sample_ids": sample_ids,
            },
            usage_details={
                "input": prompt_tokens,
                "output": completion_tokens,
            },
        )

        if score_name is not None and score_value is not None:
            observation.score(
                name=score_name,
                value=score_value,
                comment=score_comment or f"Custom Python evaluation for {score_name}",
            )

    def flush(self) -> bool:
        """Flush any pending Langfuse operations."""
        if not self.is_enabled():
            return False

        try:
            self.client.flush()
            logger.debug("Langfuse client flushed")
            return True
        except Exception as e:
            logger.error(f"Failed to flush Langfuse client: {e}")
            return False

    def get_traces(self, tags: list[str]) -> list[TraceWithDetails]:
        limit: int = 100
        traces = self.client.api.trace.list(limit=limit, tags=tags)
        return traces.data

    def get_observations(
        self, trace_id: str, type="GENERATION", fields=None
    ) -> list[ObservationV2]:
        """Fetch generations using cursor-based pagination with retry logic.

        Args:
            trace_id: The trace ID to fetch observations for
            type: The type of observation to fetch (default: "GENERATION")

        Returns:
            List of all generations across all pages
        """
        page_size = 100
        cursor = None
        all_items = []

        while True:
            if cursor:
                logger.info(f"🔄 Fetching next page with cursor: {cursor}")
            else:
                logger.info(f"🔄 Fetching first page with limit: {page_size}")

            response = self.client.api.observations.get_many(
                limit=page_size,
                cursor=cursor,
                trace_id=trace_id,
                type=type,
                fields=fields,
            )

            page_items = response.data
            all_items.extend(page_items)
            logger.info(
                f"📊 Retrieved {len(page_items)} generations, total so far: {len(all_items)}"
            )

            # Check if there are more pages using cursor-based pagination
            if hasattr(response, "meta") and response.meta:
                # Check for cursor in meta (v2 API format)
                if hasattr(response.meta, "cursor") and response.meta.cursor:
                    cursor = response.meta.cursor
                    logger.info("📄 Found cursor, fetching next generations page...")
                else:
                    # No cursor available, assume no more pages
                    logger.info("🏁 No cursor found in generations meta, pagination complete")
                    break
            else:
                # Fallback: if no meta info, assume this is the last page
                logger.info("⚠️  No meta object in generations response, assuming single page")
                break

        return all_items

    def get_scores(self, trace_id: str, type="GENERATION") -> list[GetScoresResponseData]:
        """Fetch scores using cursor-based pagination with retry logic.

        Args:
            trace_id: The trace ID to fetch observations for
            type: The type of observation to fetch (default: "GENERATION")

        Returns:
            List of all scores across all pages
        """
        page_size = 100
        page = 1
        all_items = []

        while True:
            if page > 1:
                logger.info(f"🔄 Fetching next page with page: {page}")
            else:
                logger.info(f"🔄 Fetching first page with limit: {page_size}")

            response = self.client.api.scores.get_many(
                page=page,
                limit=page_size,
                trace_id=trace_id,
            )

            page_items = response.data
            all_items.extend(page_items)
            logger.info(f"📊 Retrieved {len(page_items)} scores, total so far: {len(all_items)}")

            # Check if there are more pages using cursor-based pagination
            if hasattr(response, "meta") and response.meta:
                # Check for cursor in meta (v2 API format)
                if hasattr(response.meta, "totalPages") and response.meta.total_pages > page:
                    page += 1
                    logger.info("📄 Found cursor, fetching next generations page...")
                else:
                    # No cursor available, assume no more pages
                    logger.info("🏁 No cursor found in generations meta, pagination complete")
                    break
            else:
                # Fallback: if no meta info, assume this is the last page
                logger.info("⚠️  No meta object in generations response, assuming single page")
                break

        return all_items


@lru_cache
def get_langfuse_client() -> LangfuseClient:
    """Standard way to access the singleton without import-time side effects."""
    return LangfuseClient(langfuse_config)
