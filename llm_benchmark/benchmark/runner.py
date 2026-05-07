# Async fan-out: per-task benchmarks (all samples × all models per task), then aggregate
# Uses HuggingFace as source of truth, Langfuse as result dashboard

import asyncio
import logging

from datasets_shared.schema import EmailTemplate, Sample

from langfuse import propagate_attributes
from llm_benchmark.benchmark.task.task import BaseTask
from llm_benchmark.benchmark.task.task_factory import task_factory
from llm_benchmark.config_loader import ModelEntry, load_models_config
from llm_benchmark.langfuse.langfuse_client import LangfuseClient, get_langfuse_client

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self, langfuse: LangfuseClient | None = None):
        """
        Initialize the BenchmarkRunner with optional LangfuseClient.

        Args:
            langfuse: LangfuseClient instance for result logging
        """
        # Store the client directly as an instance attribute
        self._langfuse = langfuse if isinstance(langfuse, LangfuseClient) else get_langfuse_client()

    async def run_benchmarks(
        self,
        samples: list[Sample],
        templates: list[EmailTemplate],
        experiment_id: str,
        model_ids: list[str] | None = None,
        tasks: list[BaseTask] | None = None,
    ) -> list[str]:
        """
        Run one dataset against all models (or given model_ids).
        Uses HuggingFace as source of truth, Langfuse as result dashboard.

        Args:
            dataset: Dataset to benchmark (pulled from HuggingFace)
            model_ids: List of model IDs to test, None for all configured models
            task_type: Task type for prompt generation ("task_a" or "task_b")
            experiment_id: Unique experiment ID (e.g., HF commit hash) for Langfuse tracking
            tasks: List of tasks to run, defaults to both categorization and template extraction

        Returns:
            dict[model_id, list of {content, prompt_tokens, completion_tokens, latency_ms}]
        """

        if tasks is None:
            tasks = [
                # task_factory.get_task("email_categorization"),
                task_factory.get_task("email_template_extraction"),
            ]
        logger.info(f"Configured {len(tasks)} tasks: {[task.task_id for task in tasks]}")

        config = load_models_config()
        model_ids = model_ids or [m.id for m in config.models]
        logger.info(
            f"Running benchmark with experiment_id: {experiment_id} for {len(model_ids)} models: {model_ids}"
        )
        observation_ids = []

        # Group models by provider
        models_by_provider = {}
        for model in config.models:
            if model.provider not in models_by_provider:
                models_by_provider[model.provider] = []
            models_by_provider[model.provider].append(model)

        async def process_provider(provider, provider_models):
            logger.info(f"Processing provider: {provider}")
            local_obs_ids = []

            for model in sorted(provider_models, key=lambda m: m.id):
                if model.id not in model_ids:
                    continue

                # This will run sequentially WITHIN a provider,
                # but providers will run in parallel with each other.
                obs_ids = await self._run_model_benchmark(
                    model, samples, tasks, experiment_id, templates
                )
                local_obs_ids.extend(obs_ids)

            logger.info(f"Completed all tasks for provider '{provider}'")
            return local_obs_ids

        provider_coros = [
            process_provider(provider, models) for provider, models in models_by_provider.items()
        ]

        try:
            results = await asyncio.gather(*provider_coros, return_exceptions=True)

            # Check for exceptions in results
            exceptions_occurred = False
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    provider_name = list(models_by_provider.keys())[i]
                    logger.error(f"Provider '{provider_name}' failed with exception: {result}")
                    exceptions_occurred = True
                else:
                    observation_ids.extend(result)

            # Only flush if no exceptions occurred
            if not exceptions_occurred and self._langfuse:
                logger.info("Flushing Langfuse operations")
                self._langfuse.flush()
            elif exceptions_occurred:
                logger.error("Benchmark completed with failures - Langfuse flush skipped")

        except Exception as e:
            logger.error(f"Unexpected error during benchmark execution: {e}")
            logger.error("Benchmark failed - Langfuse flush skipped")

        logger.info(f"Benchmark run completed. Created {len(observation_ids)} observations")
        return observation_ids

    async def _run_model_benchmark(
        self,
        model: ModelEntry,
        samples: list[Sample],
        tasks: list[BaseTask],
        experiment_id: str,
        templates: list[EmailTemplate],
    ) -> list[str]:
        """Run all tasks for a single model.

        Args:
            model: Model configuration entry
            samples: List of samples to benchmark
            tasks: List of tasks to run
            experiment_id: Experiment ID for tracking
        """
        logger.info(f"Processing model: {model.id} ({model.full_id})")

        observation_ids: list[str] = []

        # Run all specified tasks
        for task in tasks:
            logger.info(
                f"Running task '{task.task_id}' for model '{model.id}' on {len(samples)} samples"
            )
            obs_ids = await self._run_task_benchmark(model, samples, task, experiment_id, templates)
            observation_ids.extend(obs_ids)
            logger.info(f"Completed task '{task.task_id}' for model '{model.id}'")

        logger.info(f"Completed all tasks for model '{model.id}'")
        return observation_ids

    async def _run_task_benchmark(
        self,
        model: ModelEntry,
        samples: list[Sample],
        task: BaseTask,
        experiment_id: str,
        templates: list[EmailTemplate],
    ) -> list[str]:
        """
        Run a specific task benchmark for a model.

        Args:
            model: Model configuration entry
            samples: List of samples to benchmark
            task: Task instance to run
            experiment_id: Unique experiment ID for Langfuse tracking

        Returns:
            List of result dictionaries
        """
        from llm_benchmark.models.litellm_factory import LLMClientFactory

        llm_client = LLMClientFactory.get_client(model.full_id)

        template_ids = {s.template_id for s in samples}
        all_templates = [t for t in templates if t.id in template_ids]

        logger.info(f"Template IDs: {template_ids}")
        logger.info(f"Templates Size: {len(templates)}")
        logger.info(f"All templates Size: {len(all_templates)}")

        batch_size = task.batch_size
        total_batches = (len(samples) + batch_size - 1) // batch_size

        trace_name = "ai-task-execution"
        generation_name = "prompt_call"

        logger.info(f"Running task: {task.task_id}, model: {model.id}")

        observation_ids = []

        root_span = self._langfuse.client.start_observation(
            as_type="trace",
            name=trace_name,
        )

        # with root_span:
        try:
            with propagate_attributes(tags=[task.task_id, experiment_id]):
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(samples))
                    batch_samples = samples[start_idx:end_idx]
                    # batch_samples = batch_samples[0:1]

                    logger.debug(
                        f"Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx + 1}-{end_idx}) for task '{task.task_id}'"
                    )

                    prompt = task.get_prompt(batch_samples)

                    # Enforce rate limits before making API call
                    await llm_client.enforce_rate_limit()

                    with root_span.start_as_current_observation(
                        as_type="generation",
                        name=generation_name,
                        metadata={"task_id": task.task_id, "n_data": len(batch_samples)},
                        model=model.id,
                    ) as observation:
                        response = await llm_client.complete(
                            prompt, task.response_model, config={"max_tokens": 4096}
                        )
                        observation.update(
                            input=prompt,
                            output=response.content,
                            usage_details={
                                "input": response.prompt_tokens,
                                "output": response.completion_tokens,
                            },
                        )
                        observation_ids.append(observation.id)

                    logger.debug(
                        f"Batch {batch_idx + 1} completed - tokens: {response.prompt_tokens}+{response.completion_tokens}, latency: {response.latency_ms}ms"
                    )
                    scores = task._get_scores(
                        response.content,
                        batch_samples,
                        batch_size=batch_size,
                        all_templates=all_templates,
                    )
                    for score in scores:
                        observation.score(
                            name=score.name,
                            value=score.value,
                            comment=score.comment,
                        )

                    logger.debug(
                        f"Created Langfuse generation span {observation.id} with scores {[f'{s.name}: {s.value}' for s in scores]} for batch {batch_idx + 1}"
                    )
        finally:
            # You MUST call end() manually since 'with' is not supported
            root_span.end()

        logger.info(f"Task '{task.task_id}' completed for model '{model.id}'")
        return observation_ids
