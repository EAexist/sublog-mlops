import logging
import time
from pathlib import Path

import pandas as pd
from langfuse.api import GetScoresResponseData_Numeric

from llm_benchmark.langfuse.langfuse_client import LangfuseClient, get_langfuse_client
from llm_benchmark.reporting.config import CSV_NAMING_PATTERN

logger = logging.getLogger(__name__)


def wait_langfuse_sync(
    experiment_id: str,
    expected_n_traces: int,
    expected_n_generations: int,
    langfuse_client: LangfuseClient | None = None,
    max_retries: int = 10,
    initial_wait_time: float = 10.0,
    backoff_factor: float = 2.0,
) -> bool:
    """
    Wait for Langfuse data synchronization with exponential backoff retry strategy.

    This function iteratively checks if the expected number of traces and generations
    are available in Langfuse, using exponential backoff between retries.

    Args:
        experiment_id: Experiment ID to filter traces
        expected_n_traces: Expected number of traces to verify
        expected_n_generations: Expected number of generations to verify
        langfuse_client: Langfuse client instance (optional, will use singleton if None)
        max_retries: Maximum number of retry attempts (default: 5)
        initial_wait_time: Initial wait time in seconds (default: 1.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)

    Returns:
        True if synchronization successful (both trace and generation counts match), False otherwise
    """
    langfuse_client = langfuse_client if langfuse_client is not None else get_langfuse_client()

    if not langfuse_client.is_enabled():
        logger.warning("Langfuse client is not enabled. Cannot sync data.")
        raise RuntimeError("Langfuse client is not enabled")

    current_wait_time = initial_wait_time

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Sync attempt {attempt + 1}/{max_retries} for experiment '{experiment_id}'"
            )

            # Step a: Check if get_traces() result length is expected_n_traces
            traces = langfuse_client.get_traces(tags=[experiment_id])
            actual_n_traces = len(traces)

            logger.debug(f"Found {actual_n_traces} traces, expected {expected_n_traces}")

            if actual_n_traces < expected_n_traces:
                logger.warning(
                    f"Trace count mismatch: got {actual_n_traces}, expected {expected_n_traces}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {current_wait_time:.1f}s before retry...")
                    time.sleep(current_wait_time)
                    current_wait_time *= backoff_factor
                    continue
                else:
                    logger.error(
                        f"Max retries reached. Final trace count: {actual_n_traces}, expected: {expected_n_traces}"
                    )
                    return False

            total_generations = 0
            for trace in traces:
                trace_id = trace.id
                generations = langfuse_client.get_observations(trace_id)
                total_generations += len(generations)

            logger.debug(
                f"Found {total_generations} generations, expected {expected_n_generations}"
            )

            if total_generations < expected_n_generations:
                logger.warning(
                    f"Generation count mismatch: got {total_generations}, expected {expected_n_generations}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {current_wait_time:.1f}s before retry...")
                    time.sleep(current_wait_time)
                    current_wait_time *= backoff_factor
                    continue
                else:
                    logger.error(
                        f"Max retries reached. Final generation count: {total_generations}, expected: {expected_n_generations}"
                    )
                    return False

            logger.info(
                f"Sync successful: {actual_n_traces} traces and {total_generations} generations verified"
            )
            return True

        except Exception as e:
            logger.error(f"Error during sync attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {current_wait_time:.1f}s before retry due to error...")
                time.sleep(current_wait_time)
                current_wait_time *= backoff_factor
            else:
                logger.error(f"Max retries reached due to errors. Last error: {e}")
                return False

    logger.error(f"Sync failed after {max_retries} attempts")
    return False


def download_langfuse_data_to_csv(
    experiment_id: str,
    langfuse_client: LangfuseClient | None = None,
    output_dir: Path | str = "data/benchmark",
) -> Path:
    """
    Download Langfuse traces and generations data, convert to DataFrame, and save as CSV.

    Args:
        output_dir: Directory to save the CSV file (default: "data/registry")
        limit: Maximum number of traces to fetch (default: 1000)
        experiment_id: Filter by experiment ID (optional)
        task_id: Filter by task ID (optional)
        model_id: Filter by model ID (optional)

    Returns:
        Path to the saved CSV file
    """
    langfuse_client = langfuse_client if langfuse_client is not None else get_langfuse_client()

    if not langfuse_client.is_enabled():
        logger.warning("Langfuse client is not enabled. Cannot download data.")
        raise RuntimeError("Langfuse client is not enabled")

    output_dir = Path(f"{output_dir}/{experiment_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = []
        traces = langfuse_client.get_traces(tags=[experiment_id])

        for trace in traces:
            trace_id = trace.id
            generations = langfuse_client.get_observations(
                trace_id, type="GENERATION", fields="core,model,usage,metadata"
            )
            scores = langfuse_client.get_scores(trace_id)

            logger.debug(f"Trace: {trace}")
            logger.debug(f"Generation: {generations[0]}")
            logger.debug(f"Score: {scores}")

            for gen in generations:
                id = gen.id
                gen_scores = [s for s in scores if s.observation_id == id]
                accuracy = next((s for s in gen_scores if s.name == "accuracy"), None)
                specificity = next((s for s in gen_scores if s.name == "specificity"), None)

                # Usage details
                # usage_details = gen.usage_details
                # input_tokens = usage_details.get("input") if usage_details else None
                # output = usage_details.get("output") if usage_details else None
                # total = gen.total_cost

                # Cost details
                cost_details = gen.cost_details

                data.append(
                    {
                        "id": id,
                        "task_name": gen.metadata.get("task_id", "") if gen.metadata else "",
                        "n_data": gen.metadata.get("n_data", 0) if gen.metadata else 0,
                        "model": gen.model,
                        "latency": gen.latency,
                        # Socres
                        "score_id_accuracy": accuracy.id if accuracy else None,
                        "score_accuracy": accuracy.value
                        if accuracy and isinstance(accuracy, GetScoresResponseData_Numeric)
                        else None,
                        "score_id_specificity": specificity.id if specificity else None,
                        "score_specificity": specificity.value
                        if specificity and isinstance(specificity, GetScoresResponseData_Numeric)
                        else None,
                        # Cost details
                        "cost_input": cost_details.get("input") if cost_details else None,
                        "cost_output": cost_details.get("output") if cost_details else None,
                        "cost_total": gen.total_cost,
                    }
                )

        df = pd.DataFrame(data)

        csv_path = output_dir / CSV_NAMING_PATTERN
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(f"Successfully downloaded {len(df)} records to {csv_path}")
        logger.info(f"Columns: {list(df.columns)}")

        return csv_path

    except Exception as e:
        logger.error(f"Failed to download Langfuse data: {e}")
        raise
