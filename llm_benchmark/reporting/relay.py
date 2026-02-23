# Stub: push_to_client_repo(result) → NotImplementedError

import logging
from llm_benchmark.benchmark.scorer import BenchmarkResult

logger = logging.getLogger(__name__)


def push_to_client_repo(result: BenchmarkResult) -> None:
    """Push best_model_id to client repo (e.g. JSON + PR). Stub — do not refactor away."""
    raise NotImplementedError("push_to_client_repo is not implemented")
