# TODO: Latency metrics — not used in the main pipeline yet.
# To enable latency: import this module and call compute_latency_ms / add_latency_to_metrics
# from llm_benchmark.benchmark.latency import compute_latency_ms, add_latency_to_metrics
# then merge latency fields into per_task_metrics in metrics.compute_task_metrics.

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def compute_latency_ms(latencies_ms: list[float]) -> dict[str, float]:
    """
    Return p50, p95, mean (ms).
    TODO: Called only when latency is enabled; not used in main flow yet.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "p50": statistics.median(s),
        "p95": s[int(0.95 * n)] if n else 0.0,
        "mean": sum(s) / n,
    }


def add_latency_to_metrics(
    raw_list: list[dict[str, Any]],
    metrics_dict: dict[str, float],
) -> None:
    """
    Mutate metrics_dict to add latency_p50, latency_p95, mean_latency_ms.
    raw_list: list of {latency_ms: float} from runner output.
    TODO: Call from metrics.compute_task_metrics when latency is enabled.
    """
    latencies = [r.get("latency_ms", 0.0) for r in raw_list]
    lat = compute_latency_ms(latencies)
    metrics_dict["latency_p50"] = lat["p50"]
    metrics_dict["latency_p95"] = lat["p95"]
    metrics_dict["mean_latency_ms"] = lat["mean"]
