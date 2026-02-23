# Unit tests for benchmark/latency (TODO module; not used in main pipeline)

import pytest
from llm_benchmark.benchmark.latency import compute_latency_ms, add_latency_to_metrics


def test_compute_latency_ms_empty() -> None:
    out = compute_latency_ms([])
    assert out["p50"] == 0.0 and out["p95"] == 0.0 and out["mean"] == 0.0


def test_compute_latency_ms_values() -> None:
    out = compute_latency_ms([100.0, 200.0, 300.0])
    assert out["mean"] == 200.0
    assert out["p50"] == 200.0


def test_add_latency_to_metrics() -> None:
    raw_list = [{"latency_ms": 100.0}, {"latency_ms": 200.0}]
    m: dict[str, float] = {}
    add_latency_to_metrics(raw_list, m)
    assert m["latency_p50"] == 150.0
    assert m["mean_latency_ms"] == 150.0