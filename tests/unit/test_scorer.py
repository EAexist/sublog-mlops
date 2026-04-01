# Unit tests for benchmark/scorer

import pytest
from llm_benchmark.benchmark.scorer import build_benchmark_result


def test_build_benchmark_result_empty() -> None:
    result = build_benchmark_result({}, {})
    assert result.best_model_id == ""
    assert result.scores == {}
    assert result.per_task_metrics == {}


def test_build_benchmark_result_single_task() -> None:
    per_task = {"t1": {"gpt-4o": {"correctness": 1.0, "cost": 0.01}}}
    task_weights = {"t1": 1.0}
    result = build_benchmark_result(per_task, task_weights)
    assert result.best_model_id == ""
    assert result.scores == {}
    assert result.per_task_metrics == per_task
    assert "gpt-4o" in result.metrics_by_model
    assert result.metrics_by_model["gpt-4o"]["correctness"] == 1.0


def test_build_benchmark_result_multi_task() -> None:
    per_task = {
        "t1": {"gpt-4o": {"correctness": 0.8, "cost": 0.02}},
        "t2": {"gpt-4o": {"correctness": 1.0, "cost": 0.01}},
    }
    task_weights = {"t1": 1.0, "t2": 0.5}
    result = build_benchmark_result(per_task, task_weights)
    assert result.best_model_id == ""
    assert list(result.per_task_metrics.keys()) == ["t1", "t2"]
    assert result.metrics_by_model["gpt-4o"]["correctness"] == pytest.approx(0.9)  # (0.8 + 1.0) / 2
