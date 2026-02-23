# Unit tests for benchmark/metrics (accuracy + cost only; no latency in main flow)

import pytest
from llm_benchmark.benchmark.metrics import compute_cost, compute_task_metrics
from llm_benchmark.config_loader import ModelEntry
from tests.utils.mock_dataset import create_mock_dataset


def test_compute_cost() -> None:
    c = compute_cost(1000, 500, 0.001, 0.002)
    assert c == pytest.approx(0.001 + 0.001)  # 1*0.001 + 0.5*0.002


def test_compute_task_metrics() -> None:
    dataset = create_mock_dataset(num_samples=1)
    raw_by_model = {
        "m1": [
            {
                "content": "out",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "latency_ms": 100.0,
            }
        ],
    }
    entries = [
        ModelEntry(
            id="m1",
            provider="openai",
            model_string="gpt-4o",
            input_price_per_1k_tokens=0.001,
            output_price_per_1k_tokens=0.002,
        )
    ]
    by_id = {e.id: e for e in entries}
    out = compute_task_metrics(dataset, raw_by_model, by_id)
    assert "m1" in out
    assert "correctness" in out["m1"]
    assert "cost" in out["m1"]
    assert out["m1"]["cost"] == pytest.approx(0.00002)  # 10/1k*0.001 + 5/1k*0.002
