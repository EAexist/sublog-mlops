# Integration tests for benchmark runner

import pytest
from llm_benchmark.benchmark.runner import run_benchmarks, run_benchmarks_multi_task
from tests.utils.mock_dataset import create_mock_dataset


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runner_stub_dataset() -> None:
    dataset = create_mock_dataset(num_samples=1)
    results = await run_benchmarks(dataset, model_ids=[])
    assert isinstance(results, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runner_multi_task_stub() -> None:
    datasets = {
        "t1": create_mock_dataset(num_samples=1),
        "t2": create_mock_dataset(num_samples=0),
    }
    raw_per_task = await run_benchmarks_multi_task(datasets, model_ids=[])
    assert set(raw_per_task.keys()) == {"t1", "t2"}
    assert isinstance(raw_per_task["t1"], dict)
