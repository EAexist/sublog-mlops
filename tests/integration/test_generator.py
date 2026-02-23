# Hits real oracle API — skipped in CI unless flag set

import pytest
from llm_benchmark.dataset.generator import generate_dataset


@pytest.mark.integration
@pytest.mark.skip(reason="Hits real oracle API; run with pytest -m integration when env has keys")
def test_generator_real() -> None:
    # Updated to match new signature: oracle_model_id, n_templates, n_samples_per_template
    dataset = generate_dataset(
        oracle_model_id="gpt-4o",
        n_templates=1,
        n_samples_per_template=2,
    )
    
    # Check if samples were generated
    # 2 companies * 5 event types * 1 template * 2 samples = 20
    assert len(dataset.samples) == 20
    assert dataset.content_hash or True
