# Pytest fixtures and config

import pytest


@pytest.fixture
def sample_models_config_path(tmp_path):
    """Path to a minimal models.yaml."""
    p = tmp_path / "models.yaml"
    p.write_text("""
models:
  - id: stub
    provider: openai
    model_string: gpt-4o
    input_price_per_1k_tokens: 0.0
    output_price_per_1k_tokens: 0.0
""")
    return p
