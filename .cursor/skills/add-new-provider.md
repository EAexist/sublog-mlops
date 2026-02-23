---
name: add-new-provider
description: Add a new LLM provider (client implementation). Use when adding a new provider in config/models.yaml or when implementing a new *_client.py for a provider.
---

# Add a New Provider

1. Create `llm_benchmark/models/<provider>_client.py`.
2. Implement the `LLMClient` abstract base (see `models/base.py`):
   - `async def complete(prompt: str) -> LLMResponse`
   - `LLMResponse` must include: `content`, `prompt_tokens`, `completion_tokens`, `latency_ms`
3. Register in `models/registry.py` — add the provider string to the match block.
4. Add an Airflow Connection for the API key (see **airflow-connections** skill).