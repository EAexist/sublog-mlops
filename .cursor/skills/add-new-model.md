---
name: add-new-model
description: Add a new LLM to the benchmark by editing config only. Use when adding a model to config/models.yaml or when the user asks to add a new model.
---

# Add a New Model

1. Add one block to `config/models.yaml`:

```yaml
- id: your-model-id
  provider: openai | google | ollama
  model_string: exact-api-string
  input_price_per_1k_tokens: 0.000
  output_price_per_1k_tokens: 0.000
```

2. If the provider already exists → done; the runner picks it up automatically.
3. If the provider is new → also apply the **add-new-provider** skill.