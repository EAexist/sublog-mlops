---
name: config-reference
description: Full reference for config/models.yaml and config/benchmark.yaml. Use when editing config files or when the user asks for config structure or defaults.
---

# Full Config Reference

## `config/models.yaml`

```yaml
models:
  - id: gpt-4o
    provider: openai
    model_string: gpt-4o
    input_price_per_1k_tokens: 0.005
    output_price_per_1k_tokens: 0.015
  - id: gemini-2.5-pro
    provider: google
    model_string: gemini-2.5-pro
    input_price_per_1k_tokens: 0.00125
    output_price_per_1k_tokens: 0.010
  - id: llama3-70b
    provider: ollama
    model_string: llama3:70b
    input_price_per_1k_tokens: 0.0
    output_price_per_1k_tokens: 0.0
```

## `config/benchmark.yaml`

Multi-task: each task is benchmarked independently; composite score is weighted across tasks and metrics.

```yaml
oracle_model_id: gpt-4o
metric_weights:
  correctness: 0.60
  latency: 0.25
  cost: 0.15
output_dir: outputs/runs
mlflow_experiment_name: llm-benchmark
tasks:
  - task_id: coding_qa
    domain: coding_qa
    n_samples: 50
    weight: 1.0
  - task_id: summarization
    domain: summarization
    n_samples: 30
    weight: 0.8
```