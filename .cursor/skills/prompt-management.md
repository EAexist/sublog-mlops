---
name: prompt-management
description: Implement versioned prompt management and A/B test DAG (Feature 8). Use when implementing PromptVersion, data/prompts/, or prompt_ab_test_dag.
---

# Implement Feature 8 (Prompt Management)

**Stub location:** `dataset/schema.py → PromptVersion(id, content, created_at, is_active)`

1. Store versioned prompts as YAML files in `data/prompts/`.
2. Only one prompt may have `is_active: true` at a time — enforce in the loader.
3. Create `dags/prompt_ab_test_dag.py`: run benchmark twice (active vs. candidate prompt), compare correctness scores, log both to MLflow under the same experiment.