---
name: relay-to-client-repo
description: Implement pushing benchmark best model to client repo (Feature 6). Use when implementing reporting/relay.py or adding human approval before report.
---

# Implement Feature 6 (Relay to Client Repo)

**Stub location:** `reporting/relay.py → push_to_client_repo(result: BenchmarkResult)`

1. `BenchmarkResult.best_model_id` is already populated by `scorer.py`.
2. Add a `human_approval_sensor` task between `score_and_rank` and `generate_report` in the DAG.
3. Implement `push_to_client_repo`: write `best_model_id` to a JSON file and open a PR or push to client repo via GitHub API.