# Dataset Guide

Primary target: HumanEval-X.

The current completed benchmark uses HumanEval-X for Python, Java, and JavaScript. Dataset scripts are resume-safe and must never overwrite data unless `--force` is provided.

Completed stages:

1. Download/cache HumanEval-X.
2. Build translation-pair records.
3. Generate positive and controlled negative pairs.
4. Attach test cases where available.
5. Save JSONL shards and manifests.

Current processed output:

```text
datasets/processed/humanevalx/verification_examples.jsonl
datasets/processed/humanevalx/manifest.json
```

The benchmark contains 2,952 directed verification examples from 164 tasks: 984 positive pairs and 1,968 controlled negative pairs.
