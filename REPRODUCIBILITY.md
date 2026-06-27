# Reproducibility

EviCode is designed around resume-safe, auditable experiments.

## Dependency Policy

- Core Python dependencies live in `requirements.txt`.
- Development dependencies live in `requirements-dev.txt`.
- Conda users can use `environment.yml`.
- Every newly installed package must be recorded immediately.

## Resume-Safety Contract

Every script must support:

```bash
--resume
--force
--dry-run
--config
--output-dir
```

Long-running scripts must:

- skip valid completed outputs unless `--force` is provided,
- write partial JSONL/CSV outputs incrementally,
- flush outputs regularly,
- save status files before and after major stages,
- record failures without stopping the full pipeline when possible,
- allow restart after interruption.

## Progress Files

Progress is tracked under:

```text
progress/progress.json
progress/completed_tasks.json
progress/failed_tasks.json
progress/todo.json
progress/logs/
```

## Smoke Reproduction Target

A fresh machine should be able to:

1. Clone or copy this repository.
2. Install Python and OS-level toolchains described in `SETUP.md`.
3. Install dependencies from `requirements-dev.txt`.
4. Install the package in editable mode with `python -m pip install -e .`.
5. Run unit tests.
6. Run the smoke pipeline.
7. Build the draft paper assets and PDF where LaTeX is available.

Commands:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest
python scripts/run_smoke.py --config configs/smoke.yaml --output-dir experiments/smoke --resume
python scripts/build_paper.py --config configs/smoke.yaml --output-dir paper/output --resume
```

The smoke test also writes `experiments/smoke/verification_reports.json`, which verifies that EviCode emits evidence-grounded explanations in addition to a score.

The manuscript is intentionally maintained as a single file, `paper/main.tex`. The paper build script regenerates tables, figures, and `paper/output/paper.pdf` from the current repository state.

The Phase III reviewer report that motivated the earlier manuscript rewrite is recorded in `docs/PHASE3_REVIEWER_REPORT.md`.

The Phase IV top-tier software engineering manuscript review and revision record is saved at `docs/PHASE4_REVIEWER_REPORT.md`. It documents the five independent simulated reviews, the aggregated revision plan, and the post-revision self-review used to guide the current manuscript rewrite.

The smoke test is the minimum reproducibility target.

## Full Benchmark Reproduction

After the smoke test succeeds and Python, Node.js, Java, and LaTeX are available, run:

```bash
python scripts/build_humanevalx.py --config configs/humanevalx.yaml --output-dir datasets/processed/humanevalx --resume
python scripts/extract_evidence.py --config configs/humanevalx.yaml --input datasets/processed/humanevalx/verification_examples.jsonl --output-dir experiments/humanevalx/evidence --resume
python scripts/refresh_static_evidence.py --config configs/humanevalx.yaml --examples datasets/processed/humanevalx/verification_examples.jsonl --evidence experiments/humanevalx/evidence/evidence.jsonl --output-dir experiments/humanevalx/evidence_rich --resume
python scripts/run_experiments.py --config configs/humanevalx.yaml --input experiments/humanevalx/evidence_rich/evidence.jsonl --output-dir experiments/humanevalx/fusion_rich --resume
python scripts/statistical_analysis.py --config configs/humanevalx.yaml --predictions experiments/humanevalx/fusion_rich/predictions.csv --output-dir statistics/humanevalx_rich --resume
python scripts/analyze_evidence.py --config configs/humanevalx.yaml --examples datasets/processed/humanevalx/verification_examples.jsonl --evidence experiments/humanevalx/evidence_rich/evidence.jsonl --metrics experiments/humanevalx/fusion_rich/metrics.csv --output-dir results/analysis --resume
python scripts/execution_budget_analysis.py --config configs/humanevalx.yaml --evidence experiments/humanevalx/evidence_rich/evidence.jsonl --metrics experiments/humanevalx/fusion_rich/metrics.csv --output-dir results/execution_budget --resume
python scripts/weak_test_analysis.py --config configs/humanevalx.yaml --metrics experiments/humanevalx/fusion_rich/metrics.csv --output-dir results/weak_tests --resume
python scripts/phase2_scientific_analysis.py --config configs/humanevalx.yaml --examples datasets/processed/humanevalx/verification_examples.jsonl --evidence experiments/humanevalx/evidence_rich/evidence.jsonl --predictions experiments/humanevalx/fusion_rich/predictions.csv --output-dir results/phase2 --resume
python scripts/evaluate_llm_predictions.py --config configs/humanevalx.yaml --predictions-dir datasets/Predictions_by_LLMs --train-evidence experiments/humanevalx/evidence_rich/evidence.jsonl --output-dir results/llm_predictions --resume
python scripts/generate_artifacts.py --config configs/humanevalx.yaml --dataset datasets/processed/humanevalx/verification_examples.jsonl --evidence experiments/humanevalx/evidence_rich/evidence.jsonl --metrics experiments/humanevalx/fusion_rich/metrics.csv --statistics-dir statistics/humanevalx_rich --output-dir results/humanevalx_rich --resume
python scripts/build_paper.py --config configs/humanevalx.yaml --output-dir paper/output --resume --force
```

The completed benchmark should produce `paper/output/paper.pdf`, `experiments/humanevalx/fusion_rich/metrics.csv`, `statistics/humanevalx_rich/bootstrap_f1.csv`, `statistics/humanevalx_rich/mcnemar.csv`, external LLM confidence outputs under `results/llm_predictions`, and the analysis outputs under `results/analysis`, `results/cost`, `results/execution_budget`, `results/weak_tests`, `results/failure_analysis`, and `results/phase2`.
