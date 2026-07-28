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

The current CodeNetTrans-QS manuscript is maintained in
`paper_v2/main_v2.tex`. The earlier HumanEval-X manuscript remains in
`paper/main.tex` and is not overwritten by the current build.

The smoke test is the minimum reproducibility target.

## Current 45-Probe Basis Audit

The current manuscript uses every eligible, nonduplicate pairwise output of the
fixed static extractor. Reproduce the 53-to-49-to-45 filtering funnel,
training-partition dependence checks, and feature-count sensitivity analysis
with:

```bash
python scripts/feature_basis_audit.py
```

The command writes the complete eligibility manifest, exact-duplicate pairs,
high-correlation pairs, selection funnel, parser-independent and
feature-count sensitivity results, and a machine-readable summary under
`outputs/authentic_only/feature_basis/`. The filtering rules do not inspect
correctness labels or held-out performance. Labels are used only afterward to
fit the sensitivity models on the fixed training partition.

## Current CodeNetTrans-QS Study

Install the research dependencies in addition to the core package:

```bash
python -m pip install -r requirements-research.txt
```

The scored CodeNetTrans-QS JSONL files are read from
`datasets/Predictions_by_LLMs/`.

The controlled study fixes Java as the target and retains the seven source
languages listed in `authentic_only_study/config.yaml`. Each retained language
has an implemented front end, appears under all three generators, and has at
least 1,184 eligible Score-1-to-3 records. The raw release also contains Haxe,
but only 103 eligible records, including three Score-3 StarCoder outputs, and
the artifact has no Haxe front end. Haxe is therefore excluded before model
fitting so that the language analysis does not mix severe sparsity with a
different extraction path.

Run the static extraction and primary analysis with:

```bash
python authentic_only_study/extract.py --config authentic_only_study/config.yaml
python authentic_only_study/analyze.py --config authentic_only_study/config.yaml
python scripts/feature_basis_audit.py
python scripts/authentic_same_information_baselines.py
python scripts/model_capacity_ablation.py
```

The frozen CodeBERT, GraphCodeBERT, UniXcoder, and CodeT5 embeddings are
preserved under `outputs/`. The GPU extraction notebook is
`colab_execution.ipynb`. Once the frozen representation result bundle has been
restored under `outputs/frozen_representation_ablation_full/`, regenerate the
combined learner matrix and paired comparisons with:

```bash
python scripts/build_full_cross_comparison.py
python scripts/analyze_representation_matrix.py
```

Regenerate the reader-facing figures from completed CSV and Parquet artifacts:

```bash
python authentic_only_study/review_visualizations.py
python authentic_only_study/presentation_figures.py
python authentic_only_study/conceptual_figure.py
```

Build `main_v2.pdf` without modifying the earlier paper:

```bash
cd paper_v2
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
bibtex main_v2
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
```

The principal split uses seed 42 and groups by `problem_id`. Bootstrap
intervals resample held-out problem clusters and are conditional on that split
and fitted model. The Score-2-versus-3 results apply the primary model to the
conditioned test subset rather than fitting a second model.

## Legacy HumanEval-X Benchmark

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
python scripts/semantic_accumulation_analysis.py --config configs/humanevalx.yaml --examples datasets/processed/humanevalx/verification_examples.jsonl --evidence experiments/humanevalx/evidence_rich/evidence.jsonl --predictions experiments/humanevalx/fusion_rich/predictions.csv --output-dir results/semantic_accumulation --resume
python scripts/evaluate_llm_predictions.py --config configs/humanevalx.yaml --predictions-dir datasets/Predictions_by_LLMs --train-evidence experiments/humanevalx/evidence_rich/evidence.jsonl --output-dir results/llm_predictions --resume
python scripts/external_dataset_validation.py --config configs/humanevalx.yaml --train-evidence experiments/humanevalx/evidence_rich/evidence.jsonl --output-dir results/external_validation --train-source-language python --train-target-language java --max-pairs 200 --resume
python scripts/generate_artifacts.py --config configs/humanevalx.yaml --dataset datasets/processed/humanevalx/verification_examples.jsonl --evidence experiments/humanevalx/evidence_rich/evidence.jsonl --metrics experiments/humanevalx/fusion_rich/metrics.csv --statistics-dir statistics/humanevalx_rich --output-dir results/humanevalx_rich --resume
python scripts/build_paper.py --config configs/humanevalx.yaml --output-dir paper/output --resume --force
```

The completed benchmark should produce `paper/output/paper.pdf`, `experiments/humanevalx/fusion_rich/metrics.csv`, `statistics/humanevalx_rich/bootstrap_f1.csv`, `statistics/humanevalx_rich/mcnemar.csv`, external LLM confidence outputs under `results/llm_predictions`, and analysis outputs under `results/analysis`, `results/cost`, `results/execution_budget`, `results/weak_tests`, `results/failure_analysis`, `results/phase2`, and `results/semantic_accumulation`.
