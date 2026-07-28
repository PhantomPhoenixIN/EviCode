# Reproducibility

This repository is organized around the current EviCode paper:

**EviCode: Interpretable Reference- and Execution-Free Verification of Code Translation Using Language-Normalized Semantic Evidence**

The active manuscript is `paper_v2/main_v2.tex`. The compiled PDF is intentionally not tracked in GitHub.

## Dependency Policy

- Core dependencies are listed in `requirements.txt`.
- Development dependencies are listed in `requirements-dev.txt`.
- Current-paper research dependencies are listed in `requirements-research.txt`.
- Install the package in editable mode before running experiments.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-research.txt
python -m pip install -e .
```

## Current Dataset

The current paper uses scored CodeNetTrans-QS generator outputs stored under:

```text
datasets/Predictions_by_LLMs/
```

The controlled study fixes Java as the target language and retains the seven source languages listed in:

```text
authentic_only_study/config.yaml
```

The extractor excludes target references, execution outcomes, compiler outcomes, test results, generator identity, and problem identifiers from the feature matrix.

## Current Pipeline

Run static extraction and the primary analysis:

```bash
python authentic_only_study/extract.py --config authentic_only_study/config.yaml
python authentic_only_study/analyze.py --config authentic_only_study/config.yaml
```

Run the probe-basis audit and same-information baselines:

```bash
python scripts/feature_basis_audit.py
python scripts/authentic_same_information_baselines.py
```

Run learner-capacity analysis for the explicit EviCode representation:

```bash
python scripts/model_capacity_ablation.py
```

If frozen CodeBERT, GraphCodeBERT, UniXcoder, and CodeT5 embeddings are available under `outputs/`, regenerate the representation comparison:

```bash
python scripts/build_full_cross_comparison.py
python scripts/analyze_representation_matrix.py
```

Regenerate current-paper figures:

```bash
python authentic_only_study/visualizations.py
python authentic_only_study/presentation_figures.py
python authentic_only_study/conceptual_figure.py
```

The paper uses the figure PDFs tracked under:

```text
outputs/authentic_only/figures/
```

## Build the Current Manuscript

```bash
cd paper_v2
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
```

The generated PDF remains local because `paper_v2/*.pdf` is ignored.

## Validation Notes

The principal split uses seed 42 and groups by `problem_id`. Bootstrap intervals resample held-out problem clusters and are conditional on the fitted model. The Score-2-versus-3 conditional results apply the primary model to the conditioned test subset rather than fitting a separate classifier.
