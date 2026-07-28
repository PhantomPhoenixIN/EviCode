# Reproduction Guide

The current reproducible target is the CodeNetTrans-QS EviCode paper.

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-research.txt
python -m pip install -e .
```

Run tests:

```bash
python -m pytest
```

Run the current extraction and analysis:

```bash
python authentic_only_study/extract.py --config authentic_only_study/config.yaml
python authentic_only_study/analyze.py --config authentic_only_study/config.yaml
python scripts/feature_basis_audit.py
python scripts/authentic_same_information_baselines.py
python scripts/model_capacity_ablation.py
```

Regenerate current-paper figures:

```bash
python authentic_only_study/visualizations.py
python authentic_only_study/presentation_figures.py
python authentic_only_study/conceptual_figure.py
```

Build the manuscript locally:

```bash
cd paper_v2
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
```

The compiled PDF is ignored and should remain local.
