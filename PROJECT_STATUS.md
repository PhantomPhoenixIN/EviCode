# Project Status

Date: 2026-06-25

## Current State

- Repository scaffold created and expanded into a reproducible benchmark artifact.
- Dependency manifests created and maintained.
- Python package metadata created in `pyproject.toml`.
- Resume-safety policy documented and implemented across dataset, evidence, experiment, statistics, artifact, and paper scripts.
- Core evidence/fusion/explanation interfaces implemented.
- Python 3.11, Git, Node.js, a project-local JDK 17, and MiKTeX are available for this project.
- Unit tests pass.
- Synthetic smoke test runs and writes evidence vectors plus verification reports.
- HumanEval-X Python, Java, and JavaScript dataset construction completed.
- Full evidence extraction completed for 2,952 directed verification examples.
- Parser-backed static evidence refresh completed.
- Logistic evidence-fusion experiments completed with grouped problem-level splitting.
- Bootstrap F1 intervals and exact McNemar comparisons generated.
- Evidence taxonomy, informativeness, complementarity, cost, execution-budget status, weak-test status, and failure-analysis outputs generated.
- Phase II scientific analysis generated metric comparison, hierarchy, feature attribution, language-specific rankings, failure detection matrices, Pareto analysis, metric-design guidelines, self-review, and generated-candidate schema/status files.
- External Python-to-Java LLM predictions from DeepSeekCoder, QwenCoder, and StarCoder evaluated for confidence and calibration.
- Phase IV manuscript rewrite completed: the paper now opens with the software engineering need for verification, explains why verification is difficult through multiple examples, frames prior metrics as progressively richer evidence, distinguishes execution strength from execution availability, expands the formal evidence model, deepens evidence-source explanations, and turns the discussion into an interpretation-centered section.
- Phase IV five-reviewer manuscript critique and post-revision self-review saved in `docs/PHASE4_REVIEWER_REPORT.md`.
- Narrative sharpening pass completed: the manuscript now uses a scientific empirical-study title, adds a conceptual evidence-ladder figure, reports interpretive findings such as weak marginal AST contribution and stronger identifier-role evidence, adds qualitative disagreement cases, expands discussion of CodeBLEU/AST/execution/API behavior, and strengthens the conclusion.
- Paper build script generates tables, figures, and a draft PDF from completed outputs.
- The manuscript is maintained in a single file: `paper/main.tex`.

## Current Framing

EviCode now centers on this question:

> What combination of static evidence best approximates execution-based semantic verification under varying execution budgets, and which evidence sources contribute the most?

The verifier is expected to produce both a score and a structured explanation.

## Blockers

- The main HumanEval-X negative examples are constructed from task mismatch and mutation; external real LLM predictions are now included as a separate Python-to-Java validation setting.
- Static evidence is still lightweight compared with full semantic static analysis.
- HumanEval-X tests are not yet normalized into individual cross-language assertions, so 1/3/5/10-test and randomized weak-test dynamic experiments are marked pending rather than fabricated.
- Related work and bibliography were expanded for Phase IV; final submission would still benefit from a manual citation audit against the target venue style.
- Additional external validation datasets beyond the current Python-to-Java LLM prediction files remain future work.

## Next Milestone

Polish the Phase IV manuscript for target venue length, citation style, and camera-ready figure quality. The current full reproduction path is:

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
