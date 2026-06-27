# EviCode

**EviCode: Understanding Static and Dynamic Evidence in Semantic Verification of Machine-Generated Code**

EviCode studies whether decomposed evidence sources can estimate and explain semantic correctness of generated code when execution evidence is incomplete, weak, or unavailable.

The first application is executable code translation verification. The current research framing is an empirical study of verification evidence: informativeness, complementarity, cost, execution-budget behavior, and failure explanation.

## Core Research Question

What combination of static evidence best approximates execution-based semantic verification under varying execution budgets, and which evidence sources contribute the most?

The verifier should not only answer whether generated code is likely correct. It should explain why:

- high or low syntax confidence,
- structural similarity or mismatch,
- control-flow preservation or drift,
- API and identifier consistency,
- retrieval support,
- execution success or failure under the available test budget.

The classifier is not the central contribution. It is a transparent tool for combining evidence so the project can quantify which evidence sources matter.

## Evidence Sources

EviCode represents verification evidence as modular sources:

- `E1`: syntax evidence
- `E2`: AST similarity evidence
- `E3`: control-flow evidence
- `E4`: API consistency evidence
- `E5`: identifier mapping evidence
- `E6`: execution evidence
- `E7`: retrieval similarity evidence

Each evidence module exposes:

- `extract()`
- `score()`
- `explain()`

## Repository Status

The first complete HumanEval-X benchmark has been run for Python, Java, and JavaScript.

- Dataset: 2,952 directed verification examples from 164 tasks.
- Labels: 984 positive pairs and 1,968 controlled negative pairs.
- Evidence: parser-backed static evidence plus example-test and full-test execution evidence.
- Analysis: source informativeness, group complementarity, cost, execution-budget status, weak-test status, and failure explanations.
- Fusion: logistic evidence models evaluated with grouped problem-level splitting as an analysis instrument.
- External LLM validation: Python-to-Java predictions from DeepSeekCoder, QwenCoder, and StarCoder are evaluated as confidence and calibration data.
- Statistics: bootstrap F1 intervals and exact McNemar paired comparisons.
- Paper: single-file manuscript at `paper/main.tex`, compiled to `paper/output/paper.pdf`.
- Manuscript status: Phase IV rewrite reframes the artifact as a mature software engineering research paper about why semantic verification is difficult, what evidence each metric observes, why execution is powerful but availability-limited, and how decomposed evidence supports explanation and calibrated triage.
- Phase IV reviewer record: `docs/PHASE4_REVIEWER_REPORT.md`.

See:

- `SETUP.md`
- `REPRODUCIBILITY.md`
- `PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`

The paper is maintained as a single LaTeX file at `paper/main.tex`. After updating it, run `python scripts/build_paper.py --config configs/humanevalx.yaml --output-dir paper/output --resume --force` to regenerate `paper/output/paper.pdf`.

## Smoke Test

After installing dependencies:

```bash
python -m pytest
python scripts/run_smoke.py --config configs/smoke.yaml --output-dir experiments/smoke --resume
```

Install with `python -m pip install -e .` after dependency installation so the `src/` package is importable.

## Full Reproduction

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
