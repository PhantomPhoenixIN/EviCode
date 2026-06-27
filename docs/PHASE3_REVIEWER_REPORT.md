# Phase III Independent Reviewer Report

Date: 2026-06-25

This report records the simulated program-committee review used to guide the Phase III manuscript rewrite.

## Reviewer 1: Software Engineering Researcher

**Summary.** The artifact studies semantic verification of generated code through decomposed evidence sources. The implementation is credible, but the original manuscript framed the work as a framework description rather than an empirical software engineering contribution.

**Strengths.** Important problem; reproducible artifact; grouped split; evidence ablations; cost and calibration analyses; honest reporting of incomplete execution-budget experiments.

**Weaknesses.** Motivation was too compressed; the paper started with EviCode instead of the verification problem; contributions were not sharply separated from implementation; results needed more interpretation.

**Novelty assessment.** Moderate to strong if positioned as an empirical study of evidence, weaker if positioned as another verifier.

**Methodology assessment.** Reasonable, but synthetic negatives and Python-to-Java-only external validation must be made explicit.

**Writing assessment.** Needed a clearer story, stronger transitions, and explicit takeaways.

**Presentation assessment.** Tables and figures needed to answer scientific questions rather than list artifact outputs.

**Missing references.** CodeBLEU, CrystalBLEU, CodeXGLUE, HumanEval, HumanEval-X, MultiPL-E, TransCoder, empirical SE methodology.

**Missing experiments.** Broader real-candidate validation and normalized weak-test budgets.

**Missing equations.** Evidence source, fusion, cost, complementarity, marginal contribution.

**Missing figures.** Metric limitations, evidence hierarchy, execution-budget intuition.

**Missing tables.** Terminology and evidence-source definitions.

**Missing explanations.** Why static evidence helps, why execution remains dominant, and when calibration matters.

**Recommended score.** Borderline reject before revision; weak accept after major narrative rewrite.

**Confidence.** 4/5.

## Reviewer 2: Program Analysis Expert

**Summary.** The work uses program-analysis terms but originally did not define their approximations clearly enough. The rewrite should distinguish full static analysis from lightweight evidence extraction.

**Strengths.** Separates syntax, AST, control flow, API, identifier, data-flow, type, retrieval, and execution evidence. Acknowledges approximation limits.

**Weaknesses.** Original evidence descriptions lacked formal definitions, algorithms, cost discussion, and examples. AST similarity and data-flow approximations could be overclaimed.

**Novelty assessment.** Stronger as a measurement study of approximate evidence than as a program verifier.

**Methodology assessment.** Needs precise definitions of evidence extraction and cost.

**Writing assessment.** Should teach the reader what each evidence source captures.

**Presentation assessment.** Add a running example to show where lexical similarity, AST shape, API consistency, and execution disagree.

**Missing references.** Classic static analysis, abstract interpretation, program equivalence, semantic code search, tree-based code models.

**Missing experiments.** Richer semantic static analysis would be valuable but is not required if limitations are clear.

**Missing equations.** Evidence source functions, cost and complementarity.

**Missing figures.** Evidence hierarchy and extraction example.

**Missing tables.** Evidence source table with strengths and limitations.

**Missing explanations.** Difference between approximate evidence and formal semantic proof.

**Recommended score.** Weak reject before revision; borderline after stronger definitions.

**Confidence.** 4/5.

## Reviewer 3: Machine Learning for Code Researcher

**Summary.** The work is useful if it is framed around evaluation and calibration rather than classifier novelty.

**Strengths.** Uses grouped splits, bootstrap intervals, McNemar tests, external LLM predictions, and calibration metrics.

**Weaknesses.** Original manuscript underexplained why logistic regression is appropriate and why probability calibration is useful. Baselines needed clearer discussion.

**Novelty assessment.** Moderate; strongest contribution is decomposed evidence evaluation under limited execution.

**Methodology assessment.** Good enough for an empirical paper, provided synthetic and external settings are separated.

**Writing assessment.** Needed clearer RQs and direct answers.

**Presentation assessment.** Calibration and score-grade figures should be connected to real downstream decisions.

**Missing references.** Learned code models, CodeXGLUE, GraphCodeBERT, CodeBERT, pass@k, calibration literature.

**Missing experiments.** More LLMs/languages and confidence calibration on additional domains.

**Missing equations.** Fusion function, expected calibration error, marginal contribution.

**Missing figures.** Reliability curve interpretation and static-versus-dynamic evidence story.

**Missing tables.** Baseline comparison table and per-score confidence table.

**Missing explanations.** Why high threshold accuracy is not the right primary use of confidence.

**Recommended score.** Borderline before revision; weak accept after rewrite.

**Confidence.** 4/5.

## Reviewer 4: Senior Journal Editor

**Summary.** The original manuscript lacked maturity in scientific exposition. It needed a broader introduction, theme-organized related work, formalization, and discussion.

**Strengths.** Clear artifact and reproducible outputs; potential to teach a useful lesson.

**Weaknesses.** Too many short paragraphs; abrupt section changes; insufficient interpretation after tables; limitations mixed with future work.

**Novelty assessment.** Acceptable if the paper presents new knowledge, not only a tool.

**Methodology assessment.** Transparent but must be narrated carefully.

**Writing assessment.** Needs a stronger authorial voice and smoother transitions.

**Presentation assessment.** Must define terminology and use it consistently.

**Missing references.** Broad SE and ML-for-code evaluation literature.

**Missing experiments.** Not fatal, but missing ones must be declared as future work.

**Missing equations.** Formal definitions should support clarity, not decoration.

**Missing figures.** Conceptual figures that orient non-specialists.

**Missing tables.** Terminology table.

**Missing explanations.** Why each result matters.

**Recommended score.** Reject before revision; weak accept after extensive writing revision.

**Confidence.** 5/5.

## Reviewer 5: General Computer Science Reader

**Summary.** The problem is interesting, but the original manuscript assumed too much prior knowledge and did not explain semantic verification intuitively.

**Strengths.** Concrete setting; useful examples can make the idea accessible.

**Weaknesses.** Needed examples showing why `a + b`, `a - b`, and `b + a` expose the weakness of textual metrics.

**Novelty assessment.** Clear once framed as decomposed evidence rather than a black-box model.

**Methodology assessment.** Understandable if the paper explains labels, examples, and tests plainly.

**Writing assessment.** Should be more educational.

**Presentation assessment.** Figures and tables need standalone explanations.

**Missing references.** Less important than intuition, but related work should not be a list.

**Missing experiments.** Not obvious to a general reader.

**Missing equations.** Only useful if explained in plain English.

**Missing figures.** Running example and evidence hierarchy.

**Missing tables.** Terminology table.

**Missing explanations.** What a practitioner should do with a confidence score.

**Recommended score.** Borderline before revision; weak accept if rewritten clearly.

**Confidence.** 3/5.

## Revision Decisions

The Phase III rewrite addresses the shared major concerns by:

- starting from the semantic verification problem rather than EviCode;
- adding a running example that exposes lexical, structural, API, and execution evidence;
- adding terminology and formal definitions;
- expanding related work thematically;
- adding per-evidence-source explanations with motivation, definition, algorithm, strengths, limitations, cost, and takeaway;
- explicitly interpreting tables and figures;
- separating HumanEval-X synthetic analysis from external Python-to-Java LLM calibration;
- replacing the old self-review with threats to validity and a clearer discussion.

