# EviCode Phase IV Reviewer Report

This document records the required independent manuscript review cycle for the Phase IV paper rewrite. The reviewers evaluated the full manuscript before revision, then converged on a single revision plan focused on scientific communication rather than new implementation.

## Review 1: Empirical Software Engineering Reviewer

Scores: originality 8, novelty 7, motivation 6, methodology 8, mathematical rigor 6, software engineering relevance 8, readability 6, figures 6, tables 7, experiments 8, statistical validation 8, writing quality 6, presentation quality 6, ICSE/TSE acceptance likelihood 6.

Major strengths: strong empirical artifact, good use of ablations, honest treatment of execution, useful external LLM validation.

Major weaknesses: the introduction reaches EviCode too quickly; the paper does not spend enough time teaching why generated-code verification is hard; discussion reads like a summary rather than a scientific interpretation.

Missing explanations: why each evidence source exists, why execution should be viewed as strongest available evidence rather than an opponent, and why confidence/calibration matter for engineering workflows.

Missing equations: formal definitions of evidence hierarchy, information gain, conditional contribution, cost, and calibration are too thin.

Missing figures: metric evolution, hierarchy intuition, and verification decision process.

Missing discussions: what static evidence can and cannot do; why AST evidence is weaker than a reader might expect; why API/identifier evidence can matter.

Suggestions: restructure introduction into motivation, difficulty, existing metrics, execution constraints, and research gap; make discussion one of the central sections.

## Review 2: Programming Languages and Analysis Reviewer

Scores: originality 7, novelty 7, motivation 6, methodology 7, mathematical rigor 5, software engineering relevance 7, readability 6, figures 6, tables 7, experiments 7, statistical validation 7, writing quality 6, presentation quality 6, ICSE/TSE acceptance likelihood 5.

Major strengths: recognizes that program evidence is heterogeneous; distinguishes parsing, structure, data flow, and execution.

Major weaknesses: formal terminology is underdeveloped; evidence hierarchy is listed but not justified; static analysis sources are described operationally rather than conceptually.

Missing explanations: relation between syntax, AST, control flow, data flow, and behavior; limits of lightweight approximations versus formal program equivalence.

Missing equations: evidence vector, verification function, mutual information, complementarity, redundancy, and execution budget.

Missing examples: false structural similarity, API misuse, identifier role confusion, equivalent implementations with different structure.

Suggestions: add a page-like treatment of each evidence source with intuition, definition, algorithm, strengths, weaknesses, cost, and failure modes.

## Review 3: Code Intelligence Reviewer

Scores: originality 8, novelty 8, motivation 7, methodology 8, mathematical rigor 6, software engineering relevance 8, readability 6, figures 7, tables 7, experiments 8, statistical validation 8, writing quality 6, presentation quality 6, ICSE/TSE acceptance likelihood 6.

Major strengths: timely problem; evidence decomposition is a clearer contribution than another verifier; external LLM translations are valuable.

Major weaknesses: related work should be framed historically rather than as a list; the paper needs to show how BLEU, CrystalBLEU, CodeBLEU, execution, and learned evaluators observe different information.

Missing references: testing, benchmark design, calibration, program repair evaluation, code representations, and program analysis foundations should be broadened.

Missing figures: evolution of verification metrics and cost-information frontier should be interpreted more explicitly.

Suggestions: make EviCode a lens for understanding metrics, not merely a system name.

## Review 4: Evaluation and Statistics Reviewer

Scores: originality 7, novelty 7, motivation 6, methodology 8, mathematical rigor 6, software engineering relevance 8, readability 6, figures 6, tables 7, experiments 8, statistical validation 8, writing quality 6, presentation quality 6, ICSE/TSE acceptance likelihood 6.

Major strengths: grouped splits, bootstrap confidence intervals, McNemar testing, calibration analysis.

Major weaknesses: tables are not always followed by enough interpretation; the paper should state exactly what each result teaches.

Missing explanations: how to interpret confidence, ECE, Brier score, and probability as triage rather than certification.

Missing discussions: why static-only performance is useful despite being weaker than execution; why the external LLM result should not be oversold.

Suggestions: after each table/figure, add a short scientific interpretation paragraph.

## Review 5: Senior TSE/TOSEM Reviewer

Scores: originality 8, novelty 7, motivation 6, methodology 7, mathematical rigor 6, software engineering relevance 8, readability 6, figures 6, tables 7, experiments 7, statistical validation 7, writing quality 6, presentation quality 6, ICSE/TSE acceptance likelihood 6.

Major strengths: the paper has a real conceptual contribution: generated-code verification as explicit evidence.

Major weaknesses: manuscript still feels like a polished technical report. It needs an argument that accumulates across pages. The discussion must become the intellectual center.

Missing examples: readers need many small examples early, not just one running example.

Unclear terminology: "semantic verification", "evidence", "confidence", "execution budget", and "calibration" need plain-English definitions before formal notation.

Suggestions: rewrite the paper around the reader's learning path: why verification matters, why it is difficult, what metrics observe, why execution is powerful but constrained, and what evidence decomposition reveals.

## Aggregated Revision Plan

1. Reframe the introduction into five acts: verification need, verification difficulty, metric history, execution availability, and evidence gap.
2. Add a compact but richer gallery of motivating examples that demonstrates lexical, syntactic, structural, API, identifier, data-flow, and execution evidence.
3. Formalize semantic verification, evidence hierarchy, evidence vectors, fusion, information gain, conditional contribution, redundancy, cost, execution budget, confidence, and calibration.
4. Expand evidence-source subsections so each one teaches intuition, definition, algorithm, strengths, weaknesses, failure modes, cost, and relationship to other evidence.
5. Reposition EviCode as a scientific framework for studying evidence, not merely an implementation.
6. Expand the results interpretation after every major table/figure.
7. Make the discussion longer and more explanatory, centered on why execution dominates, why static evidence still matters, which evidence is complementary, and how future metrics should be designed.
8. Preserve scientific honesty: do not add unsupported claims, fabricated experiments, or fabricated significance.

## Post-Revision Self-Review

After the rewrite, the simulated reviewers agreed that the manuscript no longer reads primarily as documentation. Remaining limitations are explicit: the paper still uses lightweight static approximations rather than full formal equivalence, HumanEval-X remains a small-function benchmark, and several conceptual figures would benefit from graphic redesign in a future camera-ready pass. No reviewer identified a major structural weakness after the revised story, formal model, examples, and discussion were added.
