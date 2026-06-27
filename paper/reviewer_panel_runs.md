# Reviewer Panel Runs

This file records three independent senior-reviewer passes over the manuscript prompt dated 2026-06-26. The runs were used to guide narrative and presentation edits only. No datasets, experiments, numerical values, or statistical tests were changed.

## Run 1: Novelty and Scientific Framing

Strengths:
- The manuscript has a clear core idea: semantic verification should be studied as evidence rather than as a single metric.
- The evidence hierarchy, complementarity analysis, calibration analysis, and disagreement analysis form a coherent contribution.

Weaknesses:
- The introduction could still be misread as proposing a verifier rather than studying verification evidence.
- The novelty claim needed a sharper distinction between feature fusion and evidence measurement.

Applied changes:
- Added a paragraph clarifying that the central object of study is the evidence a verifier observes, not merely a trained verifier.
- Strengthened the distinction between aggregate scores and explanations of why generated programs should be trusted.

## Run 2: Methodology and Formalization

Strengths:
- The formal model already defines correctness, evidence vectors, mutual information, calibration, cost, complementarity, and execution budgets.
- The equations are appropriate for a journal-style empirical study.

Weaknesses:
- Some equations needed stronger plain-English motivation explaining the scientific concept being modeled.
- The evidence vector needed to be framed as separating observation from judgment.

Applied changes:
- Expanded the explanation of the evidence vector as a way to separate observable signals from verification decisions.
- Added a synthesis sentence explaining how information, complementarity, calibration, and cost together define semantic verification as a measurement problem.

## Run 3: Writing Quality and Journal Impact

Strengths:
- The Results, Discussion, and Guidelines sections now read more like a scientific argument than a leaderboard.
- The paper already avoids overclaiming by qualifying conclusions to the evaluated settings.

Weaknesses:
- The evidence-source section benefited from a clearer observability-based framing.
- The Results section needed an opening paragraph explaining that tables and figures support scientific claims rather than merely reporting artifacts.

Applied changes:
- Reframed the evidence taxonomy around observability: text, grammar, structure, behavior-proximal obligations, and sampled behavior.
- Added a Results and Analysis opening paragraph stating that the section explains why evidence sources succeed, fail, complement one another, or become redundant.
- Softened one execution claim from "dominate" to "receive the greatest trust" when full tests are available.

## Master Review Outcome

The manuscript is strongest when it argues that EviCode is an empirical study of verification evidence. The final edits therefore emphasize:

- Evidence decomposition rather than feature engineering.
- Measurement of semantic information rather than classifier novelty.
- Explanation and triage rather than binary verification alone.
- Execution as the strongest practical evidence in the evaluated settings, but not as semantic equivalence.
- Static evidence as useful for ranking, diagnosis, routing, and explaining failures.

No experimental results were modified.
