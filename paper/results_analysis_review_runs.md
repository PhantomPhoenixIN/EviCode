# Results and Analysis Review Runs

This file records three independent passes over the Results and Analysis improvement prompt dated 2026-06-26. The edits were narrative and interpretive only. No datasets, experiments, numerical values, statistical tests, or conclusions were changed.

## Run 1: RQ1 Scientific Takeaways

Finding:
- RQ1 interpreted individual results correctly, but needed clearer scientific takeaways about why the evidence hierarchy matters.

Applied changes:
- Added the relative F1 improvement from static-only verification to dynamic evidence.
- Reframed static evidence as uncertainty reduction rather than a competitor to execution.
- Strengthened the token, AST, operator, and identifier interpretation so each feature is tied to the type of semantic failure it can or cannot expose.
- Added explicit takeaways for Table V-style group results and the evidence hierarchy.

## Run 2: Complementarity and Cost

Finding:
- RQ2 and RQ3 needed to better distinguish feature accumulation from meaningful evidence integration.

Applied changes:
- Reframed complementarity as the value of evidence sources that can disagree for meaningful semantic reasons.
- Added a stronger interpretation of ablation and coefficient results as support for behavior-proximal evidence.
- Expanded cost-benefit interpretation from raw performance to cost-normalized evidence design.
- Clarified that execution-budget results support staged verification under infrastructure constraints.

## Run 3: Explanations, Generalization, Metrics, and Disagreement

Finding:
- Later sections needed stronger endings that answer "what should researchers learn?"

Applied changes:
- Strengthened RQ4 as evidence-based diagnosis rather than binary detection.
- Strengthened RQ5 as generalization under generator-specific error distributions.
- Reframed conventional metric comparison as evidence-source analysis rather than a leaderboard.
- Added explicit takeaways for language sensitivity, surprising findings, qualitative disagreement cases, and statistical validation.

## Master Outcome

The Results and Analysis section now more consistently follows:

1. Scientific question.
2. Quantitative observation.
3. Explanation of why the result occurs.
4. Scientific implication.
5. Practical implication for future verification systems.

The central message reinforced across subsections is that semantic verification is an evidence integration problem: different evidence sources reduce different uncertainties, incur different costs, and fail in diagnostically useful ways.
