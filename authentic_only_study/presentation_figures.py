"""Regenerate reader-facing figures from completed authentic-only outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "authentic_only"
FIG = OUT / "figures"
BLUE = "#2878B5"
ORANGE = "#D97941"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

metrics = pd.read_csv(OUT / "metrics" / "all_metrics.csv")
pred = pd.read_csv(OUT / "metrics" / "predictions.csv")
importance = pd.read_csv(OUT / "feature_importance" / "importance.csv")
progression = pd.read_csv(OUT / "feature_importance" / "quality_progression.csv")


def domain_auc(scope: str, filename: str) -> None:
    frame = metrics[metrics.scope == scope].sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    sns.barplot(data=frame, x="roc_auc", y="value", color=BLUE, ax=ax)
    ax.axvline(0.5, ls="--", lw=1.2, color="#666666")
    ax.set(xlim=(0.45, 0.9), xlabel="ROC-AUC", ylabel=None)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / filename, bbox_inches="tight")
    plt.close(fig)


domain_auc("Language", "language_wise_auc.pdf")
domain_auc("Leave-one-language-out", "logo_language_auc.pdf")
domain_auc("Generator", "generator_wise_auc.pdf")
domain_auc("Leave-one-generator-out", "logo_generator_auc.pdf")

# Reliability: quantile bins keep each point supported by a comparable number of examples.
observed, confidence = calibration_curve(
    pred.label, pred.confidence, n_bins=10, strategy="quantile"
)
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.plot(confidence, observed, "o-", color=BLUE, lw=1.8, label="EviCode")
ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1.1, label="Perfect calibration")
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted correctness probability",
       ylabel="Observed correctness rate")
ax.legend(frameon=False, loc="upper left")
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG / "calibration.pdf", bbox_inches="tight")
plt.close(fig)

plot_pred = pred.assign(
    outcome=pred.label.map({0: "Incorrect translation", 1: "Correct translation"})
)
fig, ax = plt.subplots(figsize=(4.8, 3.6))
sns.histplot(
    data=plot_pred, x="confidence", hue="outcome", bins=20,
    stat="density", common_norm=False, element="step", alpha=0.35,
    palette=[BLUE, ORANGE], ax=ax,
)
ax.set(xlabel="Predicted correctness probability", ylabel="Density within class")
ax.get_legend().set_title(None)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG / "confidence_histogram.pdf", bbox_inches="tight")
plt.close(fig)

friendly = {
    "token_jaccard": "Token-set overlap",
    "branch_count_similarity": "Branch-count similarity",
    "identifier_role_similarity": "Identifier-role similarity",
    "ln_cyclomatic_complexity_similarity": "Cyclomatic-complexity similarity",
    "ln_identifier_role_count_similarity": "Identifier-role count similarity",
    "ln_call_count_similarity": "Call-count similarity",
    "control_flow_similarity": "Control-flow similarity",
    "ln_control_profile_similarity": "Normalized control profile",
    "ln_branching_factor_similarity": "Branching-factor similarity",
    "api_similarity": "API similarity",
    "ln_return_count_similarity": "Return-count similarity",
    "api_mismatch_score": "API mismatch",
    "ln_loop_count_similarity": "Loop-count similarity",
    "data_flow_similarity": "Data-flow similarity",
    "ln_cfg_edges_similarity": "Control-flow edge similarity",
    "type_similarity": "Type similarity",
    "edit_similarity": "Edit similarity",
    "ln_syntax_both_valid": "Joint syntax validity",
}
family_override = {}
top = importance.nlargest(10, "mean_abs_linear_shap").copy()
top["label"] = top.feature.map(lambda name: friendly.get(name, name.replace("_", " ").title()))
top["display_family"] = [family_override.get(f, fam) for f, fam in zip(top.feature, top.family)]
top = top.sort_values("mean_abs_linear_shap", ascending=True)
palette = dict(zip(top.display_family.unique(), sns.color_palette("colorblind", top.display_family.nunique())))
fig, ax = plt.subplots(figsize=(6.2, 3.9))
sns.barplot(
    data=top, y="label", x="mean_abs_linear_shap", hue="display_family",
    dodge=False, palette=palette, ax=ax,
)
ax.set(xlabel="Mean absolute contribution to model log-odds", ylabel=None)
ax.legend(title="Observation family", frameon=False, fontsize=7, title_fontsize=8,
          loc="upper right")
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG / "evidence_importance.pdf", bbox_inches="tight")
plt.close(fig)

order = ["Surface", "Syntax", "Structure", "Control flow", "Operators", "Identifiers",
         "Data flow", "APIs", "Complexity"]
pivot = progression.groupby(["family", "quality_score"])["mean"].mean().unstack().reindex(order)
pivot.columns = ["Score 1\nnon-compiling", "Score 2\ncompiling incorrect", "Score 3\ncorrect"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
    linewidths=0.5, cbar_kws={"label": "Mean normalized agreement"}, ax=ax,
)
ax.set(xlabel=None, ylabel=None)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(FIG / "quality_progression_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

print("Reader-facing figures regenerated from existing CSV outputs.")
