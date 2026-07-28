"""Supplementary visual analyses using the completed authentic-only dataset."""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evicode.taxonomy import OBSERVATION_FAMILIES  # noqa: E402

OUT = ROOT / "outputs" / "authentic_only"
FIG = OUT / "figures"
TAB = OUT / "statistics"
SEED = 42
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

df = pd.read_parquet(OUT / "datasets" / "analysis_dataset.parquet")
features = joblib.load(OUT / "models" / "authentic_all_languages.joblib")["features"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, test_idx = next(splitter.split(df, df.label, df.problem_id))
train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
assert set(train.problem_id).isdisjoint(test.problem_id)


def fit(frame: pd.DataFrame) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    model.fit(frame[features], frame.label)
    return model


def transfer_matrix(column: str, order: list[str], filename: str) -> pd.DataFrame:
    matrix = pd.DataFrame(index=order, columns=order, dtype=float)
    for source in order:
        source_train = train[train[column] == source]
        model = fit(source_train)
        for target in order:
            target_test = test[test[column] == target]
            scores = model.predict_proba(target_test[features])[:, 1]
            matrix.loc[source, target] = roc_auc_score(target_test.label, scores)
    matrix.to_csv(TAB / f"{filename}.csv")
    width = 6.6 if len(order) > 3 else 4.8
    fig, ax = plt.subplots(figsize=(width, width * 0.78))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.5, vmax=0.9,
                linewidths=0.6, cbar_kws={"label": "ROC-AUC"}, ax=ax)
    ax.set(xlabel="Test domain (problem-disjoint)", ylabel="Training domain")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return matrix


languages = ["C", "C++", "C#", "Python", "Ruby", "Kotlin", "Swift"]
generators = ["DeepSeek-Coder", "QwenCoder", "StarCoder"]
transfer_matrix("language", languages, "language_transfer_matrix")
transfer_matrix("generator", generators, "generator_transfer_matrix")

# Per-generator reliability and class-conditional score distributions.
pred = pd.read_csv(OUT / "metrics" / "predictions.csv")
fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), sharex="row")
for col, generator in enumerate(generators):
    group = pred[pred.generator == generator]
    observed, confidence = calibration_curve(group.label, group.confidence,
                                              n_bins=8, strategy="quantile")
    ax = axes[0, col]
    ax.plot(confidence, observed, "o-", color="#2878B5", lw=1.6)
    ax.plot([0, 1], [0, 1], "--", color="#666666", lw=1)
    ax.set_title(generator)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if col == 0: ax.set_ylabel("Observed correctness")
    ax.set_xlabel("Predicted probability")

    ax = axes[1, col]
    for label, name, color in [(0, "Incorrect", "#2878B5"), (1, "Correct", "#D97941")]:
        sns.kdeplot(group.loc[group.label == label, "confidence"], label=name,
                    color=color, fill=False, common_norm=False, ax=ax)
    ax.set_xlim(0, 1)
    if col == 0: ax.set_ylabel("Density within class")
    ax.set_xlabel("Predicted probability")
    if col == 2: ax.legend(frameon=False)
    elif ax.get_legend() is not None: ax.get_legend().remove()
sns.despine(fig=fig)
fig.tight_layout()
fig.savefig(FIG / "generator_reliability_and_scores.pdf", bbox_inches="tight")
plt.close(fig)

# Family-level correlations are more interpretable than a 45x45 wall of labels.
families = OBSERVATION_FAMILIES
family_frame = pd.DataFrame({name: test[cols].mean(axis=1) for name, cols in families.items()})
family_frame["label"] = test.label.to_numpy()
family_frame["quality_score"] = test.quality_score.to_numpy()
corr = family_frame.drop(columns=["label", "quality_score"]).corr()
corr.to_csv(TAB / "family_correlation.csv")
fig, ax = plt.subplots(figsize=(8.0, 6.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.tick_params(axis="x", rotation=40)
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(FIG / "family_correlation.pdf", bbox_inches="tight")
plt.close(fig)

# Grade distributions: sample equally by grade to avoid hiding minority grades.
sample = pd.concat([
    group.sample(min(len(group), 1500), random_state=SEED)
    for _, group in family_frame.groupby("quality_score")
], ignore_index=True)
long = sample.melt(id_vars="quality_score", value_vars=list(families),
                   var_name="Family", value_name="Agreement")
fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.4), sharex=True, sharey=True)
for ax, family in zip(axes.flat, families):
    sns.violinplot(data=long[long.Family == family], x="quality_score", y="Agreement",
                   inner="quart", cut=0, color="#79B6C7", linewidth=0.8, ax=ax)
    ax.set_title(family); ax.set_xlabel(None); ax.set_ylabel(None)
for ax in axes[-1, :]: ax.set_xlabel("Execution grade")
axes[1, 0].set_ylabel("Source-candidate agreement")
fig.tight_layout()
fig.savefig(FIG / "family_grade_distributions.pdf", bbox_inches="tight")
plt.close(fig)

# PCA is diagnostic only: overlap shows that the evidence space is not linearly separable.
sample_test = test.sample(min(6000, len(test)), random_state=SEED)
scaled = StandardScaler().fit_transform(sample_test[features])
coords = PCA(n_components=2, random_state=SEED).fit_transform(scaled)
pca = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                    "Outcome": sample_test.label.map({0: "Incorrect", 1: "Correct"}).to_numpy(),
                    "Generator": sample_test.generator.to_numpy()})
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sns.scatterplot(data=pca, x="PC1", y="PC2", hue="Outcome", style="Generator",
                alpha=0.32, s=18, palette=["#2878B5", "#D97941"], ax=ax)
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.set_title("Projection of the 45-observation verification space")
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG / "evidence_pca.pdf", bbox_inches="tight")
plt.close(fig)

# Central conceptual chain: observations become evidence only through relevance to Y.
fig, ax = plt.subplots(figsize=(11.6, 2.8))
ax.set_xlim(0, 6); ax.set_ylim(0, 1.5); ax.axis("off")
stages = [
    ("Program\nobservation", "A measurable property"),
    ("Semantic\nevidence", "Observation relevant to correctness"),
    ("Updated\nbelief", "What can now be inferred"),
    ("Residual\nuncertainty", "What remains unresolved"),
    ("Decision", "Accept, inspect, test, or abstain"),
]
colors = ["#E4EFF7", "#C5E0EC", "#91C5D3", "#F1D3A8", "#D97941"]
for i, ((title, subtitle), color) in enumerate(zip(stages, colors)):
    x = i * 1.2 + 0.08
    box = FancyBboxPatch((x, 0.25), 0.95, 0.9, boxstyle="round,pad=.03",
                         facecolor=color, edgecolor="#244B5A", linewidth=1.0)
    ax.add_patch(box)
    ax.text(x + .475, .82, title, ha="center", va="center", weight="bold", fontsize=9)
    ax.text(x + .475, .43, subtitle, ha="center", va="center", fontsize=6.7, wrap=True)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + 1.18, .70), xytext=(x + .96, .70),
                    arrowprops=dict(arrowstyle="-|>", color="#244B5A", lw=1.2))
ax.text(3.0, 1.34, "Evidence accumulation changes the justified decision",
        ha="center", fontsize=9.5, color="#244B5A")
fig.tight_layout(pad=.2)
fig.savefig(FIG / "evidence_reasoning_chain.pdf", bbox_inches="tight")
plt.close(fig)

print("Supplementary visualizations generated with problem-disjoint transfer cells.")
