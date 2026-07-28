"""Analyses for semantic information accumulation in EviCode."""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evicode.execution.runner import run_candidate  # noqa: E402
from evicode.features import (  # noqa: E402
    api_tokens,
    call_tokens,
    control_vector,
    data_flow_pairs,
    identifiers,
    identifier_roles,
    lexical_counts,
    operator_family_counter,
    program_profile,
    tree_sitter_counts,
)
from evicode.io import read_jsonl  # noqa: E402
from evicode.taxonomy import feature_to_category, feature_to_name  # noqa: E402


# IEEE PDF validation rejects Type 3 fonts. Embed TrueType outlines in every
# generated vector figure so text remains searchable and publication-safe.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


DIAGNOSTIC_FAMILIES = {
    "Operator misuse": {"operator"},
    "API misuse": {"api"},
    "Identifier-role mismatch": {"identifier"},
    "Control-flow mismatch": {"control"},
    "Execution disagreement": {"execution"},
    "Structurally similar, semantically incorrect": {"structure", "execution"},
}

LADDER_UNCERTAINTY = {
    "Lexical evidence": "symbol correspondence",
    "Syntax": "grammatical possibility",
    "Program structure": "organizational form",
    "Control flow": "possible execution paths",
    "Operator semantics": "local computation",
    "Identifier roles": "value responsibility",
    "Data flow": "value propagation",
    "Execution": "tested behavioral outcome",
}


def split_indices(frame: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["experiment"]["test_size"],
        random_state=config["project"]["seed"],
    )
    return next(splitter.split(frame, frame["label"], frame["problem_id"]))


def shap_comparison(frame: pd.DataFrame, features: list[str], config: dict) -> tuple[pd.DataFrame, dict]:
    train_idx, test_idx = split_indices(frame, config)
    scaler = StandardScaler().fit(frame.loc[train_idx, features].fillna(0.0))
    x_train = scaler.transform(frame.loc[train_idx, features].fillna(0.0))
    x_test = scaler.transform(frame.loc[test_idx, features].fillna(0.0))
    model = LogisticRegression(max_iter=1000, random_state=config["project"]["seed"])
    model.fit(x_train, frame.loc[train_idx, "label"])

    # For a linear log-odds model with the training mean as background, SHAP is
    # exactly coefficient * (standardized value - background mean).
    shap_values = x_test * model.coef_[0]
    mi = mutual_info_classif(
        frame.loc[train_idx, features].fillna(0.0),
        frame.loc[train_idx, "label"],
        random_state=config["project"]["seed"],
    )
    names = feature_to_name()
    categories = feature_to_category()
    result = pd.DataFrame(
        {
            "feature": features,
            "evidence_source": [names.get(f, f) for f in features],
            "category": [categories.get(f, "Unknown") for f in features],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "abs_coefficient": np.abs(model.coef_[0]),
            "mutual_information": mi,
        }
    )
    for column in ["mean_abs_shap", "abs_coefficient", "mutual_information"]:
        result[f"{column}_rank"] = result[column].rank(ascending=False, method="min").astype(int)
    correlations = {
        "shap_coefficient_spearman": float(spearmanr(result["mean_abs_shap"], result["abs_coefficient"]).statistic),
        "shap_mi_spearman": float(spearmanr(result["mean_abs_shap"], result["mutual_information"]).statistic),
        "coefficient_mi_spearman": float(spearmanr(result["abs_coefficient"], result["mutual_information"]).statistic),
    }
    return result.sort_values("mean_abs_shap", ascending=False), correlations


def diagnostic_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    failed = frame["label"].eq(0)
    return {
        "Operator misuse": failed & (frame["ln_operator_family_similarity"] < 0.55),
        "API misuse": failed & (frame["api_mismatch_score"] > 0.8),
        "Identifier-role mismatch": failed & (frame["ln_identifier_role_distribution_similarity"] < 0.7),
        "Control-flow mismatch": failed & (frame["ln_control_profile_similarity"] < 0.65),
        "Execution disagreement": failed & frame["execution_passed_example"].astype(bool) & ~frame["execution_passed_full"].astype(bool),
        "Structurally similar, semantically incorrect": failed & (frame["ast_similarity"] > 0.75) & ~frame["execution_passed_full"].astype(bool),
    }


def explainability_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    masks = diagnostic_masks(frame)
    all_families = {family for values in DIAGNOSTIC_FAMILIES.values() for family in values}
    rows = [{"removed_family": "None", "categories_retained": len(masks), "categories_lost": 0,
             "diagnostic_triggers_retained": int(sum(mask.sum() for mask in masks.values()))}]
    for removed in sorted(all_families):
        available = [name for name, required in DIAGNOSTIC_FAMILIES.items() if removed not in required]
        rows.append(
            {
                "removed_family": removed.title(),
                "categories_retained": len(available),
                "categories_lost": len(masks) - len(available),
                "diagnostic_triggers_retained": int(sum(masks[name].sum() for name in available)),
            }
        )
    return pd.DataFrame(rows)


def qualitative_cases(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used: set[str] = set()
    for category, mask in diagnostic_masks(frame).items():
        candidates = frame.loc[mask & ~frame["example_id"].isin(used)].copy()
        candidates = candidates.sort_values(["problem_id", "example_id"]).drop_duplicates("problem_id").head(2)
        for _, row in candidates.iterrows():
            used.add(row["example_id"])
            rows.append(
                {
                    "category": category,
                    "example_id": row["example_id"],
                    "language_pair": f"{row['source_language']}->{row['target_language']}",
                    "negative_type": row["negative_type"],
                    "ast": row["ast_similarity"],
                    "control": row["ln_control_profile_similarity"],
                    "operator": row["ln_operator_family_similarity"],
                    "identifier_role": row["ln_identifier_role_distribution_similarity"],
                    "api_mismatch": row["api_mismatch_score"],
                    "example_pass": bool(row["execution_passed_example"]),
                    "full_pass": bool(row["execution_passed_full"]),
                }
            )
    return pd.DataFrame(rows)


def measure(callable_, repeats: int = 3) -> tuple[float, float]:
    times, peaks = [], []
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        callable_()
        times.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / (1024 * 1024))
    return float(np.mean(times)), float(np.mean(peaks))


def benchmark_extractors(examples: pd.DataFrame, seed: int) -> pd.DataFrame:
    sample = pd.concat(
        [group.sample(min(12, len(group)), random_state=seed) for _, group in examples.groupby(["target_language", "label"])],
        ignore_index=True,
    )
    extractors = {
        "Raw lexical": lambda r: (lexical_counts(r.source_code), lexical_counts(r.target_code)),
        "Syntax": lambda r: (tree_sitter_counts(r.source_code, r.source_language), tree_sitter_counts(r.target_code, r.target_language)),
        "Program structure": lambda r: (program_profile(r.source_code, r.source_language), program_profile(r.target_code, r.target_language)),
        "Control-flow": lambda r: (control_vector(r.source_code), control_vector(r.target_code)),
        "Operator semantics": lambda r: (operator_family_counter(r.source_code), operator_family_counter(r.target_code)),
        "API/calls": lambda r: (api_tokens(r.source_code) | call_tokens(r.source_code), api_tokens(r.target_code) | call_tokens(r.target_code)),
        "Identifier roles": lambda r: (identifiers(r.source_code), identifier_roles(r.source_code), identifiers(r.target_code), identifier_roles(r.target_code)),
        "Data-flow": lambda r: (data_flow_pairs(r.source_code), data_flow_pairs(r.target_code)),
    }
    rows = []
    for name, extractor in extractors.items():
        measurements = [measure(lambda row=row: extractor(row), repeats=2) for row in sample.itertuples()]
        elapsed = np.array([x[0] for x in measurements]) * 1000
        rows.append({"extractor": name, "samples": len(sample), "avg_time_ms": elapsed.mean(),
                     "std_time_ms": elapsed.std(ddof=1), "min_time_ms": elapsed.min(), "max_time_ms": elapsed.max(),
                     "peak_python_memory_mb": np.mean([x[1] for x in measurements]), "measurement": "in-process"})

    execution_sample = sample.groupby(["target_language", "label"], group_keys=False).head(8)
    for budget, test_column in [("Example execution", "target_example_test"), ("Full execution", "target_test")]:
        timings = []
        unavailable = 0
        for row in execution_sample.itertuples():
            start = time.perf_counter()
            try:
                run_candidate(row.target_language, row.target_code, getattr(row, test_column), 5)
                timings.append(time.perf_counter() - start)
            except FileNotFoundError:
                unavailable += 1
        elapsed = np.array(timings) * 1000
        rows.append({"extractor": budget, "samples": len(timings), "avg_time_ms": elapsed.mean() if timings else np.nan,
                     "std_time_ms": elapsed.std(ddof=1) if len(timings) > 1 else np.nan,
                     "min_time_ms": elapsed.min() if timings else np.nan, "max_time_ms": elapsed.max() if timings else np.nan,
                     "peak_python_memory_mb": np.nan,
                     "measurement": f"wall-clock subprocess; child memory unavailable; {unavailable} runtime-unavailable pairs excluded"})
    return pd.DataFrame(rows)


def empirical_ladder(frame: pd.DataFrame, shap: pd.DataFrame, costs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate measured information, model use, F1 gain, and cost by semantic level."""
    groups = {
        "Lexical evidence": ["token_jaccard", "edit_similarity", "length_ratio", "retrieval_similarity"],
        "Syntax": ["syntax_proxy", "source_syntax_valid", "target_syntax_valid", "ln_syntax_both_valid"],
        "Program structure": [f for f in frame if f.startswith("ast_") or "structure" in feature_to_category().get(f, "").lower()],
        "Control flow": [f for f in frame if feature_to_category().get(f) == "Normalized-control"],
        "Operator semantics": [f for f in frame if feature_to_category().get(f) == "Normalized-operator"],
        "Identifier roles": [f for f in frame if feature_to_category().get(f) == "Normalized-identifier"] + ["identifier_role_similarity"],
        "Data flow": [f for f in frame if feature_to_category().get(f) == "Normalized-dataflow"] + ["data_flow_similarity"],
        "Execution": ["execution_passed_full"],
    }
    groups = {name: list(dict.fromkeys(f for f in values if f in frame)) for name, values in groups.items()}
    all_features = [feature for values in groups.values() for feature in values]
    mi_values = mutual_info_classif(frame[all_features].fillna(0.0), frame["label"], random_state=config["project"]["seed"])
    mi_map = dict(zip(all_features, mi_values, strict=True))
    shap_means = shap.groupby(shap["feature"].map({f: level for level, values in groups.items() for f in values}))["mean_abs_shap"].mean()
    max_shap = shap_means.max()
    cost_map = costs.set_index("extractor")["avg_time_ms"].to_dict()
    cost_names = {"Lexical evidence": "Raw lexical", "Syntax": "Syntax", "Program structure": "Program structure",
                  "Control flow": "Control-flow", "Operator semantics": "Operator semantics", "Identifier roles": "Identifier roles",
                  "Data flow": "Data-flow", "Execution": "Full execution"}
    train_idx, test_idx = split_indices(frame, config)
    cumulative: list[str] = []
    previous_f1 = 0.0
    rows = []
    for level, level_features in groups.items():
        cumulative.extend(level_features)
        scaler = StandardScaler().fit(frame.loc[train_idx, cumulative].fillna(0.0))
        model = LogisticRegression(max_iter=1000, random_state=config["project"]["seed"])
        model.fit(scaler.transform(frame.loc[train_idx, cumulative].fillna(0.0)), frame.loc[train_idx, "label"])
        probability = model.predict_proba(scaler.transform(frame.loc[test_idx, cumulative].fillna(0.0)))[:, 1]
        current_f1 = f1_score(frame.loc[test_idx, "label"], probability >= 0.5, zero_division=0)
        rows.append({"level": level, "mean_mutual_information": np.mean([mi_map[f] for f in level_features]),
                     "relative_shap": shap_means.get(level, np.nan) / max_shap if level != "Execution" else np.nan,
                     "incremental_f1": current_f1 - previous_f1, "cumulative_f1": current_f1,
                     "avg_time_ms": cost_map.get(cost_names[level], np.nan), "uncertainty_removed": LADDER_UNCERTAINTY[level]})
        previous_f1 = current_f1
    return pd.DataFrame(rows)


def plots(shap: pd.DataFrame, predictions: pd.DataFrame, costs: pd.DataFrame, ladder: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    top = shap.head(12).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(top["evidence_source"], top["mean_abs_shap"], color="#2878B5")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output / "shap_feature_importance.pdf")
    plt.close(fig)

    pred = predictions[predictions["system"] == "all_evidence"].copy()
    pred["bin"] = pd.cut(pred["probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = pred.groupby("bin", observed=False).agg(mean_confidence=("probability", "mean"), accuracy=("label", "mean"), count=("label", "size")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].hist(pred["probability"], bins=np.linspace(0, 1, 21), color="#2878B5", edgecolor="white")
    axes[0].set(xlabel="Predicted confidence", ylabel="Held-out samples", title="Confidence distribution")
    valid = calibration.dropna()
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].plot(valid["mean_confidence"], valid["accuracy"], "o-", color="#D95319")
    axes[1].set(xlabel="Mean confidence", ylabel="Empirical accuracy", title="Reliability")
    fig.tight_layout()
    fig.savefig(output / "confidence_histogram_calibration.pdf")
    plt.close(fig)
    calibration.to_csv(ROOT / "results" / "semantic_accumulation" / "confidence_bins.csv", index=False)

    ordered = costs.copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(ordered["extractor"], ordered["avg_time_ms"], color="#54A24B")
    ax.set_xscale("log")
    ax.set_xlabel("Average extraction time per pair (ms, log scale)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output / "extractor_costs.pdf")
    plt.close(fig)

    questions = [
        ("Source +\ncandidate", "What can be\nobserved?"),
        ("Individual\nobservations", "What does each\none reveal?"),
        ("Combined\nobservations", "Which observations\ncomplement?"),
        ("Acquisition\nconstraints", "What is each\nobservation worth?"),
        ("Decision\nreliability", "Are explanation and\nconfidence useful?"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.set_xlim(-0.55, len(questions) - 0.45)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, (stage, question) in enumerate(questions):
        ax.text(index, 0.64, stage, ha="center", va="center", fontsize=8.5, weight="bold",
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E8F1F8", "edgecolor": "#2878B5", "linewidth": 1.4})
        ax.text(index, 0.27, question, ha="center", va="center", fontsize=7.8, color="#444444")
        if index < len(questions) - 1:
            ax.annotate("", xy=(index + 0.72, 0.64), xytext=(index + 0.28, 0.64),
                        arrowprops={"arrowstyle": "->", "color": "#7A8793", "linewidth": 1.5})
    ax.set_title("Scientific Questions Behind Reference-Free Verification", fontsize=11.5, weight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(output / "verification_study_questions.pdf")
    plt.close(fig)

    display = ladder.iloc[::-1].copy()
    display["MI"] = display["mean_mutual_information"].map(lambda value: f"{value:.3f}")
    display["Rel. SHAP"] = display["relative_shap"].map(lambda value: "--" if pd.isna(value) else f"{value:.2f}")
    display["Delta F1"] = display["incremental_f1"].map(lambda value: f"{value:+.3f}")
    display["Cost"] = display["avg_time_ms"].map(lambda value: f"{value:.3f} ms")
    fig, ax = plt.subplots(figsize=(7.25, 4.75))
    ax.axis("off")
    table = ax.table(cellText=display[["level", "MI", "Rel. SHAP", "Delta F1", "Cost", "uncertainty_removed"]].values,
                     colLabels=["Evidence level", "Mean MI", "Rel. SHAP", "Incremental F1", "Mean cost", "Main uncertainty removed"],
                     cellLoc="left", colLoc="left", loc="center", colWidths=[0.17, 0.09, 0.10, 0.11, 0.11, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.45)
    for column in range(6):
        table[(0, column)].set_facecolor("#DCEAF7")
        table[(0, column)].set_text_props(weight="bold")
    ax.annotate("", xy=(0.025, 0.88), xytext=(0.025, 0.12), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#2878B5"})
    ax.text(0.005, 0.5, "semantic observability increases\ncomputational cost increases",
            rotation=90, transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#2878B5")
    ax.annotate("", xy=(0.975, 0.88), xytext=(0.975, 0.12), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#D95319"})
    ax.text(0.995, 0.5, "evidence-supported confidence increases\nsemantic uncertainty decreases",
            rotation=90, transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#D95319")
    ax.set_title("Empirical Semantic Information Ladder", fontsize=12, weight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(output / "empirical_semantic_information_ladder.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    panels = [
        ("Traditional reference-based verification", ["Source program", "Reference translation", "Generated candidate", "Similarity metric", "Decision"]),
        ("EviCode reference-free verification", ["Source program + candidate", "Language-normalized evidence", "Evidence aggregation", "Verification confidence"]),
    ]
    for ax, (title, items) in zip(axes, panels, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=9.5, weight="bold")
        ys = np.linspace(0.84, 0.16, len(items))
        for index, (label, y) in enumerate(zip(items, ys, strict=True)):
            ax.text(0.5, y, label, ha="center", va="center", fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#E8F1F8" if ax is axes[1] else "#F2F2F2", "edgecolor": "#2878B5"})
            if index < len(items) - 1:
                ax.annotate("", xy=(0.5, ys[index + 1] + 0.055), xytext=(0.5, y - 0.055), arrowprops={"arrowstyle": "->", "color": "#555555"})
    fig.tight_layout()
    fig.savefig(output / "reference_based_vs_evicode.pdf")
    plt.close(fig)

    stages = [
        ("Candidate", "</>"), ("Lexical", "Aa"), ("Syntax", "{}"), ("Structure", "AST"),
        ("Control", "CFG"), ("Operators", "+/-"), ("Roles", "id"), ("Data flow", "x->y"),
        ("Execution", "run"), ("Decision", "OK?"),
    ]
    x = np.arange(len(stages))
    uncertainty = np.linspace(0.92, 0.12, len(stages))
    cost = np.array([0.05, 0.08, 0.13, 0.22, 0.28, 0.34, 0.40, 0.48, 0.88, 0.90])
    confidence = np.array([0.05, 0.10, 0.16, 0.24, 0.30, 0.38, 0.45, 0.53, 0.86, 0.90])
    fig, ax = plt.subplots(figsize=(7.25, 3.35))
    ax.set_xlim(-0.55, len(stages) - 0.45)
    ax.set_ylim(0, 1.12)
    ax.axis("off")
    ax.plot(x, np.full_like(x, 0.68, dtype=float), color="#7A8793", linewidth=2, zorder=1)
    for index in range(len(stages) - 1):
        ax.annotate("", xy=(index + 0.82, 0.68), xytext=(index + 0.18, 0.68),
                    arrowprops={"arrowstyle": "->", "color": "#7A8793", "linewidth": 1.4})
    for index, (label, icon) in enumerate(stages):
        color = "#D95319" if label == "Execution" else ("#54A24B" if label == "Decision" else "#2878B5")
        ax.scatter(index, 0.68, s=560, color="white", edgecolor=color, linewidth=2, zorder=3)
        ax.text(index, 0.68, icon, ha="center", va="center", fontsize=7.5, weight="bold", color=color, zorder=4)
        ax.text(index, 0.51, label, ha="center", va="top", fontsize=7.2, weight="bold")
    ax.plot(x, uncertainty, color="#D95319", linewidth=1.8, marker="o", markersize=3, label="Remaining uncertainty")
    ax.plot(x, cost, color="#6F4E9C", linewidth=1.8, marker="s", markersize=3, label="Acquisition cost")
    ax.plot(x, confidence, color="#54A24B", linewidth=1.8, marker="^", markersize=3, label="Evidence-supported confidence")
    ax.text(0, 1.01, "High uncertainty", fontsize=7.2, color="#D95319", ha="left")
    ax.text(9, 1.01, "Higher confidence and cost", fontsize=7.2, color="#355F3D", ha="right")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=7.2)
    ax.set_title("Evidence Acquisition Timeline for Semantic Verification", fontsize=11.5, weight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(output / "evidence_acquisition_timeline.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/humanevalx.yaml")
    parser.add_argument("--examples", default="datasets/processed/humanevalx/verification_examples.jsonl")
    parser.add_argument("--evidence", default="experiments/humanevalx/evidence_rich/evidence.jsonl")
    parser.add_argument("--predictions", default="experiments/humanevalx/fusion_rich/predictions.csv")
    parser.add_argument("--output-dir", default="results/semantic_accumulation")
    parser.add_argument("--resume", action="store_true", help="Reuse complete existing analysis outputs.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing analysis outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args()

    def rooted(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path

    config_path = rooted(args.config)
    examples_path = rooted(args.examples)
    evidence_path = rooted(args.evidence)
    predictions_path = rooted(args.predictions)
    output = rooted(args.output_dir)
    inputs = [config_path, examples_path, evidence_path, predictions_path]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing analysis inputs: {missing}")
    expected = [
        output / "shap_comparison.csv",
        output / "importance_correlations.csv",
        output / "explainability_ablation.csv",
        output / "qualitative_cases.csv",
        output / "extractor_costs.csv",
        output / "empirical_semantic_information_ladder.csv",
    ]
    if args.dry_run:
        print({"status": "valid", "inputs": [str(path) for path in inputs], "output_dir": str(output)})
        return 0
    if args.resume and not args.force and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print({"status": "reused", "output_dir": str(output), "outputs": len(expected)})
        return 0

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    examples = pd.DataFrame(read_jsonl(examples_path))
    evidence = pd.DataFrame(read_jsonl(evidence_path))
    frame = evidence.merge(examples[["example_id", "source_code", "target_code"]], on="example_id")
    categories = feature_to_category()
    features = [feature for feature in categories if feature in frame and categories[feature] != "Dynamic"]
    output.mkdir(parents=True, exist_ok=True)
    shap, correlations = shap_comparison(frame, features, config)
    shap.to_csv(output / "shap_comparison.csv", index=False)
    pd.DataFrame([correlations]).to_csv(output / "importance_correlations.csv", index=False)
    explainability_ablation(frame).to_csv(output / "explainability_ablation.csv", index=False)
    qualitative_cases(frame).to_csv(output / "qualitative_cases.csv", index=False)
    costs = benchmark_extractors(examples, config["project"]["seed"])
    costs.to_csv(output / "extractor_costs.csv", index=False)
    ladder = empirical_ladder(frame, shap, costs, config)
    ladder.to_csv(output / "empirical_semantic_information_ladder.csv", index=False)
    tables = ROOT / "tables"
    shap_table = shap.head(10)[["evidence_source", "mean_abs_shap", "abs_coefficient", "mutual_information"]].copy()
    shap_table.columns = ["Evidence", "SHAP", "|Coef.|", "MI"]
    (tables / "shap_comparison.tex").write_text(
        shap_table.to_latex(index=False, float_format="%.3f", escape=True, caption="Top static evidence sources by mean absolute SHAP value on the held-out split. SHAP and coefficients measure conditional model use; MI measures marginal label information.", label="tab:shap-comparison"), encoding="utf-8")
    cost_table = costs[["extractor", "samples", "avg_time_ms", "std_time_ms", "min_time_ms", "max_time_ms", "peak_python_memory_mb"]].copy()
    cost_table.columns = ["Extractor", "n", "Mean", "SD", "Min", "Max", "Peak MB"]
    cost_tex = cost_table.to_latex(index=False, float_format="%.3f", na_rep="--", escape=True,
        caption="Measured evidence-extraction time per source--candidate pair in milliseconds (mean, SD, minimum, and maximum), with peak Python allocation. Static measurements use 72 stratified pairs; execution uses 32 runnable Python/JavaScript pairs. Java and child-process memory are unavailable.", label="tab:extractor-costs-measured")
    cost_tex = cost_tex.replace("\\begin{table}", "\\begin{table*}[t]\n\\centering\n\\footnotesize").replace("\\end{table}", "\\end{table*}")
    (tables / "extractor_costs_measured.tex").write_text(cost_tex, encoding="utf-8")
    ablation = explainability_ablation(frame)
    ablation.columns = ["Removed", "Ret.", "Lost", "Triggers"]
    (tables / "explainability_ablation.tex").write_text(
        ablation.to_latex(index=False, escape=True, caption="Diagnostic-retention ablation. Counts refer to rule-defined evidence triggers, not human-adjudicated root-cause accuracy.", label="tab:explainability-ablation"), encoding="utf-8")
    cases = qualitative_cases(frame)[["category", "language_pair", "negative_type", "example_pass", "full_pass"]].copy()
    cases["category"] = cases["category"].replace({
        "Operator misuse": "Operator",
        "API misuse": "API",
        "Identifier-role mismatch": "Identifier role",
        "Control-flow mismatch": "Control flow",
        "Execution disagreement": "Ex./full disagree",
        "Structurally similar, semantically incorrect": "Structural + wrong",
    })
    cases.insert(1, "Case", [f"C{index:02d}" for index in range(1, len(cases) + 1)])
    cases["example_pass"] = cases["example_pass"].map({True: "yes", False: "no"})
    cases["full_pass"] = cases["full_pass"].map({True: "yes", False: "no"})
    cases.columns = ["Diagnostic group", "Case", "Pair", "Negative", "Ex.", "Full"]
    cases_tex = cases.to_latex(index=False, escape=True, caption="Twelve representative evidence disagreements, two per diagnostic group. Stable case identifiers map to full example IDs in the artifact. Groups are not independently adjudicated causal labels.", label="tab:qualitative-cases-12")
    cases_tex = cases_tex.replace("\\begin{table}", "\\begin{table*}[t]\n\\centering\n\\footnotesize").replace("\\end{table}", "\\end{table*}")
    (tables / "qualitative_cases_12.tex").write_text(cases_tex, encoding="utf-8")
    plots(shap, pd.read_csv(predictions_path), costs, ladder, ROOT / "figures")
    print({"shap_features": len(shap), "qualitative_cases": len(qualitative_cases(frame)), "cost_rows": len(costs), **correlations})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
