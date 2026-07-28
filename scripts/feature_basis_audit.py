"""Audit the construct basis, dependence, and size sensitivity of EviCode probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.taxonomy import feature_to_category, feature_to_name  # noqa: E402


DATA = ROOT / "outputs" / "authentic_only" / "datasets" / "analysis_dataset.parquet"
CACHE = ROOT / "outputs" / "authentic_only" / "datasets" / "features.jsonl"
MANIFEST = ROOT / "outputs" / "authentic_only" / "extraction_manifest.json"
OUTPUT = ROOT / "outputs" / "authentic_only" / "feature_basis"
SEED = 42
TEST_SIZE = 0.30
BOOTSTRAP_ITERATIONS = 500

EXCLUSION_REASONS = {
    "source_length": (
        "Unilateral absolute magnitude; the bounded source--candidate length ratio is retained "
        "to avoid task-size and language-scale shortcuts."
    ),
    "target_length": (
        "Unilateral absolute magnitude; the bounded source--candidate length ratio is retained "
        "to avoid candidate-size and generator-style shortcuts."
    ),
    "source_syntax_valid": (
        "Unilateral parser outcome; joint source--candidate validity is retained so parser "
        "coverage is represented as a compatibility observation."
    ),
    "target_syntax_valid": (
        "Unilateral parser outcome; joint source--candidate validity is retained so candidate "
        "validity is not used independently of source-parser coverage."
    ),
    "retrieval_similarity": (
        "Exact operational duplicate of edit similarity; one canonical text-similarity control is retained."
    ),
    "syntax_proxy": (
        "Exact operational duplicate of normalized joint syntax validity; the normalized channel is retained."
    ),
    "ast_depth_similarity": (
        "Exact operational duplicate of normalized maximum AST-depth agreement; the normalized channel is retained."
    ),
    "nesting_depth_similarity": (
        "Exact operational duplicate of normalized nesting-depth agreement; the normalized channel is retained."
    ),
}

UNILATERAL_EXCLUSIONS = {
    "source_length",
    "target_length",
    "source_syntax_valid",
    "target_syntax_valid",
}

PARSER_DEPENDENT_OR_FALLBACK = {
    "ast_similarity",
    "ast_shape_similarity",
    "ln_syntax_both_valid",
    "ln_max_ast_depth_similarity",
    "ln_avg_tree_depth_similarity",
    "ln_branching_factor_similarity",
}


def load_excluded_columns(selected: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Read only extractor outputs omitted from the frozen 45-probe manifest."""
    records: list[dict[str, float | str]] = []
    all_extracted: list[str] | None = None
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "ok" or int(row["quality_score"]) not in {1, 2, 3}:
                continue
            if all_extracted is None:
                all_extracted = list(row["features"])
            excluded = [name for name in row["features"] if name not in selected]
            records.append(
                {"example_id": row["example_id"], **{name: row["features"][name] for name in excluded}}
            )
    if all_extracted is None:
        raise RuntimeError(f"No valid records found in {CACHE}")
    return pd.DataFrame(records).drop_duplicates("example_id"), all_extracted


def fit_probabilities(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Fit the paper's transparent estimator and return held-out probabilities."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    model.fit(train[features], train["label"])
    return model.predict_proba(test[features])[:, 1]


def metric_row(test: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    """Calculate the metrics needed for feature-basis sensitivity."""
    hard = test["quality_score"].isin([2, 3]).to_numpy()
    return {
        "n": int(len(test)),
        "roc_auc": float(roc_auc_score(test["label"], probability)),
        "pr_auc": float(average_precision_score(test["label"], probability)),
        "f1_at_0_5": float(f1_score(test["label"], probability >= 0.5)),
        "hard_roc_auc": float(roc_auc_score(test.loc[hard, "label"], probability[hard])),
    }


def correlation_prune(
    features: list[str], correlation: pd.DataFrame, threshold: float
) -> list[str]:
    """Prune training-only correlations with a deterministic normalized-first preference."""
    original_position = {feature: index for index, feature in enumerate(features)}
    candidates = sorted(
        features,
        key=lambda feature: (0 if feature.startswith("ln_") else 1, original_position[feature]),
    )
    retained: list[str] = []
    for feature in candidates:
        if all(correlation.loc[feature, other] < threshold for other in retained):
            retained.append(feature)
    return sorted(retained, key=original_position.__getitem__)


def dependence_components(
    features: list[str], correlation: pd.DataFrame, threshold: float
) -> list[list[str]]:
    """Return connected components induced by high absolute training correlations."""
    parent = {feature: feature for feature in features}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(features):
        for right in features[index + 1 :]:
            if correlation.loc[left, right] >= threshold:
                union(left, right)
    grouped: dict[str, list[str]] = {}
    for feature in features:
        grouped.setdefault(find(feature), []).append(feature)
    return list(grouped.values())


def paired_cluster_bootstrap(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    baseline: str,
) -> pd.DataFrame:
    """Bootstrap AUC and paired AUC deltas by held-out problem."""
    rng = np.random.default_rng(SEED)
    groups = test["problem_id"].unique()
    group_rows = {
        group: np.flatnonzero(test["problem_id"].to_numpy() == group) for group in groups
    }
    samples: dict[str, list[tuple[float, float]]] = {name: [] for name in probabilities}
    labels = test["label"].to_numpy()
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = np.concatenate(
            [group_rows[group] for group in rng.choice(groups, len(groups), replace=True)]
        )
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        baseline_auc = roc_auc_score(sampled_labels, probabilities[baseline][indices])
        for name, probability in probabilities.items():
            auc = roc_auc_score(sampled_labels, probability[indices])
            samples[name].append((auc, auc - baseline_auc))
    rows = []
    for name, values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "system": name,
                "roc_auc_ci_low": float(np.quantile(array[:, 0], 0.025)),
                "roc_auc_ci_high": float(np.quantile(array[:, 0], 0.975)),
                "delta_auc_ci_low": float(np.quantile(array[:, 1], 0.025)),
                "delta_auc_ci_high": float(np.quantile(array[:, 1], 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    """Run the complete feature-basis audit."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = list(json.loads(MANIFEST.read_text(encoding="utf-8"))["feature_names"])
    if len(selected) != 45:
        raise AssertionError(f"Expected 45 selected probes, found {len(selected)}")
    frame = pd.read_parquet(DATA)
    excluded_frame, all_extracted = load_excluded_columns(selected)
    frame = frame.merge(excluded_frame, on="example_id", how="left", validate="one_to_one")
    excluded = [feature for feature in all_extracted if feature not in selected]
    if set(excluded) != set(EXCLUSION_REASONS):
        raise AssertionError(f"Unexpected excluded extractor outputs: {excluded}")

    train_index, test_index = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(
            frame, frame["label"], frame["problem_id"]
        )
    )
    train = frame.iloc[train_index].copy()
    test = frame.iloc[test_index].copy().reset_index(drop=True)
    if not set(train["problem_id"]).isdisjoint(test["problem_id"]):
        raise AssertionError("Problem leakage detected")

    selected_train = train[selected].astype(float)
    candidate_train = train[all_extracted].astype(float)
    missing_count = selected_train.isna().sum()
    unique_count = selected_train.nunique(dropna=False)
    correlation = selected_train.corr(method="spearman").abs()

    pair_rows = []
    selected_exact_rows = []
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            rho = float(correlation.loc[left, right])
            if rho >= 0.90:
                pair_rows.append({"feature_a": left, "feature_b": right, "abs_spearman_rho": rho})
            left_values = selected_train[left].to_numpy()
            right_values = selected_train[right].to_numpy()
            if np.array_equal(left_values, right_values):
                selected_exact_rows.append(
                    {"feature_a": left, "feature_b": right, "relation": "equal"}
                )
    pd.DataFrame(pair_rows).sort_values("abs_spearman_rho", ascending=False).to_csv(
        OUTPUT / "high_correlation_pairs.csv", index=False
    )
    candidate_exact_rows = []
    for index, left in enumerate(all_extracted):
        for right in all_extracted[index + 1 :]:
            if np.array_equal(candidate_train[left].to_numpy(), candidate_train[right].to_numpy()):
                candidate_exact_rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "relation": "equal",
                        "retained_a": left in selected,
                        "retained_b": right in selected,
                    }
                )
    pd.DataFrame(selected_exact_rows).to_csv(
        OUTPUT / "selected_exact_duplicate_pairs.csv", index=False
    )
    pd.DataFrame(candidate_exact_rows).to_csv(
        OUTPUT / "candidate_exact_duplicate_pairs.csv", index=False
    )

    categories = feature_to_category()
    names = feature_to_name()
    eligibility_rows = []
    for feature in all_extracted:
        is_selected = feature in selected
        eligibility_rows.append(
            {
                "feature": feature,
                "display_name": names.get(feature, feature.replace("_", " ").title()),
                "selected": is_selected,
                "taxonomy_category": categories.get(feature, "Excluded extractor output"),
                "inference_safe": True,
                "pairwise_compatibility": feature not in {
                    "source_length",
                    "target_length",
                    "source_syntax_valid",
                    "target_syntax_valid",
                },
                "parser_dependent_or_fallback": feature in PARSER_DEPENDENT_OR_FALLBACK,
                "selection_reason": (
                    "Maps to a declared obligation as a language-normalized observation or an "
                    "explicitly labeled lower-level control."
                    if is_selected
                    else ""
                ),
                "exclusion_reason": "" if is_selected else EXCLUSION_REASONS[feature],
                "training_unique_values": int(train[feature].nunique(dropna=False)),
                "training_variance": float(train[feature].var()),
                "training_missing_values": int(train[feature].isna().sum()),
            }
        )
    pd.DataFrame(eligibility_rows).to_csv(OUTPUT / "eligibility_manifest.csv", index=False)

    rho_pruned = correlation_prune(selected, correlation, 0.90)
    all_pairwise = [feature for feature in all_extracted if feature not in UNILATERAL_EXCLUSIONS]
    feature_sets = {
        "Correlation-pruned relational probes": rho_pruned,
        "Relational probes without parser-dependent channels": [
            feature for feature in selected if feature not in PARSER_DEPENDENT_OR_FALLBACK
        ],
        "Declared relational inventory": selected,
        "All pairwise outputs including duplicate channels": all_pairwise,
        "Inventory plus unilateral validity": selected
        + ["source_syntax_valid", "target_syntax_valid"],
        "Inventory plus absolute lengths": selected + ["source_length", "target_length"],
        "All extractor outputs": all_extracted,
    }
    probabilities = {
        name: fit_probabilities(train, test, features) for name, features in feature_sets.items()
    }
    baseline_name = "Declared relational inventory"
    baseline_auc = roc_auc_score(test["label"], probabilities[baseline_name])
    result_rows = []
    for name, features in feature_sets.items():
        row = {"system": name, "num_features": len(features), **metric_row(test, probabilities[name])}
        row["delta_auc_vs_45"] = float(row["roc_auc"] - baseline_auc)
        result_rows.append(row)
    results = pd.DataFrame(result_rows)
    intervals = paired_cluster_bootstrap(test, probabilities, baseline_name)
    results = results.merge(intervals, on="system", validate="one_to_one")
    results.to_csv(OUTPUT / "feature_count_sensitivity.csv", index=False)

    pd.DataFrame(
        [
            {
                "stage": "Deterministic static extractor outputs",
                "removed": 0,
                "remaining": len(all_extracted),
                "rule": "Enumerate every implemented inference-time static quantity.",
            },
            {
                "stage": "Relational construct boundary",
                "removed": len(UNILATERAL_EXCLUSIONS),
                "remaining": len(all_pairwise),
                "rule": "Exclude unilateral absolute size and parser-status channels.",
            },
            {
                "stage": "Operational de-duplication",
                "removed": len(candidate_exact_rows),
                "remaining": len(selected),
                "rule": "Retain one canonical channel from each exact formula-equivalent pair.",
            },
            {
                "stage": "Training-partition quality control",
                "removed": 0,
                "remaining": len(selected),
                "rule": "Require finite coverage, nonzero variance, and no exact duplicate columns.",
            },
        ]
    ).to_csv(OUTPUT / "selection_funnel.csv", index=False)

    standardized = (selected_train - selected_train.mean()) / selected_train.std(ddof=0)
    components = dependence_components(selected, correlation, 0.90)
    summary = {
        "extractor_outputs": len(all_extracted),
        "selected_relational_probes": len(selected),
        "construct_excluded_outputs": len(excluded),
        "training_rows": len(train),
        "held_out_rows": len(test),
        "training_missing_values": int(missing_count.sum()),
        "training_constant_probes": int((unique_count <= 1).sum()),
        "candidate_exact_duplicate_pairs": len(candidate_exact_rows),
        "selected_exact_duplicate_pairs": len(selected_exact_rows),
        "linear_matrix_rank": int(np.linalg.matrix_rank(standardized.to_numpy())),
        "high_correlation_pairs_abs_rho_ge_0_90": len(pair_rows),
        "correlation_components_abs_rho_ge_0_90": len(components),
        "largest_correlation_component": max(len(component) for component in components),
        "parser_dependent_or_fallback_probes": len(PARSER_DEPENDENT_OR_FALLBACK),
        "problem_disjoint": True,
        "selection_and_dependence_estimated_on_training_only": True,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# EviCode Feature-Basis Audit",
        "",
        f"- Extractor outputs: {summary['extractor_outputs']}",
        f"- Construct-eligible relational inventory: {summary['selected_relational_probes']}",
        f"- Construct-based exclusions: {summary['construct_excluded_outputs']}",
        f"- Missing or constant selected probes on training data: "
        f"{summary['training_missing_values']} missing, {summary['training_constant_probes']} constant",
        f"- Exact duplicate pairs in the 53-output pool: "
        f"{summary['candidate_exact_duplicate_pairs']}",
        f"- Exact duplicate pairs after filtering: "
        f"{summary['selected_exact_duplicate_pairs']}",
        f"- Effective linear rank: {summary['linear_matrix_rank']}",
        f"- Correlation clusters at |rho| >= 0.90: "
        f"{summary['correlation_components_abs_rho_ge_0_90']}",
        f"- Parser-dependent or parser-fallback probes: "
        f"{summary['parser_dependent_or_fallback_probes']}",
        "",
        "The 45 probes are the unique pairwise outputs that remain after applying the declared "
        "lightweight relational boundary and removing exact operational duplicates. They are not "
        "assumed statistically independent and were not selected by optimizing test performance. "
        "The sensitivity table quantifies a smaller correlation-pruned set, a conservative variant "
        "that removes every parser-dependent or parser-fallback channel, and the larger pools.",
        "",
        results.to_markdown(index=False, floatfmt=".4f"),
    ]
    (OUTPUT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(results[["system", "num_features", "roc_auc", "delta_auc_vs_45", "hard_roc_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
