"""Evaluate same-information baselines on the authentic problem-disjoint split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "authentic_only" / "datasets" / "analysis_dataset.parquet"
OUTPUT = ROOT / "outputs" / "authentic_same_information_baselines"
SEED = 42
BOOTSTRAPS = 500

METADATA = {"example_id", "problem_id", "language", "generator", "quality_score", "label"}
SURFACE = ["token_jaccard", "edit_similarity", "length_ratio"]
VALIDITY = ["ln_syntax_both_valid"]


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        selected = membership == index
        if selected.any():
            value += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(value)


def clustered_auc_interval(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sample])
        if np.unique(y[indices]).size == 2:
            estimates.append(roc_auc_score(y[indices], probability[indices]))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def metrics(
    name: str,
    scope: str,
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    prediction = (probability >= 0.5).astype(int)
    low, high = clustered_auc_interval(y, probability, groups, rng)
    return {
        "baseline": name,
        "scope": scope,
        "n": len(y),
        "positives": int(y.sum()),
        "roc_auc": roc_auc_score(y, probability),
        "roc_auc_ci_low": low,
        "roc_auc_ci_high": high,
        "pr_auc": average_precision_score(y, probability),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "brier": brier_score_loss(y, probability),
        "ece": expected_calibration_error(y, probability),
    }


def main() -> None:
    frame = pd.read_parquet(DATA)
    feature_names = [column for column in frame.columns if column not in METADATA]
    normalized = [column for column in feature_names if column.startswith("ln_")]
    unnormalized = [column for column in feature_names if not column.startswith("ln_")]
    feature_sets = {
        "Token overlap only": ["token_jaccard"],
        "Surface only": SURFACE,
        "Validity only": VALIDITY,
        "Unnormalized static": unnormalized,
        "Language-normalized only": normalized,
        "No surface or validity": [
            column for column in feature_names if column not in set(SURFACE + VALIDITY)
        ],
        "EviCode full": feature_names,
    }

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_index, test_index = next(
        splitter.split(frame, frame["label"], groups=frame["problem_id"])
    )
    train = frame.iloc[train_index]
    test = frame.iloc[test_index]
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []

    prior = np.full(len(test), train["label"].mean())
    rows.append(
        metrics(
            "Class prior",
            "All grades",
            test["label"].to_numpy(),
            prior,
            test["problem_id"].to_numpy(),
            rng,
        )
    )

    for name, features in feature_sets.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        )
        model.fit(train[features], train["label"])
        probability = model.predict_proba(test[features])[:, 1]
        rows.append(
            metrics(
                name,
                "All grades",
                test["label"].to_numpy(),
                probability,
                test["problem_id"].to_numpy(),
                rng,
            )
        )

        hard = test["quality_score"].isin([2, 3]).to_numpy()
        rows.append(
            metrics(
                name,
                "Score 2 vs 3",
                test.loc[hard, "label"].to_numpy(),
                probability[hard],
                test.loc[hard, "problem_id"].to_numpy(),
                rng,
            )
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "baseline_results.csv", index=False)
    manifest = {
        "data": str(DATA.relative_to(ROOT)),
        "seed": SEED,
        "test_size": 0.30,
        "bootstrap_iterations": BOOTSTRAPS,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_problems": int(train["problem_id"].nunique()),
        "test_problems": int(test["problem_id"].nunique()),
        "feature_sets": feature_sets,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
