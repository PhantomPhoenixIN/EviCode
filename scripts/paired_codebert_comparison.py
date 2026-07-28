"""Compare frozen CodeBERT with EviCode learners using paired problem bootstraps."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "evicode_ablation_results.zip"
LOCAL = ROOT / "outputs" / "model_capacity_ablation" / "predictions.parquet"
OUTPUT = ROOT / "outputs" / "model_capacity_ablation" / "representation_comparisons.csv"
SEED = 42
BOOTSTRAPS = 500


def interval(
    frame: pd.DataFrame, first: str, second: str
) -> tuple[float, float, float]:
    y = frame["label"].to_numpy()
    first_probability = frame[first].to_numpy()
    second_probability = frame[second].to_numpy()
    groups = frame["problem_id"].to_numpy()
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        if np.unique(y[indices]).size == 2:
            estimates.append(
                roc_auc_score(y[indices], first_probability[indices])
                - roc_auc_score(y[indices], second_probability[indices])
            )
    observed = roc_auc_score(y, first_probability) - roc_auc_score(
        y, second_probability
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(observed), float(low), float(high)


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        codebert = pd.read_parquet(
            io.BytesIO(
                archive.read("frozen_codebert_baseline/predictions.parquet")
            )
        )
    local = pd.read_parquet(LOCAL)
    keys = ["example_id", "problem_id", "quality_score", "label"]
    frame = codebert.merge(local, on=keys, validate="one_to_one")
    rows = []
    comparisons = {
        "CodeBERT minus Logistic Regression": (
            "codebert_probability",
            "Logistic regression",
        ),
        "CodeBERT minus XGBoost": ("codebert_probability", "XGBoost"),
    }
    for scope, selected in {
        "All grades": np.ones(len(frame), dtype=bool),
        "Score 2 vs 3": frame["quality_score"].isin([2, 3]).to_numpy(),
    }.items():
        subset = frame.loc[selected].reset_index(drop=True)
        for comparison, (first, second) in comparisons.items():
            difference, low, high = interval(subset, first, second)
            rows.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "n": len(subset),
                    "auc_difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_iterations": BOOTSTRAPS,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
