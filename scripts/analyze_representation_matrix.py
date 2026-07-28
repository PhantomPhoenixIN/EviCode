"""Run paired problem-clustered comparisons for representation ablations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATION_DIR = ROOT / "outputs" / "frozen_representation_ablation_full"
REPRESENTATIONS = REPRESENTATION_DIR / "predictions.parquet"
EVICODE = ROOT / "outputs" / "model_capacity_ablation" / "predictions.parquet"
OUTPUT = REPRESENTATION_DIR / "paired_comparisons.csv"
SEED = 42
BOOTSTRAPS = 500


def paired_interval(
    frame: pd.DataFrame, first: str, second: str
) -> tuple[float, float, float]:
    y = frame["label"].to_numpy()
    groups = frame["problem_id"].to_numpy()
    first_probability = frame[first].to_numpy()
    second_probability = frame[second].to_numpy()
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    estimates = []
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
    representation = pd.read_parquet(REPRESENTATIONS)
    evicode = pd.read_parquet(EVICODE)
    keys = ["example_id", "problem_id", "quality_score", "label"]
    frame = representation.merge(evicode, on=keys, validate="one_to_one")
    comparisons = []
    names = ["CodeBERT", "GraphCodeBERT", "UniXcoder", "CodeT5"]
    for name in names:
        comparisons.append(
            (
                "learner gain",
                f"{name}__XGBoost",
                f"{name}__Logistic Regression",
            )
        )
        comparisons.append(
            (
                "representation vs EviCode under Logistic Regression",
                f"{name}__Logistic Regression",
                "Logistic regression",
            )
        )
        comparisons.append(
            (
                "representation vs EviCode under XGBoost",
                f"{name}__XGBoost",
                "XGBoost",
            )
        )
    for name in ["CodeBERT", "UniXcoder", "CodeT5"]:
        comparisons.append(
            (
                "best frozen representation comparison",
                "GraphCodeBERT__XGBoost",
                f"{name}__XGBoost",
            )
        )

    rows = []
    for scope, selected in {
        "All grades": np.ones(len(frame), dtype=bool),
        "Score 2 vs 3": frame["quality_score"].isin([2, 3]).to_numpy(),
    }.items():
        subset = frame.loc[selected].reset_index(drop=True)
        for question, first, second in comparisons:
            difference, low, high = paired_interval(subset, first, second)
            rows.append(
                {
                    "question": question,
                    "scope": scope,
                    "first": first,
                    "second": second,
                    "auc_difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_iterations": BOOTSTRAPS,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
