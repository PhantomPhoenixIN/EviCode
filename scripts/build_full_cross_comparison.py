"""Merge EviCode and frozen-representation learner results into one table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVICODE = ROOT / "outputs" / "model_capacity_ablation" / "results.csv"
REPRESENTATIONS = ROOT / "outputs" / "frozen_representation_ablation_full" / "results.csv"
OUTPUT = ROOT / "outputs" / "full_cross_comparison.csv"


def reshape(frame: pd.DataFrame) -> pd.DataFrame:
    values = ["roc_auc", "pr_auc", "f1_at_0_5"]
    wide = frame.pivot(index=["representation", "learner"], columns="scope", values=values)
    wide.columns = [f"{metric}_{scope.lower().replace(' ', '_')}" for metric, scope in wide.columns]
    return wide.reset_index()


def main() -> None:
    representation = pd.read_csv(REPRESENTATIONS)
    evicode = pd.read_csv(EVICODE).rename(columns={"learner": "learner"})
    evicode["representation"] = "EviCode observations"
    evicode = evicode.rename(columns={"f1_at_0_5": "f1_at_0_5"})
    columns = [
        "representation",
        "learner",
        "scope",
        "roc_auc",
        "pr_auc",
        "f1_at_0_5",
    ]
    combined = pd.concat([evicode[columns], representation[columns]], ignore_index=True)
    order = {
        name: index
        for index, name in enumerate(
            ["EviCode observations", "CodeBERT", "GraphCodeBERT", "UniXcoder", "CodeT5"]
        )
    }
    learner_order = {
        name.lower(): index
        for index, name in enumerate(
            [
                "Logistic Regression",
                "Random Forest",
                "Extra Trees",
                "Histogram Gradient Boosting",
                "XGBoost",
                "LightGBM",
                "Multilayer Perceptron",
            ]
        )
    }
    wide = reshape(combined)
    wide["representation_order"] = wide["representation"].map(order)
    wide["learner_order"] = wide["learner"].str.lower().map(learner_order)
    wide = wide.sort_values(["representation_order", "learner_order"]).drop(
        columns=["representation_order", "learner_order"]
    )
    wide.to_csv(OUTPUT, index=False)
    print(wide.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
