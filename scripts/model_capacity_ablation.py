"""Compare transparent and nonlinear learners on the same EviCode observations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "authentic_only" / "datasets" / "analysis_dataset.parquet"
OUTPUT = ROOT / "outputs" / "model_capacity_ablation"
SEED = 42
BOOTSTRAPS = 500
METADATA = {"example_id", "problem_id", "language", "generator", "quality_score", "label"}


def clustered_interval(
    y: np.ndarray, probability: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        if np.unique(y[indices]).size == 2:
            estimates.append(roc_auc_score(y[indices], probability[indices]))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def paired_auc_difference_interval(
    y: np.ndarray,
    baseline_probability: np.ndarray,
    stronger_probability: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    differences: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        if np.unique(y[indices]).size == 2:
            differences.append(
                roc_auc_score(y[indices], stronger_probability[indices])
                - roc_auc_score(y[indices], baseline_probability[indices])
            )
    observed = roc_auc_score(y, stronger_probability) - roc_auc_score(
        y, baseline_probability
    )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(low), float(high)


def main() -> None:
    frame = pd.read_parquet(DATA)
    features = [column for column in frame.columns if column not in METADATA]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_index, test_index = next(
        splitter.split(frame, frame["label"], groups=frame["problem_id"])
    )
    train = frame.iloc[train_index]
    test = frame.iloc[test_index].copy()
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=SEED,
            verbosity=-1,
        ),
        "Multilayer perceptron": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                max_iter=300,
                random_state=SEED,
            ),
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(train[features], train["label"])
        fit_seconds = time.perf_counter() - started
        probability = model.predict_proba(test[features])[:, 1]
        test[name] = probability
        for scope, selected in {
            "All grades": np.ones(len(test), dtype=bool),
            "Score 2 vs 3": test["quality_score"].isin([2, 3]).to_numpy(),
        }.items():
            y = test.loc[selected, "label"].to_numpy()
            p = probability[selected]
            groups = test.loc[selected, "problem_id"].to_numpy()
            low, high = clustered_interval(y, p, groups)
            rows.append(
                {
                    "learner": name,
                    "scope": scope,
                    "n": len(y),
                    "roc_auc": roc_auc_score(y, p),
                    "roc_auc_ci_low": low,
                    "roc_auc_ci_high": high,
                    "pr_auc": average_precision_score(y, p),
                    "f1_at_0_5": f1_score(y, p >= 0.5),
                    "fit_seconds": fit_seconds,
                }
            )
        joblib.dump(model, OUTPUT / f"{name.lower().replace(' ', '_')}.joblib")
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "results.csv", index=False)
    comparisons = []
    for stronger_name in models:
        if stronger_name == "Logistic regression":
            continue
        for scope, selected in {
            "All grades": np.ones(len(test), dtype=bool),
            "Score 2 vs 3": test["quality_score"].isin([2, 3]).to_numpy(),
        }.items():
            difference, low, high = paired_auc_difference_interval(
                test.loc[selected, "label"].to_numpy(),
                test.loc[selected, "Logistic regression"].to_numpy(),
                test.loc[selected, stronger_name].to_numpy(),
                test.loc[selected, "problem_id"].to_numpy(),
            )
            comparisons.append(
                {
                    "learner": stronger_name,
                    "scope": scope,
                    "auc_difference_vs_lr": difference,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(OUTPUT / "paired_auc_differences.csv", index=False)
    test[["example_id", "problem_id", "quality_score", "label", *models]].to_parquet(
        OUTPUT / "predictions.parquet", index=False
    )
    manifest = {
        "data": str(DATA.relative_to(ROOT)),
        "features": features,
        "seed": SEED,
        "test_size": 0.30,
        "bootstrap_iterations": BOOTSTRAPS,
        "random_forest": {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "comparison_policy": "Representative untuned learner families on one fixed split",
        "xgboost": {
            "version": "3.3.0",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "lightgbm": {
            "version": "4.6.0",
            "n_estimators": 500,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(comparison_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
