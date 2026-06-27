"""Run EviCode evidence-fusion experiments on extracted evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.io import read_jsonl  # noqa: E402
from evicode.taxonomy import feature_to_category  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


CATEGORIES = feature_to_category()
STATIC_FEATURES = [feature for feature, category in CATEGORIES.items() if category != "Dynamic"]
LANGUAGE_NORMALIZED_FEATURES = [
    feature for feature, category in CATEGORIES.items() if category.startswith("Normalized")
]
WEAK_PROXY_FEATURES = [feature for feature, category in CATEGORIES.items() if category == "Weak-proxy"]

CATEGORY_FEATURES = {
    category.lower().replace("-", "_"): [
        feature for feature, feature_category in CATEGORIES.items() if feature_category == category
    ]
    for category in sorted(set(CATEGORIES.values()))
}
CATEGORY_FEATURES.pop("dynamic", None)


def cat(name: str) -> list[str]:
    """Return a category feature list if present."""
    return CATEGORY_FEATURES.get(name, [])

FEATURE_SETS = {
    "syntax_only": ["syntax_proxy"],
    "ast_only": ["ast_similarity"],
    "cfg_only": ["control_flow_similarity"],
    "api_only": ["api_similarity"],
    "identifier_only": ["identifier_similarity"],
    "retrieval_only": ["retrieval_similarity"],
    "static_only": STATIC_FEATURES,
    "language_normalized_only": LANGUAGE_NORMALIZED_FEATURES,
    "weak_proxy_only": WEAK_PROXY_FEATURES,
    "lexical_only": cat("lexical"),
    "syntactic_only": cat("syntactic"),
    "structural_only": cat("normalized_control") + cat("normalized_structure") + cat("normalized_operator"),
    "semantic_static_only": cat("semantic_static") + cat("normalized_identifier") + cat("normalized_dataflow") + cat("normalized_call"),
    "normalized_control_only": cat("normalized_control"),
    "normalized_structure_only": cat("normalized_structure"),
    "normalized_operator_only": cat("normalized_operator"),
    "normalized_identifier_only": cat("normalized_identifier"),
    "normalized_dataflow_only": cat("normalized_dataflow"),
    "normalized_call_only": cat("normalized_call"),
    "execution_example_only": ["execution_passed_example"],
    "execution_full_only": ["execution_passed_full"],
    "dynamic_only": ["execution_passed_example", "execution_passed_full"],
    "static_plus_example_execution": STATIC_FEATURES + ["execution_passed_example"],
    "static_plus_full_execution": STATIC_FEATURES + ["execution_passed_full"],
    "all_evidence": STATIC_FEATURES + ["execution_passed_example", "execution_passed_full"],
}

for left_name, left_features in CATEGORY_FEATURES.items():
    for right_name, right_features in CATEGORY_FEATURES.items():
        if left_name < right_name:
            FEATURE_SETS[f"{left_name}_plus_{right_name}"] = sorted(set(left_features + right_features))

for feature in STATIC_FEATURES:
    FEATURE_SETS[f"static_without_{feature}"] = [item for item in STATIC_FEATURES if item != feature]

for feature in STATIC_FEATURES:
    FEATURE_SETS[f"example_without_{feature}"] = [
        item for item in STATIC_FEATURES + ["execution_passed_example"] if item != feature
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Evidence JSONL file.")
    return add_common_args(parser).parse_args()


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_features: int) -> dict:
    """Build one metric row."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "system": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "num_features": n_features,
    }


def main() -> int:
    """Run experiments."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and args.resume and not args.force:
        print(f"Skipping experiments because {metrics_path} exists.")
        return 0
    if metrics_path.exists() and not args.force:
        raise FileExistsError(f"{metrics_path} exists. Use --resume or --force.")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "run_experiments", "started", {"input": args.input})
    frame = pd.DataFrame(read_jsonl(Path(args.input)))
    bool_cols = [col for col in frame.columns if col.startswith("execution_passed")]
    for col in bool_cols:
        frame[col] = frame[col].astype(float)
    y = frame["label"].to_numpy()
    groups = frame["problem_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=config["experiment"]["test_size"], random_state=config["project"]["seed"])
    train_idx, test_idx = next(splitter.split(frame, y, groups))
    metrics = []
    predictions = []
    for name, features in FEATURE_SETS.items():
        if not features:
            continue
        x = frame[features].fillna(0.0).to_numpy(dtype=float)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=config["project"]["seed"]),
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        prob = model.predict_proba(x[test_idx])[:, 1]
        metrics.append(metric_row(name, y[test_idx], pred, prob, len(features)))
        for idx, p, pr in zip(test_idx, pred, prob, strict=True):
            predictions.append(
                {
                    "system": name,
                    "example_id": frame.loc[idx, "example_id"],
                    "problem_id": int(frame.loc[idx, "problem_id"]),
                    "source_language": frame.loc[idx, "source_language"],
                    "target_language": frame.loc[idx, "target_language"],
                    "label": int(frame.loc[idx, "label"]),
                    "prediction": int(p),
                    "probability": float(pr),
                }
            )
    metrics_frame = pd.DataFrame(metrics).sort_values("f1", ascending=False)
    metrics_frame.to_csv(metrics_path, index=False)
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    summary = {
        "num_examples": int(len(frame)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "best_system": str(metrics_frame.iloc[0]["system"]),
        "best_f1": float(metrics_frame.iloc[0]["f1"]),
    }
    write_json(output_dir / "summary.json", summary)
    mark_stage(output_dir, "run_experiments", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
