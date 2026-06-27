"""Evaluate EviCode confidence on real LLM-generated Python-to-Java translations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.features import static_features  # noqa: E402
from evicode.io import read_jsonl  # noqa: E402
from evicode.taxonomy import feature_to_category  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


STATIC_FEATURES = [feature for feature, category in feature_to_category().items() if category != "Dynamic"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", required=True, help="Directory containing LLM JSONL files.")
    parser.add_argument("--train-evidence", required=True, help="HumanEval-X rich evidence JSONL.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for debugging; 0 means all rows.")
    return add_common_args(parser).parse_args()


def model_name_from_path(path: Path) -> str:
    """Infer model name from file name."""
    name = path.name.lower()
    if "deepseek" in name:
        return "deepseekcoder"
    if "qwen" in name:
        return "qwencoder"
    if "starcoder" in name:
        return "starcoder"
    return path.stem


def candidate_code(row: dict, model_name: str) -> str | None:
    """Return the best available candidate code."""
    clean_key = f"{model_name}_translation_clean"
    raw_key = f"{model_name}_translation_raw"
    return row.get(clean_key) or row.get("translated_java_code") or row.get(raw_key)


def read_external_rows(predictions_dir: Path, max_rows: int) -> list[dict]:
    """Read and normalize external LLM prediction rows."""
    rows = []
    for path in sorted(predictions_dir.glob("*.jsonl")):
        model_name = model_name_from_path(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("input_language") != "Python" or raw.get("output_language") != "Java":
                    continue
                code = candidate_code(raw, model_name)
                if not code:
                    continue
                score = int(raw.get("score", 0))
                rows.append(
                    {
                        "example_id": f"{model_name}_{raw.get('problem_id')}_{len(rows)}",
                        "problem_id": raw.get("problem_id"),
                        "model_name": model_name,
                        "split": raw.get("split"),
                        "source_language": "python",
                        "target_language": "java",
                        "source_code": raw.get("source_code") or "",
                        "target_code": code,
                        "score": score,
                        "label": int(score == 3),
                        "failure_grade": {
                            0: "no_clean_candidate_or_unparsable",
                            1: "parsable_not_compilable",
                            2: "compilable_functionally_incorrect",
                            3: "functionally_correct",
                        }.get(score, "unknown"),
                    }
                )
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Compute reliability bins."""
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_low": low, "bin_high": high, "count": 0, "mean_confidence": np.nan, "accuracy": np.nan})
            continue
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "mean_confidence": float(probabilities[mask].mean()),
                "accuracy": float(labels[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(bins: pd.DataFrame, total: int) -> float:
    """Compute ECE from reliability bins."""
    error = 0.0
    for row in bins.dropna().itertuples(index=False):
        error += (row.count / total) * abs(row.accuracy - row.mean_confidence)
    return float(error)


def main() -> int:
    """Evaluate external LLM prediction confidence."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "llm_calibration_summary.json"
    if summary_path.exists() and args.resume and not args.force:
        print(f"Skipping LLM evaluation because {summary_path} exists.")
        return 0
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "evaluate_llm_predictions", "started", {})

    external_rows = read_external_rows(Path(args.predictions_dir), args.max_rows)
    if not external_rows:
        raise ValueError("No Python-to-Java LLM prediction rows found.")
    evidence_rows = []
    for row in external_rows:
        features = static_features(row["source_code"], row["target_code"], "python", "java")
        evidence_rows.append({**{k: row[k] for k in row if k not in {"source_code", "target_code"}}, **features})
    external = pd.DataFrame(evidence_rows)
    external.to_json(output_dir / "llm_prediction_evidence.jsonl", orient="records", lines=True)

    train = pd.DataFrame(read_jsonl(Path(args.train_evidence)))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=config["project"]["seed"]),
    )
    model.fit(train[STATIC_FEATURES].fillna(0.0), train["label"].astype(int))
    probabilities = model.predict_proba(external[STATIC_FEATURES].fillna(0.0))[:, 1]
    labels = external["label"].to_numpy(dtype=int)
    predictions = (probabilities >= 0.5).astype(int)
    external["verification_confidence"] = probabilities
    external["prediction"] = predictions
    external.to_csv(output_dir / "llm_prediction_confidence.csv", index=False)

    bins = calibration_bins(labels, probabilities)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    by_model = []
    for model_name, group in external.groupby("model_name"):
        y = group["label"].to_numpy(dtype=int)
        p = group["verification_confidence"].to_numpy()
        pred = group["prediction"].to_numpy(dtype=int)
        by_model.append(
            {
                "model_name": model_name,
                "rows": int(len(group)),
                "correct": int(y.sum()),
                "accuracy_at_0_5": accuracy_score(y, pred),
                "f1": f1_score(y, pred, zero_division=0),
                "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                "brier": brier_score_loss(y, p),
                "mean_confidence": float(p.mean()),
                "empirical_accuracy": float(y.mean()),
            }
        )
    by_model_frame = pd.DataFrame(by_model)
    by_model_frame.to_csv(output_dir / "llm_model_calibration.csv", index=False)

    by_score = (
        external.groupby(["score", "failure_grade"])
        .agg(rows=("label", "size"), mean_confidence=("verification_confidence", "mean"), empirical_accuracy=("label", "mean"))
        .reset_index()
    )
    by_score.to_csv(output_dir / "confidence_by_score.csv", index=False)

    summary = {
        "rows": int(len(external)),
        "correct": int(labels.sum()),
        "empirical_accuracy": float(labels.mean()),
        "accuracy_at_0_5": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(bins, len(external)),
        "mean_confidence": float(probabilities.mean()),
        "models": sorted(external["model_name"].unique()),
        "score_counts": {str(k): int(v) for k, v in external["score"].value_counts().sort_index().items()},
        "training_source": str(args.train_evidence),
        "evaluation_note": "Static verifier trained on HumanEval-X synthetic pairs and evaluated on real Python-to-Java LLM translations.",
    }
    write_json(summary_path, summary)
    mark_stage(output_dir, "evaluate_llm_predictions", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
