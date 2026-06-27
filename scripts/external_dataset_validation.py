"""Validate a Python-to-Java-trained static verifier on external translation corpora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    parser.add_argument("--train-evidence", required=True, help="HumanEval-X rich evidence JSONL.")
    parser.add_argument("--max-pairs", type=int, default=500, help="Maximum positive pairs per external dataset.")
    parser.add_argument("--train-source-language", default="python")
    parser.add_argument("--train-target-language", default="java")
    return add_common_args(parser).parse_args()


def normalize_language(language: str) -> str:
    """Normalize language names used by external datasets."""
    aliases = {"c++": "cpp", "c#": "csharp", "cs": "csharp", "javascript": "js"}
    value = language.strip().lower()
    return aliases.get(value, value)


def train_static_model(train_evidence: Path, config: dict, source_language: str, target_language: str):
    """Train the transparent static verifier on one HumanEval-X direction."""
    train = pd.DataFrame(read_jsonl(train_evidence))
    train = train[
        (train["source_language"].map(normalize_language) == normalize_language(source_language))
        & (train["target_language"].map(normalize_language) == normalize_language(target_language))
    ].copy()
    if train.empty:
        raise ValueError("No training rows remain after source/target filtering.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=config["project"]["seed"]),
    )
    model.fit(train[STATIC_FEATURES].fillna(0.0), train["label"].astype(int))
    return model, int(len(train))


def add_mismatches(positive_rows: list[dict]) -> list[dict]:
    """Create one deterministic mismatched negative for each positive row."""
    rows = []
    positives = list(positive_rows)
    for row in positives:
        rows.append({**row, "label": 1, "negative_type": "none"})
    if len(positives) < 2:
        return rows
    targets = [row["target_code"] for row in positives]
    target_ids = [row["target_id"] for row in positives]
    for index, row in enumerate(positives):
        mismatch = (index + 1) % len(positives)
        rows.append(
            {
                **row,
                "target_code": targets[mismatch],
                "target_id": target_ids[mismatch],
                "label": 0,
                "negative_type": "mismatch",
            }
        )
    return rows


def codexglue_cs_java(max_pairs: int) -> list[dict]:
    """Build C# to Java aligned/mismatched examples from CodeXGLUE."""
    dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans", split="test", trust_remote_code=True)
    positives = []
    for row in dataset.select(range(min(max_pairs, len(dataset)))):
        positives.append(
            {
                "dataset": "CodeXGLUE-CS-Java",
                "example_id": f"codexglue_{row['id']}",
                "source_language": "csharp",
                "target_language": "java",
                "source_code": row["cs"],
                "target_code": row["java"],
                "target_id": str(row["id"]),
            }
        )
    return add_mismatches(positives)


def humanevalx_extra_to_java(max_pairs: int) -> list[dict]:
    """Build C++/Go to Java canonical-pair examples from HumanEval-X held-out languages."""
    target = load_dataset("THUDM/humaneval-x", "java", split="test", trust_remote_code=True)
    target_by_id = {row["task_id"].split("/")[-1]: row for row in target}
    rows = []
    for source_name, source_language in [("cpp", "cpp"), ("go", "go")]:
        source = load_dataset("THUDM/humaneval-x", source_name, split="test", trust_remote_code=True)
        positives = []
        for row in source:
            problem_id = row["task_id"].split("/")[-1]
            if problem_id not in target_by_id:
                continue
            java_row = target_by_id[problem_id]
            positives.append(
                {
                    "dataset": f"HumanEvalX-{source_language}-Java",
                    "example_id": f"humanevalx_{source_language}_{problem_id}",
                    "source_language": source_language,
                    "target_language": "java",
                    "source_code": row["declaration"] + row["canonical_solution"],
                    "target_code": java_row["declaration"] + java_row["canonical_solution"],
                    "target_id": problem_id,
                }
            )
            if len(positives) >= max_pairs:
                break
        rows.extend(add_mismatches(positives))
    return rows


def xlcost_to_java(max_pairs: int) -> list[dict]:
    """Build XLCoST program-level source-to-Java examples by aligning natural-language task text."""
    java = load_dataset("codeparrot/xlcost-text-to-code", "Java-program-level", split="test", trust_remote_code=True)
    java_by_text = {row["text"]: row["code"] for row in java}
    rows = []
    configs = [("Python-program-level", "python"), ("C++-program-level", "cpp"), ("Csharp-program-level", "csharp")]
    for config_name, source_language in configs:
        source = load_dataset("codeparrot/xlcost-text-to-code", config_name, split="test", trust_remote_code=True)
        positives = []
        for index, row in enumerate(source):
            if row["text"] not in java_by_text:
                continue
            positives.append(
                {
                    "dataset": f"XLCoST-{source_language}-Java",
                    "example_id": f"xlcost_{source_language}_{index}",
                    "source_language": source_language,
                    "target_language": "java",
                    "source_code": row["code"],
                    "target_code": java_by_text[row["text"]],
                    "target_id": row["text"],
                }
            )
            if len(positives) >= max_pairs:
                break
        rows.extend(add_mismatches(positives))
    return rows


def evaluate_rows(model, rows: list[dict], output_dir: Path) -> pd.DataFrame:
    """Extract evidence and evaluate external rows."""
    evidence_rows = []
    jsonl_path = output_dir / "external_validation_evidence.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            features = static_features(row["source_code"], row["target_code"], row["source_language"], row["target_language"])
            record = {k: row[k] for k in row if k not in {"source_code", "target_code"}}
            record.update(features)
            evidence_rows.append(record)
            handle.write(pd.Series(record).to_json() + "\n")
            handle.flush()
    frame = pd.DataFrame(evidence_rows)
    probabilities = model.predict_proba(frame[STATIC_FEATURES].fillna(0.0))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    frame["verification_confidence"] = probabilities
    frame["prediction"] = predictions
    frame.to_csv(output_dir / "external_validation_predictions.csv", index=False)
    return frame


def summarize(frame: pd.DataFrame, training_rows: int) -> tuple[pd.DataFrame, dict]:
    """Summarize external validation by dataset."""
    rows = []
    for dataset_name, group in frame.groupby("dataset"):
        labels = group["label"].to_numpy(dtype=int)
        probabilities = group["verification_confidence"].to_numpy(dtype=float)
        predictions = group["prediction"].to_numpy(dtype=int)
        rows.append(
            {
                "dataset": dataset_name,
                "rows": int(len(group)),
                "positive": int(labels.sum()),
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, zero_division=0),
                "roc_auc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan"),
                "mean_confidence": float(probabilities.mean()),
            }
        )
    summary_frame = pd.DataFrame(rows).sort_values("dataset")
    summary = {
        "status": "completed",
        "training_rows": training_rows,
        "rows": int(len(frame)),
        "datasets": sorted(frame["dataset"].unique()),
        "mean_roc_auc": float(np.nanmean(summary_frame["roc_auc"])),
    }
    return summary_frame, summary


def main() -> int:
    """Run external validation."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "external_validation_summary.json"
    if summary_path.exists() and args.resume and not args.force:
        print(f"Skipping external validation because {summary_path} exists.")
        return 0
    if args.dry_run:
        print("external validation dry run")
        return 0
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "external_dataset_validation", "started", {})
    model, training_rows = train_static_model(
        Path(args.train_evidence), config, args.train_source_language, args.train_target_language
    )
    external_rows = []
    external_rows.extend(codexglue_cs_java(args.max_pairs))
    external_rows.extend(humanevalx_extra_to_java(args.max_pairs))
    external_rows.extend(xlcost_to_java(args.max_pairs))
    frame = evaluate_rows(model, external_rows, output_dir)
    summary_frame, summary = summarize(frame, training_rows)
    summary_frame.to_csv(output_dir / "external_validation_summary.csv", index=False)
    write_json(summary_path, summary)
    mark_stage(output_dir, "external_dataset_validation", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
