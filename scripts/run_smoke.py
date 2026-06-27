"""Run the EviCode synthetic smoke pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.evidence import CodePair, build_evidence_sources  # noqa: E402
from evicode.fusion import LogisticEvidenceFusion, build_report  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume and skip valid outputs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration only.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def synthetic_pairs() -> tuple[list[CodePair], np.ndarray]:
    """Create a tiny deterministic dataset for smoke testing."""
    pairs = [
        CodePair("p1_pos", "python", "python", "def add(a,b):\n return a+b\n", "def add(a,b):\n return a+b\n", {}),
        CodePair("p1_neg", "python", "python", "def add(a,b):\n return a+b\n", "def add(a,b):\n return a-b\n", {}),
        CodePair("p2_pos", "python", "python", "def is_even(x):\n return x%2==0\n", "def is_even(x):\n return x%2==0\n", {}),
        CodePair("p2_neg", "python", "python", "def is_even(x):\n return x%2==0\n", "def is_even(x):\n return x%2==1\n", {}),
    ]
    labels = np.array([1, 0, 1, 0])
    return pairs, labels


def main() -> int:
    """Run the smoke pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and args.resume and not args.force:
        print(f"Skipping smoke run because {summary_path} exists. Use --force to rerun.")
        return 0
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"{summary_path} exists. Use --resume to skip or --force to overwrite.")

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.dry_run:
        print(json.dumps({"config": config, "dry_run": True}, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "smoke", "started", {"config": str(args.config)})

    pairs, labels = synthetic_pairs()
    sources = build_evidence_sources(config["pipeline"]["evidence_sources"])
    matrix: list[list[float]] = []
    rows: list[dict[str, str | float | int]] = []
    for pair, label in zip(pairs, labels, strict=True):
        results = [source.run(pair) for source in sources]
        matrix.append([result.score for result in results])
        row: dict[str, str | float | int] = {"pair_id": pair.pair_id, "label": int(label)}
        row.update({result.name: result.score for result in results})
        rows.append(row)

    evidence_csv = output_dir / "evidence.csv"
    with evidence_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array(matrix)
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config["fusion"]["test_size"], random_state=config["project"]["seed"], stratify=y
    )
    model = LogisticEvidenceFusion(seed=config["project"]["seed"])
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)
    reports = []
    for pair, row in zip(pairs, matrix, strict=True):
        pair_prob = float(model.predict_proba(np.array([row]))[0])
        pair_pred = int(pair_prob >= 0.5)
        evidence = [source.run(pair) for source in sources]
        report = build_report(pair.pair_id, pair_prob, pair_pred, evidence)
        reports.append(
            {
                "pair_id": report.pair_id,
                "score": report.score,
                "prediction": report.prediction,
                "summary": report.summary,
                "likely_failure_modes": report.likely_failure_modes,
            }
        )
    write_json(output_dir / "verification_reports.json", reports)
    summary = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "num_examples": len(pairs),
        "num_evidence_sources": len(sources),
        "report_path": str(output_dir / "verification_reports.json"),
    }
    write_json(summary_path, summary)
    mark_stage(output_dir, "smoke", "completed", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
