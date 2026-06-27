"""Summarize execution-budget behavior without fabricating unavailable tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.io import read_jsonl  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, help="Evidence JSONL.")
    parser.add_argument("--metrics", required=True, help="Metrics CSV.")
    return add_common_args(parser).parse_args()


def metric_row(name: str, budget: str, y: pd.Series, score: pd.Series, feasible: bool, note: str) -> dict:
    """Build one execution-budget row."""
    if not feasible:
        return {
            "system": name,
            "budget": budget,
            "feasible": False,
            "actual_test_source": "not_normalized",
            "accuracy": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "note": note,
        }
    pred = score.astype(int)
    try:
        auc = roc_auc_score(y, score)
    except ValueError:
        auc = float("nan")
    return {
        "system": name,
        "budget": budget,
        "feasible": True,
        "actual_test_source": budget,
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": auc,
        "note": note,
    }


def main() -> int:
    """Generate execution-budget summary."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / "execution_budget_results.csv"
    if output_path.exists() and args.resume and not args.force:
        print(f"Skipping execution budget analysis because {output_path} exists.")
        return 0
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "execution_budget_analysis", "started", {})
    evidence = pd.DataFrame(read_jsonl(Path(args.evidence)))
    metrics = pd.read_csv(args.metrics)
    y = evidence["label"].astype(int)
    rows = [
        metric_row("execution_only", "0", y, pd.Series([0] * len(y)), True, "No dynamic evidence."),
        metric_row(
            "execution_only",
            "example",
            y,
            evidence["execution_passed_example"].astype(float),
            True,
            "HumanEval-X example test program.",
        ),
        metric_row(
            "execution_only",
            "full",
            y,
            evidence["execution_passed_full"].astype(float),
            True,
            "Full HumanEval-X test program.",
        ),
    ]
    for budget in ["1", "3", "5", "10"]:
        rows.append(
            metric_row(
                "execution_only",
                budget,
                y,
                pd.Series(dtype=float),
                False,
                "Pending language-specific normalization of individual assertions.",
            )
        )
    for system in ["static_only", "static_plus_example_execution", "static_plus_full_execution", "all_evidence"]:
        row = metrics[metrics["system"] == system]
        if not row.empty:
            item = row.iloc[0].to_dict()
            item.update(
                {
                    "budget": "static/example/full" if "example" in system or "full" in system else "0",
                    "feasible": True,
                    "actual_test_source": "model_features",
                    "note": "Grouped split fusion result; classifier is an analysis tool, not the contribution.",
                }
            )
            rows.append(item)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    write_json(output_dir / "execution_budget_results.json", frame.to_dict(orient="records"))
    summary = {"rows": len(frame), "feasible_rows": int(frame["feasible"].sum())}
    write_json(output_dir / "execution_budget_summary.json", summary)
    mark_stage(output_dir, "execution_budget_analysis", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
