"""Statistical validation for EviCode experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Predictions CSV from run_experiments.py.")
    return add_common_args(parser).parse_args()


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, seed: int, n: int = 1000) -> tuple[float, float]:
    """Bootstrap a 95 percent confidence interval for F1."""
    rng = np.random.default_rng(seed)
    scores = []
    indices = np.arange(len(y_true))
    for _ in range(n):
        sample = rng.choice(indices, size=len(indices), replace=True)
        scores.append(f1_score(y_true[sample], y_pred[sample], zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, float | int]:
    """Exact McNemar test using discordant pairs."""
    b01 = int(np.sum(~a_correct & b_correct))
    b10 = int(np.sum(a_correct & ~b_correct))
    discordant = b01 + b10
    p_value = 1.0 if discordant == 0 else float(binomtest(min(b01, b10), discordant, 0.5).pvalue)
    return {"b01": b01, "b10": b10, "discordant": discordant, "p_value": p_value}


def main() -> int:
    """Run statistical analysis."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    ci_path = output_dir / "bootstrap_f1.csv"
    mc_path = output_dir / "mcnemar.csv"
    if ci_path.exists() and mc_path.exists() and args.resume and not args.force:
        print(f"Skipping statistics because {ci_path} exists.")
        return 0
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "statistical_analysis", "started", {})
    predictions = pd.read_csv(args.predictions)
    ci_rows = []
    for system, group in predictions.groupby("system"):
        y_true = group["label"].to_numpy()
        y_pred = group["prediction"].to_numpy()
        lo, hi = bootstrap_f1(y_true, y_pred, seed=config["project"]["seed"])
        ci_rows.append({"system": system, "f1": f1_score(y_true, y_pred), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(ci_rows).to_csv(ci_path, index=False)

    pivot = predictions.pivot(index="example_id", columns="system", values="prediction")
    labels = predictions.drop_duplicates("example_id").set_index("example_id")["label"]
    comparisons = []
    baseline_names = ["execution_full_only", "execution_example_only", "static_only"]
    for baseline in baseline_names:
        if baseline not in pivot.columns:
            continue
        base_correct = pivot[baseline].astype(int).eq(labels)
        for system in ["all_evidence", "static_plus_example_execution", "static_plus_full_execution"]:
            if system not in pivot.columns or system == baseline:
                continue
            sys_correct = pivot[system].astype(int).eq(labels)
            row = {"baseline": baseline, "system": system}
            row.update(mcnemar(base_correct.to_numpy(), sys_correct.to_numpy()))
            comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(mc_path, index=False)
    summary = {"bootstrap_rows": len(ci_rows), "mcnemar_rows": len(comparisons)}
    write_json(output_dir / "statistics_summary.json", summary)
    mark_stage(output_dir, "statistical_analysis", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
