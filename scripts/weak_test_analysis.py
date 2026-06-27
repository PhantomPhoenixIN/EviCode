"""Document weak-test robustness status without fabricating retained-test outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="Metrics CSV.")
    return add_common_args(parser).parse_args()


def main() -> int:
    """Generate weak-test status results."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / "weak_test_results.csv"
    if output_path.exists() and args.resume and not args.force:
        print(f"Skipping weak-test analysis because {output_path} exists.")
        return 0
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "weak_test_analysis", "started", {})
    metrics = pd.read_csv(args.metrics)
    rows = []
    for retention in [1.0, 0.75, 0.5, 0.25, 0.1]:
        rows.append(
            {
                "retention": retention,
                "system": "static_only",
                "mean_f1": float(metrics.loc[metrics["system"] == "static_only", "f1"].iloc[0]),
                "std_f1": 0.0,
                "feasible": True,
                "note": "Static evidence is independent of retained dynamic tests.",
            }
        )
        rows.append(
            {
                "retention": retention,
                "system": "execution_only",
                "mean_f1": float("nan"),
                "std_f1": float("nan"),
                "feasible": False,
                "note": "Requires normalized individual tests; not fabricated from aggregate HumanEval-X programs.",
            }
        )
        rows.append(
            {
                "retention": retention,
                "system": "static_plus_weak_execution",
                "mean_f1": float("nan"),
                "std_f1": float("nan"),
                "feasible": False,
                "note": "Pending normalized retained-test execution.",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    summary = {"rows": len(frame), "feasible_rows": int(frame["feasible"].sum())}
    write_json(output_dir / "weak_test_results.json", frame.to_dict(orient="records"))
    write_json(output_dir / "weak_test_summary.json", summary)
    mark_stage(output_dir, "weak_test_analysis", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
