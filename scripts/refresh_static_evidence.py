"""Refresh static evidence columns without rerunning execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.features import static_features  # noqa: E402
from evicode.io import read_jsonl, write_jsonl  # noqa: E402
from evicode.taxonomy import feature_to_category  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402


STATIC_COLUMNS = {
    feature for feature, category in feature_to_category().items() if category != "Dynamic"
} | {
    "source_syntax_valid",
    "target_syntax_valid",
    "source_length",
    "target_length",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Verification examples JSONL.")
    parser.add_argument("--evidence", required=True, help="Existing evidence JSONL.")
    return add_common_args(parser).parse_args()


def main() -> int:
    """Refresh static evidence."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / "evidence.jsonl"
    if output_path.exists() and args.resume and not args.force:
        print(f"Skipping refresh because {output_path} exists.")
        return 0
    examples = {row["example_id"]: row for row in read_jsonl(Path(args.examples))}
    refreshed = []
    for row in read_jsonl(Path(args.evidence)):
        example = examples[row["example_id"]]
        for column in STATIC_COLUMNS:
            row.pop(column, None)
        row.update(
            static_features(
                example["source_code"],
                example["target_code"],
                example["source_language"],
                example["target_language"],
            )
        )
        refreshed.append(row)
    write_jsonl(output_path, refreshed)
    print({"refreshed": len(refreshed), "output": str(output_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
