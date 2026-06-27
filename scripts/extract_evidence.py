"""Extract static and execution evidence for HumanEval-X verification examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.execution.runner import run_candidate, select_test  # noqa: E402
from evicode.features import static_features  # noqa: E402
from evicode.io import read_jsonl  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input verification_examples.jsonl.")
    return add_common_args(parser).parse_args()


def main() -> int:
    """Extract evidence."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    evidence_path = output_dir / "evidence.jsonl"
    manifest_path = output_dir / "manifest.json"
    if evidence_path.exists() and manifest_path.exists() and args.resume and not args.force:
        print(f"Skipping evidence extraction because {evidence_path} exists.")
        return 0
    if evidence_path.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{evidence_path} exists. Use --resume or --force.")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "extract_evidence", "started", {"input": args.input})
    rows = read_jsonl(Path(args.input))
    budgets = config["execution"]["budgets"]
    timeout = int(config["execution"]["timeout_seconds"])
    completed_ids: set[str] = set()
    if evidence_path.exists() and args.resume and not args.force:
        completed_ids = {row["example_id"] for row in read_jsonl(evidence_path)}
    mode = "w" if args.force or not evidence_path.exists() else "a"
    processed = len(completed_ids)
    with evidence_path.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="extract-evidence"):
            if row["example_id"] in completed_ids:
                continue
            output_row = extract_row(row, budgets, timeout)
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            if processed % 100 == 0:
                write_json(
                    manifest_path,
                    {"num_examples": processed, "total_examples": len(rows), "budgets": budgets, "partial": True},
                )
    manifest = {"num_examples": processed, "total_examples": len(rows), "budgets": budgets, "partial": False}
    write_json(manifest_path, manifest)
    mark_stage(output_dir, "extract_evidence", "completed", manifest)
    print(manifest)
    return 0


def extract_row(row: dict, budgets: list[str], timeout: int) -> dict:
    """Extract one evidence row."""
    features = static_features(
        row["source_code"],
        row["target_code"],
        row["source_language"],
        row["target_language"],
    )
    execution = {}
    for budget in budgets:
        selected_test = select_test(row["target_test"], row["target_example_test"], budget)
        result = run_candidate(row["target_language"], row["target_code"], selected_test, timeout)
        execution[f"execution_available_{budget}"] = result.available
        execution[f"execution_passed_{budget}"] = result.passed
        execution[f"execution_timed_out_{budget}"] = result.timed_out
        execution[f"execution_returncode_{budget}"] = result.returncode
    return {
        "example_id": row["example_id"],
        "problem_id": row["problem_id"],
        "source_language": row["source_language"],
        "target_language": row["target_language"],
        "label": row["label"],
        "negative_type": row["negative_type"],
        **features,
        **execution,
    }


if __name__ == "__main__":
    raise SystemExit(main())
