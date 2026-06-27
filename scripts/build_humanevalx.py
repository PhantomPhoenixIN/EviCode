"""Build the full supported-language HumanEval-X verification dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.data import ProgramRecord, build_pairs, parse_problem_id  # noqa: E402
from evicode.io import write_jsonl  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    return add_common_args(parser).parse_args()


def main() -> int:
    """Build HumanEval-X records and verification pairs."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    examples_path = output_dir / "verification_examples.jsonl"
    manifest_path = output_dir / "manifest.json"
    if examples_path.exists() and manifest_path.exists() and args.resume and not args.force:
        print(f"Skipping dataset build because {examples_path} exists.")
        return 0
    if examples_path.exists() and not args.force:
        raise FileExistsError(f"{examples_path} exists. Use --resume or --force.")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.dry_run:
        print(config)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "build_humanevalx", "started", {"config": str(args.config)})
    languages = config["dataset"]["languages"]
    hf_name = config["dataset"]["hf_name"]
    split = config["dataset"]["split"]
    records: dict[str, list[ProgramRecord]] = {}
    for language in languages:
        dataset = load_dataset(hf_name, language, split=split, trust_remote_code=True)
        records[language] = [
            ProgramRecord(
                problem_id=parse_problem_id(row["task_id"]),
                language=language,
                task_id=row["task_id"],
                prompt=row["prompt"],
                declaration=row["declaration"],
                canonical_solution=row["canonical_solution"],
                test=row["test"],
                example_test=row["example_test"],
            )
            for row in dataset
        ]
        write_jsonl(output_dir / f"{language}_programs.jsonl", [record.__dict__ for record in records[language]])
    examples = build_pairs(records, languages=languages, seed=config["project"]["seed"])
    write_jsonl(examples_path, [example.to_json() for example in examples])
    manifest = {
        "hf_name": hf_name,
        "split": split,
        "languages": languages,
        "num_programs": {language: len(records[language]) for language in languages},
        "num_examples": len(examples),
        "labels": {
            "positive": sum(example.label == 1 for example in examples),
            "negative": sum(example.label == 0 for example in examples),
        },
    }
    write_json(manifest_path, manifest)
    mark_stage(output_dir, "build_humanevalx", "completed", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
