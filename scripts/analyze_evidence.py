"""Analyze evidence informativeness, complementarity, costs, and failures."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.features import static_features  # noqa: E402
from evicode.io import read_jsonl  # noqa: E402
from evicode.taxonomy import feature_to_category, feature_to_name, taxonomy_rows  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Verification examples JSONL.")
    parser.add_argument("--evidence", required=True, help="Evidence JSONL.")
    parser.add_argument("--metrics", required=True, help="Experiment metrics CSV.")
    return add_common_args(parser).parse_args()


def safe_auc(y_true: pd.Series, scores: pd.Series) -> float:
    """Compute ROC-AUC when defined."""
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def safe_ap(y_true: pd.Series, scores: pd.Series) -> float:
    """Compute PR-AUC when defined."""
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return float("nan")


def threshold_f1(y_true: pd.Series, scores: pd.Series) -> float:
    """Compute simple thresholded F1 for one evidence source."""
    return float(f1_score(y_true, (scores >= 0.5).astype(int), zero_division=0))


def estimate_costs(examples: list[dict], evidence: pd.DataFrame) -> pd.DataFrame:
    """Estimate feature extraction costs and availability."""
    feature_names = feature_to_name()
    categories = feature_to_category()
    timings = []
    sample = examples[: min(250, len(examples))]
    for row in sample:
        start = time.perf_counter()
        features = static_features(
            row["source_code"],
            row["target_code"],
            row["source_language"],
            row["target_language"],
        )
        elapsed = time.perf_counter() - start
        static_count = max(len(features), 1)
        for feature in features:
            timings.append({"feature": feature, "seconds": elapsed / static_count, "available": not math.isnan(features[feature])})
    timing_frame = pd.DataFrame(timings)
    rows = []
    for feature, name in feature_names.items():
        if feature in timing_frame["feature"].values:
            subset = timing_frame[timing_frame["feature"] == feature]
            avg_time = float(subset["seconds"].mean())
        elif feature in evidence:
            avg_time = 0.0 if not feature.startswith("execution_") else 0.01
        else:
            avg_time = float("nan")
        if feature in evidence:
            availability = float(evidence[feature].notna().mean())
            failure_rate = float((evidence[feature].fillna(0) == 0).mean()) if feature.startswith("execution_") else 0.0
        else:
            availability = 0.0
            failure_rate = 1.0
        rows.append(
            {
                "feature": feature,
                "evidence_source": name,
                "category": categories[feature],
                "avg_time_seconds": avg_time,
                "total_time_seconds_est": avg_time * len(evidence) if not math.isnan(avg_time) else float("nan"),
                "availability": availability,
                "failure_rate": failure_rate,
                "cost_level": "high" if feature.startswith("execution_") else ("medium" if avg_time > 0.0005 else "low"),
                "output_type": "binary" if feature.startswith("execution_") or feature == "syntax_proxy" else "continuous",
            }
        )
    return pd.DataFrame(rows)


def failure_label(row: pd.Series) -> str:
    """Assign a rule-based likely failure type."""
    if row.get("target_syntax_valid", 1.0) < 0.5:
        return "syntax failure"
    if row.get("api_mismatch_score", 0.0) > 0.8:
        return "API mismatch"
    if row.get("identifier_similarity", 1.0) < 0.15 and row.get("identifier_role_similarity", 1.0) < 0.15:
        return "identifier mismatch"
    if row.get("control_flow_similarity", 1.0) < 0.35 or row.get("nesting_depth_similarity", 1.0) < 0.35:
        return "control-flow mismatch"
    if row.get("data_flow_similarity", 1.0) < 0.20:
        return "data-flow mismatch"
    if row.get("execution_passed_example", False) and not row.get("execution_passed_full", False):
        return "execution-only failure"
    if row.get("retrieval_similarity", 1.0) < 0.10 and row.get("token_jaccard", 1.0) < 0.10:
        return "possible hallucination"
    if not row.get("execution_available_full", False):
        return "insufficient evidence"
    return "semantic mismatch"


def main() -> int:
    """Run evidence analysis."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            print(yaml.safe_load(handle))
        return 0
    done_path = output_dir / "analysis_summary.json"
    if done_path.exists() and args.resume and not args.force:
        print(f"Skipping analysis because {done_path} exists.")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cost").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "failure_analysis").mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "analyze_evidence", "started", {})

    examples = read_jsonl(Path(args.examples))
    evidence = pd.DataFrame(read_jsonl(Path(args.evidence)))
    metrics = pd.read_csv(args.metrics)
    y = evidence["label"].astype(int)
    feature_names = feature_to_name()
    categories = feature_to_category()
    present_features = [feature for feature in feature_names if feature in evidence.columns]

    mi = mutual_info_classif(evidence[present_features].fillna(0.0), y, random_state=42)
    info_rows = []
    for feature, mi_value in zip(present_features, mi, strict=True):
        scores = evidence[feature].astype(float).fillna(0.0)
        info_rows.append(
            {
                "feature": feature,
                "evidence_source": feature_names[feature],
                "category": categories[feature],
                "f1_at_0_5": threshold_f1(y, scores),
                "roc_auc": safe_auc(y, scores),
                "pr_auc": safe_ap(y, scores),
                "mutual_information": float(mi_value),
                "corr_with_full_execution": float(scores.corr(evidence["execution_passed_full"].astype(float))),
            }
        )
    info_frame = pd.DataFrame(info_rows).sort_values(["mutual_information", "roc_auc"], ascending=False)
    info_frame.to_csv(output_dir / "evidence_informativeness.csv", index=False)
    write_json(output_dir / "evidence_informativeness.json", info_frame.to_dict(orient="records"))

    group_rows = []
    group_systems = [
        "lexical_only",
        "syntactic_only",
        "structural_only",
        "semantic_static_only",
        "dynamic_only",
        "static_only",
        "static_plus_example_execution",
        "static_plus_full_execution",
        "all_evidence",
    ]
    for system in group_systems:
        row = metrics[metrics["system"] == system]
        if not row.empty:
            group_rows.append(row.iloc[0].to_dict())
    groups_frame = pd.DataFrame(group_rows)
    groups_frame.to_csv(output_dir / "evidence_groups.csv", index=False)

    pairwise = metrics[metrics["system"].str.contains("_plus_")].copy()
    pairwise.to_csv(output_dir / "evidence_complementarity.csv", index=False)
    corr = evidence[present_features + ["label"]].astype(float).corr()
    corr.to_csv(output_dir / "evidence_redundancy_correlation.csv")

    cost_frame = estimate_costs(examples, evidence)
    cost_frame.to_csv(ROOT / "results" / "cost" / "evidence_costs.csv", index=False)
    write_json(ROOT / "results" / "cost" / "evidence_costs.json", cost_frame.to_dict(orient="records"))

    failed = evidence[evidence["label"] == 0].copy()
    failed["failure_type"] = failed.apply(failure_label, axis=1)
    failed[["example_id", "problem_id", "source_language", "target_language", "negative_type", "failure_type"]].to_json(
        ROOT / "results" / "failure_analysis" / "failure_cases.jsonl",
        orient="records",
        lines=True,
    )
    summary = failed["failure_type"].value_counts().rename_axis("failure_type").reset_index(name="count")
    summary["fraction"] = summary["count"] / summary["count"].sum()
    summary.to_csv(ROOT / "results" / "failure_analysis" / "failure_summary.csv", index=False)

    taxonomy = pd.DataFrame(taxonomy_rows())
    taxonomy.to_csv(output_dir / "evidence_taxonomy.csv", index=False)
    result = {
        "informativeness_rows": int(len(info_frame)),
        "group_rows": int(len(groups_frame)),
        "pairwise_rows": int(len(pairwise)),
        "cost_rows": int(len(cost_frame)),
        "failure_rows": int(len(summary)),
    }
    write_json(done_path, result)
    mark_stage(output_dir, "analyze_evidence", "completed", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
