"""Generate EviCode tables and figures from completed experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evicode.io import read_jsonl  # noqa: E402
from evicode.taxonomy import taxonomy_rows  # noqa: E402
from evicode.utils.cli import add_common_args  # noqa: E402
from evicode.utils.progress import mark_stage, write_json  # noqa: E402

DISPLAY_NAMES = {
    "syntax_only": "Syntax",
    "ast_only": "AST",
    "cfg_only": "CFG",
    "api_only": "API",
    "identifier_only": "Ident.",
    "retrieval_only": "Retr.",
    "static_only": "Static",
    "execution_example_only": "Ex.",
    "static_plus_example_execution": "Static+Ex.",
    "execution_full_only": "FullExec",
    "static_plus_full_execution": "Static+Full",
    "all_evidence": "All",
    "lexical_only": "Lexical",
    "syntactic_only": "Syntactic",
    "structural_only": "Structural",
    "semantic_static_only": "SemStatic",
    "dynamic_only": "Dynamic",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Verification examples JSONL.")
    parser.add_argument("--evidence", required=True, help="Evidence JSONL.")
    parser.add_argument("--metrics", required=True, help="Metrics CSV.")
    parser.add_argument("--statistics-dir", default=None, help="Optional directory with statistical analysis CSVs.")
    return add_common_args(parser).parse_args()


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    """Convert a frame to LaTeX."""
    body = frame.to_latex(index=False, escape=True, float_format="%.3f", na_rep="-")
    return (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\small\n"
        f"{body}"
        "\\vspace{0.5em}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def latex_table_wide(frame: pd.DataFrame, caption: str, label: str) -> str:
    """Convert a frame to a two-column LaTeX table."""
    body = frame.to_latex(index=False, escape=True, float_format="%.3f", na_rep="-")
    return (
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        "\\small\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{body}"
        "}\n"
        "\\vspace{0.5em}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table*}\n"
    )


def make_evidence_ladder_figure(path: Path) -> None:
    """Create a conceptual figure explaining evidence integration."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axis("off")
    stages = [
        ("Machine-generated code", 0.93, "#4C78A8"),
        ("Candidate program", 0.80, "#4C78A8"),
        ("Evidence extraction", 0.67, "#72B7B2"),
        ("Lexical  |  Syntax  |  Structure", 0.53, "#F58518"),
        ("API  |  Identifier  |  Data flow", 0.40, "#F58518"),
        ("Execution when available", 0.27, "#54A24B"),
        ("Trust decision + explanation", 0.12, "#B279A2"),
    ]
    for text, y, color in stages:
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "0.25"},
        )
    for (_, y1, _), (_, y2, _) in zip(stages, stages[1:], strict=False):
        ax.annotate(
            "",
            xy=(0.5, y2 + 0.045),
            xytext=(0.5, y1 - 0.045),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "0.25"},
        )
    ax.text(
        0.03,
        0.5,
        "Increasing semantic information\nand usually increasing cost",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
        color="0.25",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    """Generate paper artifacts."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        print("artifact generation dry run")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    mark_stage(output_dir, "generate_artifacts", "started", {})
    dataset = pd.DataFrame(read_jsonl(Path(args.dataset)))
    evidence = pd.DataFrame(read_jsonl(Path(args.evidence)))
    metrics = pd.read_csv(args.metrics)

    tables = ROOT / "tables"
    figures = ROOT / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    make_evidence_ladder_figure(figures / "evidence_ladder.pdf")

    dataset_stats = pd.DataFrame(
        [
            ["Languages", ", ".join(sorted(dataset["target_language"].unique()))],
            ["Problems", dataset["problem_id"].nunique()],
            ["Verification pairs", len(dataset)],
            ["Positive pairs", int((dataset["label"] == 1).sum())],
            ["Negative pairs", int((dataset["label"] == 0).sum())],
            ["Avg. source chars", int(dataset["source_code"].str.len().mean())],
            ["Avg. target chars", int(dataset["target_code"].str.len().mean())],
        ],
        columns=["Statistic", "Value"],
    )
    (tables / "dataset_statistics.tex").write_text(
        latex_table(
            dataset_stats,
            "HumanEval-X verification examples used in the main study, including directed source-target pairs and positive/negative correctness labels.",
            "tab:dataset-statistics",
        ),
        encoding="utf-8",
    )

    main_cols = ["system", "accuracy", "f1", "roc_auc", "num_features"]
    main_systems = [
        "syntax_only",
        "ast_only",
        "cfg_only",
        "api_only",
        "identifier_only",
        "retrieval_only",
        "static_only",
        "execution_example_only",
        "static_plus_example_execution",
        "execution_full_only",
        "static_plus_full_execution",
        "all_evidence",
    ]
    main_results = metrics[metrics["system"].isin(main_systems)][main_cols].copy()
    main_results["system"] = main_results["system"].map(DISPLAY_NAMES).fillna(main_results["system"])
    main_results = main_results.rename(
        columns={"system": "System", "accuracy": "Acc.", "f1": "F1", "roc_auc": "AUC", "num_features": "Feat."}
    )
    main_results = main_results.sort_values("F1", ascending=False)
    (tables / "main_results.tex").write_text(
        latex_table(
            main_results,
            "Main HumanEval-X verification results comparing static, dynamic, and fused evidence configurations.",
            "tab:main-results",
        ),
        encoding="utf-8",
    )

    ablation = metrics[metrics["system"].str.startswith("static_without_")][
        ["system", "accuracy", "f1", "roc_auc"]
    ].copy()
    ablation["removed"] = ablation["system"].str.replace("static_without_", "", regex=False)
    ablation["removed"] = ablation["removed"].replace(
        {
            "syntax_proxy": "Syntax",
            "ast_similarity": "AST",
            "control_flow_similarity": "CFG",
            "api_similarity": "API",
            "identifier_similarity": "Ident.",
            "retrieval_similarity": "Retr.",
            "length_ratio": "Length",
        }
    )
    ablation = ablation[["removed", "accuracy", "f1", "roc_auc"]]
    ablation = ablation.rename(columns={"removed": "Removed", "accuracy": "Acc.", "f1": "F1", "roc_auc": "AUC"})
    ablation = ablation.sort_values("F1", ascending=False)
    (tables / "static_ablation.tex").write_text(
        latex_table(
            ablation,
            "Leave-one-out ablation over static evidence features, measuring how much each feature family contributes to static-only verification.",
            "tab:static-ablation",
        ),
        encoding="utf-8",
    )

    stats_dir = Path(args.statistics_dir) if args.statistics_dir else ROOT / "statistics" / "humanevalx"
    ci_path = stats_dir / "bootstrap_f1.csv"
    mc_path = stats_dir / "mcnemar.csv"
    stats_tables = {}
    if ci_path.exists():
        selected = [
            "static_only",
            "execution_example_only",
            "static_plus_example_execution",
            "execution_full_only",
            "all_evidence",
        ]
        ci = pd.read_csv(ci_path)
        ci = ci[ci["system"].isin(selected)].copy()
        ci["system"] = ci["system"].map(DISPLAY_NAMES).fillna(ci["system"])
        ci["95\\% CI"] = ci.apply(lambda row: f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]", axis=1)
        ci = ci.rename(columns={"system": "System", "f1": "F1"})
        ci = ci[["System", "F1", "95\\% CI"]].sort_values("F1", ascending=False)
        (tables / "bootstrap_f1.tex").write_text(
            latex_table(
                ci,
                "Bootstrap 95\\% confidence intervals for selected HumanEval-X F1 scores, estimating uncertainty from resampled held-out predictions.",
                "tab:bootstrap-f1",
            ),
            encoding="utf-8",
        )
        stats_tables["bootstrap_f1"] = str(tables / "bootstrap_f1.tex")
    if mc_path.exists():
        mcnemar_frame = pd.read_csv(mc_path)
        selected_pairs = {
            ("static_only", "all_evidence"),
            ("execution_example_only", "all_evidence"),
            ("execution_full_only", "all_evidence"),
        }
        mcnemar_frame = mcnemar_frame[
            mcnemar_frame.apply(lambda row: (row["baseline"], row["system"]) in selected_pairs, axis=1)
        ].copy()
        mcnemar_frame["baseline"] = mcnemar_frame["baseline"].map(DISPLAY_NAMES).fillna(mcnemar_frame["baseline"])
        mcnemar_frame["system"] = mcnemar_frame["system"].map(DISPLAY_NAMES).fillna(mcnemar_frame["system"])
        mcnemar_frame = mcnemar_frame.rename(
            columns={
                "baseline": "Baseline",
                "system": "System",
                "b01": "B only",
                "b10": "S only",
                "discordant": "Disc.",
                "p_value": "$p$",
            }
        )
        mcnemar_frame = mcnemar_frame[["Baseline", "System", "B only", "S only", "Disc.", "$p$"]]
        (tables / "mcnemar.tex").write_text(
            latex_table(
                mcnemar_frame,
                "Exact McNemar tests comparing selected evidence configurations on paired held-out HumanEval-X predictions.",
                "tab:mcnemar",
            ),
            encoding="utf-8",
        )
        stats_tables["mcnemar"] = str(tables / "mcnemar.tex")

    evidence_summary = pd.DataFrame(
        [
            ["Syntax", "Parser validity", "Static"],
            ["AST", "Node similarity", "Static"],
            ["CFG", "Control proxy", "Static"],
            ["API", "API overlap", "Static"],
            ["Identifier", "Name overlap", "Static"],
            ["Retrieval", "Text similarity", "Static"],
            ["Execution", "Test pass", "Dynamic"],
        ],
        columns=["Evidence", "Signal", "Type"],
    )
    (tables / "evidence_summary.tex").write_text(
        latex_table(
            evidence_summary,
            "Implemented evidence sources in the current EviCode benchmark and the verification signal each source is intended to capture.",
            "tab:evidence-summary",
        ),
        encoding="utf-8",
    )

    taxonomy = pd.DataFrame(taxonomy_rows())
    taxonomy_table = taxonomy[["name", "category", "cost_level"]].head(12).rename(
        columns={"name": "Evidence", "category": "Category", "cost_level": "Cost"}
    )
    (tables / "evidence_taxonomy.tex").write_text(
        latex_table(
            taxonomy_table,
            "EviCode evidence taxonomy, organized by source type so that syntax, structure, API, identifier, retrieval, and execution evidence can be studied separately.",
            "tab:evidence-taxonomy",
        ),
        encoding="utf-8",
    )

    analysis_dir = ROOT / "results" / "analysis"
    cost_dir = ROOT / "results" / "cost"
    failure_dir = ROOT / "results" / "failure_analysis"
    budget_dir = ROOT / "results" / "execution_budget"
    weak_dir = ROOT / "results" / "weak_tests"
    if (analysis_dir / "evidence_informativeness.csv").exists():
        informativeness = pd.read_csv(analysis_dir / "evidence_informativeness.csv")
        info_table = informativeness.head(6)[
            ["evidence_source", "category", "roc_auc", "mutual_information"]
        ].rename(
            columns={
                "evidence_source": "Evidence",
                "category": "Category",
                "roc_auc": "AUC",
                "mutual_information": "MI",
            }
        )
        (tables / "evidence_informativeness.tex").write_text(
            latex_table(
                info_table,
                "Evidence features with the highest individual mutual information with the HumanEval-X correctness label.",
                "tab:evidence-informativeness",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(7.2, 4.0))
        plot_frame = informativeness.head(12).sort_values("mutual_information", ascending=True)
        sns.barplot(data=plot_frame, x="mutual_information", y="evidence_source", hue="category", dodge=False)
        plt.xlabel("Mutual information with label")
        plt.ylabel("")
        plt.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures / "evidence_informativeness.pdf")
        plt.close()

    if (analysis_dir / "evidence_groups.csv").exists():
        groups = pd.read_csv(analysis_dir / "evidence_groups.csv")
        groups["system"] = groups["system"].map(DISPLAY_NAMES).fillna(groups["system"])
        group_table = groups[["system", "accuracy", "f1", "roc_auc", "num_features"]].rename(
            columns={"system": "Group", "accuracy": "Acc.", "f1": "F1", "roc_auc": "AUC", "num_features": "Feat."}
        )
        (tables / "evidence_group_results.tex").write_text(
            latex_table(
                group_table,
                "Held-out HumanEval-X verification performance for individual evidence groups and fused combinations.",
                "tab:evidence-group-results",
            ),
            encoding="utf-8",
        )
    if (analysis_dir / "evidence_complementarity.csv").exists():
        comp = pd.read_csv(analysis_dir / "evidence_complementarity.csv")
        comp_table = comp[["system", "f1", "roc_auc"]].sort_values("f1", ascending=False).head(8).rename(
            columns={"system": "Combination", "f1": "F1", "roc_auc": "AUC"}
        )
        (tables / "evidence_complementarity.tex").write_text(
            latex_table(
                comp_table,
                "Best pairwise and fused evidence-group combinations on HumanEval-X, used to identify complementary sources beyond single evidence families.",
                "tab:evidence-complementarity",
            ),
            encoding="utf-8",
        )
    if (analysis_dir / "evidence_redundancy_correlation.csv").exists():
        redundancy = pd.read_csv(analysis_dir / "evidence_redundancy_correlation.csv", index_col=0)
        ordered = [
            column
            for column in [
                "token_jaccard",
                "edit_similarity",
                "length_ratio",
                "syntax_proxy",
                "ast_similarity",
                "ast_shape_similarity",
                "control_flow_similarity",
                "operator_pattern_similarity",
                "api_similarity",
                "api_mismatch_score",
                "identifier_similarity",
                "identifier_role_similarity",
                "execution_passed_example",
                "execution_passed_full",
            ]
            if column in redundancy.columns
        ]
        redundancy = redundancy.loc[ordered, ordered]
        labels = [name.replace("_similarity", "").replace("_", "\n") for name in redundancy.columns]
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        sns.heatmap(
            redundancy,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Correlation"},
        )
        clusters = [
            (0, 3, "Lexical"),
            (3, 3, "Syntax/AST"),
            (6, 2, "Structure"),
            (8, 4, "API/identifier"),
            (12, 2, "Execution"),
        ]
        for start, size, label in clusters:
            if start >= len(ordered):
                continue
            size = min(size, len(ordered) - start)
            ax.add_patch(patches.Rectangle((start, start), size, size, fill=False, edgecolor="black", linewidth=1.4))
            ax.text(start + size / 2, start - 0.28, label, ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Evidence source")
        ax.set_ylabel("Evidence source")
        ax.tick_params(axis="both", labelsize=6)
        plt.tight_layout()
        plt.savefig(figures / "evidence_complementarity_heatmap.pdf")
        plt.close()

    if (cost_dir / "evidence_costs.csv").exists():
        costs = pd.read_csv(cost_dir / "evidence_costs.csv")
        cost_source = costs.copy()
        if (analysis_dir / "evidence_informativeness.csv").exists():
            info_for_cost_table = pd.read_csv(analysis_dir / "evidence_informativeness.csv")[
                ["feature", "mutual_information"]
            ]
            cost_source = cost_source.merge(info_for_cost_table, on="feature", how="left")
            cost_source["mi_per_ms"] = cost_source["mutual_information"] / (
                cost_source["avg_time_seconds"].clip(lower=1e-9) * 1000
            )
        else:
            cost_source["mutual_information"] = pd.NA
            cost_source["mi_per_ms"] = pd.NA
        cost_table = cost_source.sort_values("mutual_information", ascending=False)[
            ["evidence_source", "avg_time_seconds", "mutual_information", "mi_per_ms", "cost_level"]
        ]
        cost_table = cost_table.rename(
            columns={
                "evidence_source": "Evidence",
                "avg_time_seconds": "Sec.",
                "mutual_information": "MI",
                "mi_per_ms": "MI/ms",
                "cost_level": "Cost",
            }
        )
        (tables / "evidence_costs.tex").write_text(
            latex_table(
                cost_table.head(6),
                "Cost-normalized evidence summary showing extraction time, informativeness, and mutual information per millisecond where label information is available.",
                "tab:evidence-costs",
            ),
            encoding="utf-8",
        )
        if (analysis_dir / "evidence_informativeness.csv").exists():
            informativeness_for_cost = pd.read_csv(analysis_dir / "evidence_informativeness.csv")
            merged = costs.merge(
                informativeness_for_cost,
                on=["feature", "evidence_source", "category"],
                how="inner",
            )
            static_value_rows = [
                (
                    "token_jaccard",
                    "Surface triage",
                    "Strong cheap ranking signal, but vulnerable to lexical mimicry.",
                ),
                (
                    "ast_similarity",
                    "Structural alignment",
                    "Informative in isolation, but weakly separates correct from incorrect candidates.",
                ),
                (
                    "operator_pattern_similarity",
                    "Local computation",
                    "Behavior-proximal evidence for arithmetic, predicate, and operator errors.",
                ),
                (
                    "syntax_proxy",
                    "Candidate validity",
                    "Cheaply removes malformed candidates before deeper verification.",
                ),
                (
                    "control_flow_similarity",
                    "Execution shape",
                    "Captures branch and loop obligations missed by lexical evidence.",
                ),
                (
                    "identifier_role_similarity",
                    "Value responsibility",
                    "Useful for explaining role swaps and suspicious data movement.",
                ),
                (
                    "api_mismatch_score",
                    "Target API misuse",
                    "Directional diagnostic signal; low values often indicate likely API failure.",
                ),
            ]
            static_value = pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "Primary value": primary_value,
                        "Design interpretation": interpretation,
                    }
                    for feature, primary_value, interpretation in static_value_rows
                ]
            )
            static_value = static_value.merge(
                informativeness_for_cost[
                    ["feature", "evidence_source", "category", "mutual_information", "roc_auc"]
                ],
                on="feature",
                how="left",
            )
            static_value = static_value.merge(
                costs[["feature", "cost_level", "availability"]],
                on="feature",
                how="left",
            )
            static_value["availability"] = static_value["availability"].map(lambda value: f"{value:.1f}")
            static_value = static_value[
                [
                    "evidence_source",
                    "category",
                    "Primary value",
                    "mutual_information",
                    "roc_auc",
                    "cost_level",
                    "availability",
                    "Design interpretation",
                ]
            ].rename(
                columns={
                    "evidence_source": "Evidence",
                    "category": "Category",
                    "mutual_information": "MI",
                    "roc_auc": "AUC",
                    "cost_level": "Cost",
                    "availability": "Avail.",
                }
            )
            (tables / "static_evidence_value.tex").write_text(
                latex_table_wide(
                    static_value,
                    "Cost-normalized interpretation of non-execution evidence. Because these sources are inexpensive in the current implementation, their scientific value comes from the semantic failure modes they expose rather than from small elapsed-time differences.",
                    "tab:static-evidence-value",
                ),
                encoding="utf-8",
            )
            plt.figure(figsize=(6.2, 4.0))
            sns.scatterplot(data=merged, x="avg_time_seconds", y="mutual_information", hue="category")
            plt.xscale("symlog", linthresh=1e-5)
            plt.xlabel("Average extraction time (s)")
            plt.ylabel("Mutual information")
            plt.tight_layout()
        plt.savefig(figures / "cost_vs_quality.pdf")
        plt.close()

    if (budget_dir / "execution_budget_results.csv").exists():
        budget = pd.read_csv(budget_dir / "execution_budget_results.csv")
        budget_table = budget[["system", "budget", "feasible", "f1"]].rename(
            columns={"system": "System", "budget": "Budget", "feasible": "Feasible", "f1": "F1"}
        )
        budget_table["Feasible"] = budget_table["Feasible"].map({True: "Yes", False: "No"})
        (tables / "execution_budget_results.tex").write_text(
            latex_table(
                budget_table.head(8),
                "Verification results for available execution budgets and status of finer-grained test-budget experiments.",
                "tab:execution-budget",
            ),
            encoding="utf-8",
        )
        feasible_budget = budget[budget["feasible"] == True].dropna(subset=["f1"]).copy()  # noqa: E712
        display = {
            "execution_only": {
                "0": "Execution only: no tests",
                "example": "Execution only: example tests",
                "full": "Execution only: full tests",
            },
            "static_only": {"0": "EviCode static evidence"},
            "static_plus_example_execution": {
                "static/example/full": "EviCode static + example tests",
            },
            "static_plus_full_execution": {
                "static/example/full": "EviCode static + full tests",
            },
            "all_evidence": {"0": "EviCode all evidence"},
        }
        feasible_budget["setting"] = feasible_budget.apply(
            lambda row: display.get(row["system"], {}).get(row["budget"], row["system"].replace("_", " ")),
            axis=1,
        )
        feasible_budget["evidence_type"] = feasible_budget["system"].map(
            lambda value: "Execution only" if value == "execution_only" else "EviCode fusion"
        )
        feasible_budget = feasible_budget.sort_values("f1", ascending=True)
        plt.figure(figsize=(6.6, 4.0))
        ax = sns.barplot(data=feasible_budget, y="setting", x="f1", hue="evidence_type", dodge=False)
        ax.bar_label(ax.containers[0], fmt="%.2f", padding=3, fontsize=8)
        if len(ax.containers) > 1:
            ax.bar_label(ax.containers[1], fmt="%.2f", padding=3, fontsize=8)
        plt.xlabel("F1")
        plt.ylabel("")
        plt.xlim(0, 1.0)
        plt.legend(title="", loc="upper right")
        plt.tight_layout()
        plt.savefig(figures / "execution_budget_curve.pdf")
        plt.close()

    if (weak_dir / "weak_test_results.csv").exists():
        weak = pd.read_csv(weak_dir / "weak_test_results.csv")
        weak_table = weak[["retention", "system", "feasible", "mean_f1"]].rename(
            columns={"retention": "Retention", "system": "System", "feasible": "Feasible", "mean_f1": "F1"}
        )
        weak_table["Feasible"] = weak_table["Feasible"].map({True: "Yes", False: "No"})
        (tables / "weak_test_results.tex").write_text(
            latex_table(
                weak_table.head(8),
                "Weak-test robustness status, separating completed static baselines from dynamic retention experiments that require normalized individual tests.",
                "tab:weak-tests",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(6.4, 3.8))
        feasible_weak = weak[weak["feasible"] == True].copy()  # noqa: E712
        sns.lineplot(data=feasible_weak, x="retention", y="mean_f1", hue="system", marker="o")
        plt.xlabel("Retained test fraction")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(figures / "weak_test_robustness.pdf")
        plt.close()

    if (failure_dir / "failure_summary.csv").exists():
        failures = pd.read_csv(failure_dir / "failure_summary.csv")
        failure_table = failures.rename(
            columns={"failure_type": "Failure type", "count": "Count", "fraction": "Fraction"}
        )
        (tables / "failure_summary.tex").write_text(
            latex_table(
                failure_table,
                "Frequency of rule-based failure explanations assigned to negative examples in the current HumanEval-X benchmark.",
                "tab:failure-summary",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(6.6, 3.8))
        sns.barplot(data=failures.sort_values("count", ascending=True), x="count", y="failure_type", color="#F58518")
        plt.xlabel("Examples")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figures / "failure_type_distribution.pdf")
        plt.close()

    phase2_dir = ROOT / "results" / "phase2"
    if (phase2_dir / "metric_comparison.csv").exists():
        metric_comp = pd.read_csv(phase2_dir / "metric_comparison.csv")
        fusion_rows = metrics[metrics["system"].isin(["static_only", "all_evidence"])]
        if not fusion_rows.empty:
            evicode_rows = fusion_rows.assign(
                metric=fusion_rows["system"].replace(
                    {
                        "static_only": "EviCode Static",
                        "all_evidence": "EviCode All",
                    }
                ),
                pr_auc=float("nan"),
            )[["metric", "accuracy", "f1", "roc_auc", "pr_auc"]]
            metric_comp = pd.concat([metric_comp, evicode_rows], ignore_index=True)
        metric_comp["metric"] = metric_comp["metric"].replace(
            {
                "CrystalBLEU-proxy": "Crystal",
                "CodeBLEU-proxy": "CodeBLEU",
                "TF-IDF embedding similarity": "TF-IDF",
                "Execution accuracy": "Exec.",
            }
        )
        metric_table = metric_comp.rename(
            columns={"metric": "Metric", "f1": "F1", "roc_auc": "AUC", "pr_auc": "PR-AUC"}
        )
        metric_table = metric_table[["Metric", "F1", "AUC", "PR-AUC"]]
        metric_table["PR-AUC"] = metric_table["PR-AUC"].map(lambda value: "-" if pd.isna(value) else value)
        (tables / "metric_comparison.tex").write_text(
            latex_table(
                metric_table,
                "Comparison of lightweight similarity metrics, execution, and explicit EviCode evidence-fusion configurations on the same HumanEval-X verification split.",
                "tab:metric-comparison",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(6.4, 3.8))
        sns.barplot(data=metric_comp.sort_values("roc_auc"), x="roc_auc", y="metric", color="#54A24B")
        plt.xlabel("ROC-AUC")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figures / "metric_comparison_matrix.pdf")
        plt.close()

    if (phase2_dir / "feature_importance.csv").exists():
        importance = pd.read_csv(phase2_dir / "feature_importance.csv")
        imp = importance.set_index("feature")
        metric_comp_path = phase2_dir / "metric_comparison.csv"
        metric_comp_raw = pd.read_csv(metric_comp_path) if metric_comp_path.exists() else pd.DataFrame()
        example_f1 = float(metrics.loc[metrics["system"] == "execution_example_only", "f1"].iloc[0])
        full_f1 = float(metrics.loc[metrics["system"] == "execution_full_only", "f1"].iloc[0])
        all_auc = float(metrics.loc[metrics["system"] == "all_evidence", "roc_auc"].iloc[0])
        full_auc = float(metrics.loc[metrics["system"] == "execution_full_only", "roc_auc"].iloc[0])
        codebleu_f1 = float(metric_comp_raw.loc[metric_comp_raw["metric"] == "CodeBLEU-proxy", "f1"].iloc[0])
        evicode_static_auc = float(metrics.loc[metrics["system"] == "static_only", "roc_auc"].iloc[0])
        findings = pd.DataFrame(
            [
                [
                    "AST is weak after richer evidence",
                    f"AST node coefficient {imp.loc['ast_similarity', 'abs_importance']:.3f}; operator-pattern coefficient {imp.loc['operator_pattern_similarity', 'abs_importance']:.3f}.",
                    "Tree shape can preserve the wrong computation.",
                ],
                [
                    "Roles beat raw names",
                    f"Role coefficient {imp.loc['identifier_role_similarity', 'abs_importance']:.3f}; raw-name coefficient {imp.loc['identifier_similarity', 'abs_importance']:.3f}.",
                    "Names help most when they encode data roles.",
                ],
                [
                    "Example tests are strong",
                    f"Example F1 {example_f1:.3f}; full-execution F1 {full_f1:.3f}; ratio {example_f1 / full_f1:.2f}.",
                    "A small dynamic signal has high early payoff.",
                ],
                [
                    "Fusion refines ranking",
                    f"All-evidence AUC {all_auc:.3f}; full-execution AUC {full_auc:.3f}.",
                    "Static evidence can refine confidence.",
                ],
                [
                    "CodeBLEU is incomplete",
                    f"CodeBLEU F1 {codebleu_f1:.3f}; EviCode static AUC {evicode_static_auc:.3f}.",
                    "Structure-aware similarity still misses behavior.",
                ],
            ],
            columns=["Finding", "Result", "Lesson"],
        )
        (tables / "surprising_findings.tex").write_text(
            latex_table_wide(
                findings,
                "Interpretive findings that turn aggregate results into scientific lessons about verification evidence.",
                "tab:surprising-findings",
            ),
            encoding="utf-8",
        )

    metric_scores_path = phase2_dir / "metric_scores.csv"
    if metric_scores_path.exists():
        scores = pd.read_csv(metric_scores_path)
        evidence_subset = evidence[
            [
                "example_id",
                "label",
                "ast_similarity",
                "operator_pattern_similarity",
                "identifier_similarity",
                "identifier_role_similarity",
                "api_similarity",
                "execution_passed_example",
                "execution_passed_full",
            ]
        ]
        joined = scores.merge(evidence_subset, on=["example_id", "label"], how="left")
        cases = []

        def add_case(pattern: str, frame: pd.DataFrame, reason: str, sort_col: str, ascending: bool = False) -> None:
            if frame.empty:
                return
            row = frame.sort_values(sort_col, ascending=ascending).iloc[0]
            cases.append(
                {
                    "Pattern": pattern,
                    "Example": row["example_id"],
                    "BLEU": row["bleu"],
                    "CodeBLEU": row["codebleu_proxy"],
                    "AST": row["ast_similarity"],
                    "Exec.": "pass" if bool(row["execution_passed_full"]) else "fail",
                    "Reason": reason,
                }
            )

        add_case(
            "High structure, wrong behavior",
            joined[(joined["label"] == 0) & (joined["ast_similarity"] > 0.80) & (joined["execution_accuracy"] == 0)],
            "High tree similarity can hide wrong behavior.",
            "ast_similarity",
        )
        add_case(
            "High CodeBLEU, execution fail",
            joined[(joined["label"] == 0) & (joined["codebleu_proxy"] > 0.50) & (joined["execution_accuracy"] == 0)],
            "Code-aware similarity cannot observe failing tests.",
            "codebleu_proxy",
        )
        add_case(
            "Low lexical, execution pass",
            joined[(joined["label"] == 1) & (joined["bleu"] < 0.002) & (joined["execution_accuracy"] == 1)],
            "Correct translations can use different tokens.",
            "bleu",
            ascending=True,
        )
        add_case(
            "Role signal beats raw names",
            joined[
                (joined["label"] == 1)
                & (joined["identifier_similarity"] < 0.20)
                & (joined["identifier_role_similarity"] > joined["identifier_similarity"])
            ],
            "Role consistency can survive renaming.",
            "identifier_role_similarity",
        )
        add_case(
            "Example pass, full fail",
            joined[
                (joined["execution_passed_example"] == True)  # noqa: E712
                & (joined["execution_passed_full"] == False)  # noqa: E712
            ],
            "Partial tests can miss full-suite behavior.",
            "codebleu_proxy",
        )
        if cases:
            qualitative = pd.DataFrame(cases).head(5)
            (tables / "qualitative_disagreements.tex").write_text(
                latex_table_wide(
                    qualitative,
                    "Qualitative disagreement cases illustrating why average metric scores hide important evidence conflicts.",
                    "tab:qualitative-disagreements",
                ),
                encoding="utf-8",
            )

    if (phase2_dir / "evidence_hierarchy.csv").exists():
        hierarchy = pd.read_csv(phase2_dir / "evidence_hierarchy.csv")
        hierarchy_table = hierarchy[["level", "num_features", "f1", "roc_auc", "incremental_f1"]].rename(
            columns={"level": "Level", "num_features": "Feat.", "f1": "F1", "roc_auc": "AUC", "incremental_f1": "Delta F1"}
        )
        (tables / "evidence_hierarchy.tex").write_text(
            latex_table(
                hierarchy_table,
                "Incremental predictive value as evidence groups are added from lexical/static features toward dynamic execution evidence.",
                "tab:evidence-hierarchy",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(6.4, 3.8))
        sns.lineplot(data=hierarchy, x="level", y="f1", marker="o")
        plt.xticks(rotation=20, ha="right")
        plt.xlabel("Evidence hierarchy")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(figures / "information_gain_hierarchy.pdf")
        plt.close()

    if (phase2_dir / "feature_importance.csv").exists():
        importance = pd.read_csv(phase2_dir / "feature_importance.csv")
        imp_table = importance.head(8)[["evidence_source", "category", "abs_importance"]].rename(
            columns={"evidence_source": "Evidence", "category": "Category", "abs_importance": "Importance"}
        )
        (tables / "evidence_importance.tex").write_text(
            latex_table(
                imp_table,
                "Most influential evidence features in the transparent logistic-fusion model, ranked by coefficient magnitude.",
                "tab:evidence-importance",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(6.6, 3.8))
        sns.barplot(data=importance.head(12).sort_values("abs_importance"), x="abs_importance", y="evidence_source", hue="category", dodge=False)
        plt.xlabel("Absolute standardized coefficient")
        plt.ylabel("")
        plt.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures / "marginal_contribution_sources.pdf")
        plt.close()

    if (phase2_dir / "language_specific_importance.csv").exists():
        language = pd.read_csv(phase2_dir / "language_specific_importance.csv")
        language_top = language.sort_values("mutual_information", ascending=False).groupby("target_language").head(3)
        language_table = language_top[["target_language", "evidence_source", "category", "mutual_information"]].rename(
            columns={
                "target_language": "Lang.",
                "evidence_source": "Evidence",
                "category": "Category",
                "mutual_information": "MI",
            }
        )
        (tables / "language_specific_rankings.tex").write_text(
            latex_table(
                language_table,
                "Highest-ranked evidence features within each target language, showing language-specific differences in verification signal.",
                "tab:language-rankings",
            ),
            encoding="utf-8",
        )
        plt.figure(figsize=(7.0, 4.0))
        sns.barplot(data=language_top, x="mutual_information", y="evidence_source", hue="target_language")
        plt.xlabel("Mutual information")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figures / "language_specific_importance.pdf")
        plt.close()

    if (phase2_dir / "failure_detection_matrix.csv").exists():
        failure_matrix = pd.read_csv(phase2_dir / "failure_detection_matrix.csv")
        failure_table = failure_matrix[
            ["failure_type", "syntax_coverage", "api_coverage", "data_flow_coverage", "execution_coverage"]
        ].rename(
            columns={
                "failure_type": "Failure",
                "syntax_coverage": "Syn.",
                "api_coverage": "API",
                "data_flow_coverage": "DF",
                "execution_coverage": "Exec.",
            }
        )
        (tables / "failure_detection_matrix.tex").write_text(
            latex_table(
                failure_table.head(8),
                "Mapping from rule-based failure types to the evidence detectors that can flag each failure mode.",
                "tab:failure-detection",
            ),
            encoding="utf-8",
        )
        matrix = failure_matrix.set_index("failure_type").drop(columns=["count"])
        plt.figure(figsize=(7.0, 4.2))
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1)
        plt.xlabel("Evidence detector")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figures / "failure_detection_matrix.pdf")
        plt.close()

    if (phase2_dir / "cost_information_pareto.csv").exists():
        pareto = pd.read_csv(phase2_dir / "cost_information_pareto.csv")
        pareto["mi_per_ms"] = pareto["mutual_information"] / (pareto["avg_time_seconds"].clip(lower=1e-9) * 1000)
        plot_frame = pareto.sort_values("mutual_information", ascending=False).head(10).copy()
        plot_frame["short_name"] = plot_frame["evidence_source"].replace(
            {
                "Full execution": "Full exec.",
                "Example execution": "Example exec.",
                "Token overlap": "Token",
                "AST node similarity": "AST",
                "Condition/operator patterns": "Operators",
                "Identifier overlap": "Identifiers",
                "Parse validity": "Parse",
                "Control flow": "Control flow",
                "Length ratio": "Length",
                "Identifier role consistency": "Roles",
            }
        )
        plot_frame = plot_frame.sort_values("mutual_information", ascending=True)
        colors = plot_frame["category"].map(
            {
                "Dynamic": "#2ca25f",
                "Lexical": "#4C78A8",
                "Syntactic": "#72B7B2",
                "Structural": "#F58518",
                "Semantic-static": "#B279A2",
            }
        ).fillna("#666666")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
        y_pos = range(len(plot_frame))
        axes[0].barh(y_pos, plot_frame["mutual_information"], color=colors, alpha=0.88)
        axes[0].set_yticks(list(y_pos))
        axes[0].set_yticklabels(plot_frame["short_name"], fontsize=8)
        axes[0].set_xlabel("Mutual information")
        axes[0].set_title("Information")
        axes[0].grid(axis="x", alpha=0.25)

        axes[1].barh(y_pos, plot_frame["mi_per_ms"], color=colors, alpha=0.88)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("MI per millisecond (log)")
        axes[1].set_title("Cost-normalized value")
        axes[1].grid(axis="x", alpha=0.25)
        axes[1].tick_params(axis="y", left=False, labelleft=False)

        legend_items = [
            patches.Patch(facecolor=color, label=label)
            for label, color in [
                ("Dynamic", "#2ca25f"),
                ("Lexical", "#4C78A8"),
                ("Syntactic", "#72B7B2"),
                ("Structural", "#F58518"),
                ("Semantic-static", "#B279A2"),
            ]
            if label in set(plot_frame["category"])
        ]
        fig.legend(
            handles=legend_items,
            loc="lower center",
            ncol=min(len(legend_items), 5),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.suptitle("Evidence information and cost-normalized value", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=(0, 0.06, 1, 0.94))
        plt.savefig(figures / "cost_information_pareto.pdf")
        plt.close()

    if (phase2_dir / "metric_design_guidelines.csv").exists():
        guidelines = pd.read_csv(phase2_dir / "metric_design_guidelines.csv")
        guidelines_table = pd.DataFrame(
            {
                "No.": range(1, len(guidelines) + 1),
                "Guideline": [
                    "Use execution when available",
                    "Combine static evidence",
                    "Treat examples as incomplete",
                    "Report cost and availability",
                    "Avoid lexical-only claims",
                ][: len(guidelines)],
            }
        )
        (tables / "metric_design_guidelines.tex").write_text(
            latex_table(
                guidelines_table,
                "Evidence-backed guidelines for designing verification metrics that report signal type, cost, availability, and failure coverage.",
                "tab:metric-guidelines",
            ),
            encoding="utf-8",
        )

    llm_dir = ROOT / "results" / "llm_predictions"
    if (llm_dir / "llm_model_calibration.csv").exists():
        llm_models = pd.read_csv(llm_dir / "llm_model_calibration.csv")
        llm_table = llm_models[
            ["model_name", "rows", "empirical_accuracy", "mean_confidence", "roc_auc", "brier"]
        ].rename(
            columns={
                "model_name": "Model",
                "rows": "Rows",
                "empirical_accuracy": "Acc.",
                "mean_confidence": "Conf.",
                "roc_auc": "AUC",
                "brier": "Brier",
            }
        )
        (tables / "llm_external_calibration.tex").write_text(
            latex_table(
                llm_table,
                "Per-model confidence and calibration results when the HumanEval-X-trained static verifier is evaluated on external Python-to-Java LLM translations.",
                "tab:llm-calibration",
            ),
            encoding="utf-8",
        )
    if (llm_dir / "confidence_by_score.csv").exists():
        score_frame = pd.read_csv(llm_dir / "confidence_by_score.csv")
        score_table = score_frame[["score", "failure_grade", "rows", "mean_confidence"]].rename(
            columns={"score": "Score", "failure_grade": "Grade", "rows": "Rows", "mean_confidence": "Conf."}
        )
        score_table["Grade"] = score_table["Grade"].replace(
            {
                "no_clean_candidate_or_unparsable": "unusable",
                "parsable_not_compilable": "parse-only",
                "compilable_functionally_incorrect": "compile-wrong",
                "functionally_correct": "correct",
            }
        )
        (tables / "llm_confidence_by_score.tex").write_text(
            latex_table(
                score_table,
                "Mean EviCode confidence for each graded outcome in the external Python-to-Java LLM prediction dataset.",
                "tab:llm-score-confidence",
            ),
            encoding="utf-8",
        )
    if (llm_dir / "calibration_bins.csv").exists():
        calibration = pd.read_csv(llm_dir / "calibration_bins.csv")
        plt.figure(figsize=(5.2, 4.0))
        plt.plot([0, 1], [0, 1], "--", color="0.5", label="perfect")
        sns.lineplot(data=calibration.dropna(), x="mean_confidence", y="accuracy", marker="o", label="EviCode")
        plt.xlabel("Mean confidence")
        plt.ylabel("Empirical accuracy")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(figures / "llm_calibration_curve.pdf")
        plt.close()
    if (llm_dir / "confidence_by_score.csv").exists():
        score_frame = pd.read_csv(llm_dir / "confidence_by_score.csv")
        plt.figure(figsize=(5.8, 3.8))
        sns.barplot(data=score_frame, x="score", y="mean_confidence", color="#B279A2")
        plt.xlabel("LLM translation score")
        plt.ylabel("Mean verifier confidence")
        plt.tight_layout()
        plt.savefig(figures / "llm_confidence_by_score.pdf")
        plt.close()

    plt.figure(figsize=(7.0, 4.0))
    ax = plt.gca()
    ax.axis("off")
    y_pos = 0.92
    for category, group in taxonomy.groupby("category", sort=False):
        ax.text(0.02, y_pos, category, weight="bold", fontsize=10, va="top")
        ax.text(0.30, y_pos, ", ".join(group["name"].head(5)), fontsize=8, va="top")
        y_pos -= 0.16
    plt.tight_layout()
    plt.savefig(figures / "evidence_taxonomy.pdf")
    plt.close()

    plt.figure(figsize=(7.0, 3.6))
    ax = plt.gca()
    ax.axis("off")
    report_text = (
        "Example evidence report\\n"
        "Syntax: valid | AST: moderate | API: weak\\n"
        "Control flow: suspicious | Execution: failed full tests\\n"
        "Likely failure: semantic or control-flow mismatch"
    )
    ax.text(0.03, 0.82, report_text, family="monospace", fontsize=11, va="top")
    plt.tight_layout()
    plt.savefig(figures / "qualitative_evidence_report.pdf")
    plt.close()

    plt.figure(figsize=(7.2, 4.0))
    plot_frame = main_results.sort_values("F1", ascending=True)
    sns.barplot(data=plot_frame, x="F1", y="System", color="#4C78A8")
    plt.xlabel("F1")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures / "main_results_f1.pdf")
    plt.close()

    evidence_cols = [
        "ast_similarity",
        "control_flow_similarity",
        "api_similarity",
        "identifier_similarity",
        "retrieval_similarity",
        "execution_passed_example",
        "execution_passed_full",
        "label",
    ]
    corr = evidence[evidence_cols].astype(float).corr()
    plt.figure(figsize=(6.8, 5.8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    plt.tight_layout()
    plt.savefig(figures / "evidence_correlation.pdf")
    plt.close()

    summary = {
        "dataset_rows": int(len(dataset)),
        "evidence_rows": int(len(evidence)),
        "metrics_rows": int(len(metrics)),
        "best_system": str(main_results.iloc[0]["System"]),
        "best_f1": float(main_results.iloc[0]["F1"]),
        "statistics_tables": stats_tables,
    }
    write_json(output_dir / "artifact_summary.json", summary)
    mark_stage(output_dir, "generate_artifacts", "completed", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
