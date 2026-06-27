"""Generate paper assets and optionally compile the EviCode manuscript."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Skip valid existing outputs.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing outputs.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML config path.")
    parser.add_argument("--output-dir", default="paper/output", help="Paper output directory.")
    return parser.parse_args()


def should_skip(path: Path, resume: bool, force: bool) -> bool:
    """Return whether an existing output should be reused."""
    return path.exists() and resume and not force and path.stat().st_size > 0


def write_text(path: Path, content: str, resume: bool, force: bool) -> None:
    """Write text unless a valid resumable output exists."""
    if should_skip(path, resume, force):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate_pipeline_figure(resume: bool, force: bool) -> None:
    """Generate the EviCode pipeline figure from code."""
    path = FIGURES / "pipeline.pdf"
    if should_skip(path, resume, force):
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.axis("off")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    blue = "#245ea8"
    purple = "#7651a8"
    green = "#23824f"
    orange = "#f39c12"
    red = "#d6452f"
    gray = "#5b6470"

    def rounded_box(x_coord: float, y_coord: float, width: float, height: float, color: str, alpha: float = 0.08):
        box = patches.FancyBboxPatch(
            (x_coord, y_coord),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(box)
        return box

    def header(x_coord: float, y_coord: float, width: float, text: str, color: str) -> None:
        rounded_box(x_coord, y_coord, width, 0.055, color, alpha=0.98)
        ax.text(
            x_coord + width / 2,
            y_coord + 0.028,
            text,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    def arrow(x_start: float, y_start: float, x_end: float, y_end: float, color: str) -> None:
        ax.add_patch(
            patches.FancyArrowPatch(
                (x_start, y_start),
                (x_end, y_end),
                arrowstyle="simple",
                mutation_scale=20,
                linewidth=0,
                color=color,
                alpha=0.95,
            )
        )

    def tick(x_coord: float, y_coord: float, color: str = green) -> None:
        ax.text(x_coord, y_coord, r"$\checkmark$", ha="center", va="center", fontsize=14, color=color, fontweight="bold")

    def cross(x_coord: float, y_coord: float, color: str = red) -> None:
        ax.text(x_coord, y_coord, r"$\times$", ha="center", va="center", fontsize=13, color=color, fontweight="bold")

    ax.text(
        0.5,
        0.975,
        "EviCode: Evidence-Grounded Semantic Verification",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#0f1f4d",
    )
    ax.text(
        0.5,
        0.93,
        "Structured static and dynamic signals are extracted, fused, calibrated, and explained",
        ha="center",
        va="top",
        fontsize=11,
        color=gray,
        style="italic",
    )

    # Input column.
    rounded_box(0.02, 0.19, 0.18, 0.67, blue)
    header(0.02, 0.82, 0.18, "1. INPUT PROGRAM PAIR", blue)
    ax.text(0.11, 0.77, "Reference x", ha="center", fontsize=10, fontweight="bold")
    rounded_box(0.045, 0.665, 0.13, 0.08, blue, alpha=0.04)
    ax.text(0.11, 0.707, "def add(a, b):\n    return a + b", ha="center", va="center", fontsize=9, family="monospace")
    ax.text(0.11, 0.60, "Candidate y", ha="center", fontsize=10, fontweight="bold")
    rounded_box(0.045, 0.48, 0.13, 0.09, blue, alpha=0.04)
    ax.text(0.11, 0.525, "int add(int a,b) {\n    return a - b;\n}", ha="center", va="center", fontsize=8.5, family="monospace")
    ax.text(0.11, 0.405, "Optional tests T", ha="center", fontsize=10, fontweight="bold")
    rounded_box(0.058, 0.30, 0.105, 0.075, blue, alpha=0.04)
    ax.text(0.07, 0.34, r"$\checkmark$", fontsize=12, color=green)
    ax.text(0.09, 0.34, "example", fontsize=8.5)
    ax.text(0.07, 0.315, r"$\times$", fontsize=11, color=red)
    ax.text(0.09, 0.315, "boundary", fontsize=8.5)
    ax.text(0.11, 0.235, "language pair  |  task id  |  budget", ha="center", fontsize=8.5, color=gray)

    # Evidence extraction column.
    rounded_box(0.25, 0.12, 0.31, 0.74, purple)
    header(0.25, 0.82, 0.31, "2. EVIDENCE EXTRACTION", purple)
    ax.text(0.405, 0.785, "Heterogeneous evidence sources", ha="center", fontsize=9.5, color="#2f2154")
    evidence_rows = [
        ("Lexical", "token overlap", [True, True, True]),
        ("Syntax", "parse validity / AST", [True, True, False]),
        ("Structure", "CFG and operators", [True, False, True]),
        ("API", "calls and library use", [True, False, False]),
        ("Identifiers", "roles and consistency", [True, True, True]),
        ("Data flow", "def-use relations", [True, False, True]),
        ("Type", "type consistency", [True, True, False]),
        ("Retrieval", "nearest solved tasks", [True, True, True]),
        ("Execution", "examples / full tests", [True, False, False]),
    ]
    y = 0.745
    for idx, (name, detail, vals) in enumerate(evidence_rows, start=1):
        rounded_box(0.265, y - 0.048, 0.255, 0.045, purple, alpha=0.035)
        ax.text(0.278, y - 0.025, name, ha="left", va="center", fontsize=9.2, fontweight="bold")
        ax.text(0.365, y - 0.025, detail, ha="left", va="center", fontsize=8.2, color=gray)
        for col, val in enumerate(vals):
            (tick if val else cross)(0.468 + col * 0.019, y - 0.025)
        ax.text(0.535, y - 0.025, f"$e_{idx}$", ha="center", va="center", fontsize=11)
        y -= 0.063
    rounded_box(0.29, 0.155, 0.22, 0.045, purple, alpha=0.025)
    ax.text(0.40, 0.178, r"$E(x,y)=[e_1,\ldots,e_9]\in\mathbb{R}^{9}$", ha="center", va="center", fontsize=11)

    # Fusion column.
    rounded_box(0.60, 0.12, 0.18, 0.74, green)
    header(0.60, 0.82, 0.18, "3. EVIDENCE FUSION", green)
    ax.text(0.69, 0.775, "cost-aware logistic fusion", ha="center", fontsize=9.2)
    rounded_box(0.625, 0.67, 0.13, 0.06, green, alpha=0.04)
    ax.text(0.69, 0.70, r"$w_1e_1+\cdots+w_9e_9+b$", ha="center", va="center", fontsize=10.5)
    arrow(0.69, 0.66, 0.69, 0.59, green)
    rounded_box(0.625, 0.50, 0.13, 0.08, green, alpha=0.04)
    ax.text(0.69, 0.545, r"$P(correct\mid x,y)=\sigma(z)$", ha="center", va="center", fontsize=10.5)
    arrow(0.69, 0.49, 0.69, 0.42, green)
    rounded_box(0.625, 0.32, 0.13, 0.08, green, alpha=0.04)
    ax.text(0.69, 0.36, "calibration\nand uncertainty", ha="center", va="center", fontsize=9.2)
    arrow(0.69, 0.31, 0.69, 0.24, green)
    rounded_box(0.625, 0.19, 0.13, 0.055, green, alpha=0.04)
    ax.text(0.69, 0.218, "confidence score", ha="center", va="center", fontsize=9.2)

    # Output column.
    rounded_box(0.83, 0.19, 0.15, 0.67, blue)
    header(0.83, 0.82, 0.15, "4. DECISION", blue)
    ax.text(0.905, 0.75, "Verification score", ha="center", fontsize=10, fontweight="bold", color=blue)
    ax.add_patch(patches.Wedge((0.905, 0.66), 0.065, 0, 180, width=0.017, facecolor="#d9d9d9", edgecolor="none"))
    for start, end, color in [(0, 50, green), (50, 120, "#d6c400"), (120, 180, red)]:
        ax.add_patch(patches.Wedge((0.905, 0.66), 0.065, start, end, width=0.017, facecolor=color, edgecolor="none"))
    ax.plot([0.905, 0.955], [0.66, 0.70], color="#1f2933", linewidth=2.2)
    ax.text(0.905, 0.585, "0.87", ha="center", fontsize=18, fontweight="bold")
    for y_dec, color, label in [(0.51, green, "Accept"), (0.455, orange, "Review"), (0.40, red, "Reject")]:
        ax.scatter([0.855], [y_dec], s=170, color=color)
        ax.text(0.855, y_dec - 0.001, r"$\checkmark$" if label == "Accept" else ("?" if label == "Review" else r"$\times$"), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(0.877, y_dec, label, ha="left", va="center", fontsize=9.5)
    ax.text(0.905, 0.335, "Top evidence", ha="center", fontsize=10, fontweight="bold", color=blue)
    bars = [("Execution", 0.09, green), ("API", 0.065, green), ("Identifiers", 0.045, green), ("AST", 0.035, red)]
    for i, (label, width, color) in enumerate(bars):
        yy = 0.295 - i * 0.038
        ax.text(0.845, yy, label, ha="left", va="center", fontsize=8)
        ax.add_patch(patches.Rectangle((0.905, yy - 0.009), width, 0.017, facecolor=color, alpha=0.85))

    # Analyses band.
    rounded_box(0.02, 0.025, 0.76, 0.105, "#7aa6d9", alpha=0.10)
    header(0.02, 0.115, 0.76, "5. ANALYSES ENABLED", blue)
    analyses = [
        ("Informative\npower", r"$F_1$/AUC/MI"),
        ("Complementarity", r"$\Delta F_1$"),
        ("Redundancy", "correlation"),
        ("Cost", "time/budget"),
        ("Calibration", "reliability"),
        ("Failures", "case studies"),
    ]
    for i, (label, sub) in enumerate(analyses):
        x = 0.055 + i * 0.118
        rounded_box(x, 0.04, 0.095, 0.055, "#7aa6d9", alpha=0.08)
        ax.text(x + 0.0475, 0.071, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + 0.0475, 0.049, sub, ha="center", va="center", fontsize=7.5, color=gray)

    rounded_box(0.81, 0.025, 0.17, 0.105, green, alpha=0.08)
    header(0.81, 0.115, 0.17, "6. OUTCOMES", blue)
    outcomes = ["Interpret evidence", "Route candidates", "Use tests wisely", "Design better metrics"]
    for i, text in enumerate(outcomes):
        yy = 0.095 - i * 0.02
        tick(0.83, yy, green)
        ax.text(0.85, yy, text, ha="left", va="center", fontsize=8.5)

    arrow(0.205, 0.525, 0.245, 0.525, blue)
    arrow(0.565, 0.525, 0.595, 0.525, purple)
    arrow(0.785, 0.525, 0.825, 0.525, green)

    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_smoke_table(resume: bool, force: bool) -> None:
    """Generate a small table from the synthetic smoke output if available."""
    summary_path = ROOT / "experiments" / "smoke" / "summary.json"
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    frame = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": f"{summary['accuracy']:.3f}"},
            {"Metric": "ROC-AUC", "Value": f"{summary['roc_auc']:.3f}"},
            {"Metric": "Examples", "Value": str(summary["num_examples"])},
            {"Metric": "Evidence sources", "Value": str(summary["num_evidence_sources"])},
        ]
    )
    table = frame.to_latex(index=False, escape=True, caption="Synthetic smoke-test output.", label="tab:smoke")
    write_text(TABLES / "smoke_summary.tex", table, resume=resume, force=force)


def generate_dataset_placeholder(resume: bool, force: bool) -> None:
    """Generate a placeholder dataset table when no benchmark artifact is present."""
    benchmark_outputs = [
        ROOT / "datasets" / "processed" / "humanevalx" / "verification_examples.jsonl",
        ROOT / "experiments" / "humanevalx" / "fusion_ts" / "metrics.csv",
    ]
    table_path = TABLES / "dataset_statistics.tex"
    if table_path.exists() and any(path.exists() for path in benchmark_outputs):
        return
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Dataset status when no benchmark artifact is present.}\n"
        "\\label{tab:dataset-status}\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "Item & Status \\\\\n"
        "\\midrule\n"
        "HumanEval-X & Planned \\\\\n"
        "MultiPL-E & Planned \\\\\n"
        "xCodeEval & Optional \\\\\n"
        "Project CodeNet & Optional \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write_text(table_path, content, resume=resume, force=force)


def check_manuscript() -> list[str]:
    """Run lightweight manuscript quality checks."""
    problems: list[str] = []
    required = ["main.tex", "references.bib"]
    for name in required:
        path = PAPER / name
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"Missing or empty paper file: {name}")
    banned = ["TODO", "FIXME", "XXX"]
    for tex_path in PAPER.glob("*.tex"):
        text = tex_path.read_text(encoding="utf-8")
        forbidden_inputs = [
            "\\input{abstract}",
            "\\input{introduction}",
            "\\input{background}",
            "\\input{problem_definition}",
            "\\input{methodology}",
            "\\input{architecture}",
            "\\input{datasets}",
            "\\input{implementation}",
            "\\input{experimental_setup}",
            "\\input{evaluation_metrics}",
            "\\input{results}",
            "\\input{ablation}",
            "\\input{statistical_validation}",
            "\\input{discussion}",
            "\\input{limitations}",
            "\\input{threats_to_validity}",
            "\\input{future_work}",
            "\\input{conclusion}",
            "\\input{appendix}",
        ]
        if tex_path.name == "main.tex":
            for forbidden in forbidden_inputs:
                if forbidden in text:
                    problems.append(
                        "main.tex should be self-contained; remove section-level "
                        f"{forbidden} calls."
                    )
        for marker in banned:
            if marker in text:
                problems.append(f"{tex_path.name} contains marker {marker}")
    return problems


def compile_latex(output_dir: Path) -> dict[str, str]:
    """Compile LaTeX if a compiler is available."""
    compiler = shutil.which("tectonic") or shutil.which("pdflatex")
    if compiler is None:
        return {"status": "skipped", "reason": "No tectonic or pdflatex executable found on PATH."}
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(compiler).name.lower().startswith("tectonic"):
        cmd = [compiler, "main.tex", "--outdir", str(output_dir)]
        runs = 1
    else:
        cmd = [compiler, "-interaction=nonstopmode", "-output-directory", str(output_dir), "main.tex"]
        runs = 1
    for _ in range(runs):
        subprocess.run(cmd, cwd=PAPER, check=True)
    if not Path(compiler).name.lower().startswith("tectonic") and (PAPER / "references.bib").exists():
        bibtex = shutil.which("bibtex")
        aux_path = output_dir / "main.aux"
        if bibtex and aux_path.exists():
            subprocess.run([bibtex, str(aux_path)], cwd=PAPER, check=False)
            for _ in range(2):
                subprocess.run(cmd, cwd=PAPER, check=True)
    pdf = output_dir / "main.pdf"
    if pdf.exists():
        target = output_dir / "paper.pdf"
        if target.exists():
            target.unlink()
        pdf.rename(target)
    return {"status": "completed", "compiler": compiler}


def main() -> int:
    """Build paper assets and optionally compile the manuscript."""
    args = parse_args()
    output_dir = ROOT / args.output_dir
    if args.dry_run:
        print(json.dumps({"checks": check_manuscript(), "dry_run": True}, indent=2))
        return 0
    generate_pipeline_figure(resume=args.resume, force=args.force)
    generate_smoke_table(resume=args.resume, force=args.force)
    generate_dataset_placeholder(resume=args.resume, force=args.force)
    problems = check_manuscript()
    if problems:
        print(json.dumps({"status": "failed", "problems": problems}, indent=2))
        return 1
    compile_status = compile_latex(output_dir)
    summary = {"status": "completed", "compile": compile_status}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
