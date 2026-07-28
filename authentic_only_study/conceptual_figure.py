from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "outputs" / "authentic_only" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

stages = [
    ("Candidate", "Source +\ntranslation", "What can be\nobserved?"),
    ("Surface", "Tokens\nand names", "Did related\nconcepts appear?"),
    ("Validity", "Syntax", "Can the candidate\nexist?"),
    ("Structure", "Organization", "Is computation\norganized alike?"),
    ("Control", "Paths\nand loops", "Can control\nproceed alike?"),
    ("Operators", "Local\ncomputation", "Are operations\ncompatible?"),
    ("Roles", "Responsibilities", "Do values play\nsimilar roles?"),
    ("Flow", "Value\npropagation", "Do values reach\nsimilar uses?"),
    ("APIs", "Calls", "Are target calls\nplausible?"),
    ("Execution", "Concrete\nbehavior", "Do observed\noutcomes agree?"),
    ("Decision", "Act or\nacquire", "Accept, inspect,\ntest, or abstain"),
]
colors = [
    "#F0F2F4", "#E4EFF7", "#D5E8F3", "#C5E0EC",
    "#AED4E1", "#AED4E1", "#AED4E1", "#91C5D3",
    "#91C5D3", "#4694A8", "#E9D4A7",
]

fig, ax = plt.subplots(figsize=(10.8, 3.35))
ax.set_xlim(-0.08, 11.42)
ax.set_ylim(-0.72, 2.42)
ax.axis("off")

step = 1.03
box_w = 0.91

for i, ((title, detail, question), color) in enumerate(zip(stages, colors)):
    x = i * step
    box = FancyBboxPatch(
        (x + 0.02, 0.10), box_w, 1.42,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        facecolor=color, edgecolor="#244B5A", linewidth=1.1,
    )
    ax.add_patch(box)
    text_color = "white" if i in {8, 9} else "#173642"
    center = x + 0.02 + box_w / 2
    ax.text(center, 1.28, title, ha="center", va="center", fontsize=8.0,
            fontweight="bold", color=text_color)
    ax.text(center, 0.94, detail, ha="center", va="center", fontsize=5.9,
            color=text_color, linespacing=1.1)
    ax.text(center, 0.43, question, ha="center", va="center", fontsize=5.55,
            color=text_color, linespacing=1.12)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + step + 0.01, 0.81), xytext=(x + 0.94, 0.81),
                    arrowprops=dict(arrowstyle="-|>", color="#244B5A", lw=1.15))

phase_y = 1.78
for start, end, label in [
    (0.04, 0.91, "Input"),
    (1.07, 3.94, "Representation observations"),
    (4.10, 7.00, "Parallel computation views"),
    (7.16, 9.03, "Relational views"),
    (9.19, 10.06, "Behavior"),
    (10.22, 11.09, "Action"),
]:
    ax.plot([start, end], [phase_y, phase_y], color="#5C7480", lw=1.0)
    ax.text((start + end) / 2, phase_y + 0.12, label, ha="center", va="bottom",
            fontsize=7.0, color="#405965", fontweight="bold")

ax.add_patch(FancyBboxPatch(
    (3.14, -0.13), 5.95, 0.25,
    boxstyle="round,pad=0.015,rounding_size=0.025",
    facecolor="#F3E7C9", edgecolor="#9A7B35", linewidth=0.9,
))
ax.text(6.115, -0.005, "Complexity is cross-cutting: is the amount of computation compatible?",
        ha="center", va="center", fontsize=6.8, color="#6D541E")

ax.annotate("",
            xy=(11.05, 2.27), xytext=(0.25, 2.27),
            arrowprops=dict(arrowstyle="-|>", color="#8B2E2E", lw=1.35))
ax.text(5.65, 2.36, "Typically greater semantic specificity and acquisition cost",
        ha="center", va="center", fontsize=8.3, color="#8B2E2E")
ax.annotate("",
            xy=(11.05, -0.48), xytext=(0.25, -0.48),
            arrowprops=dict(arrowstyle="-|>", color="#315A35", lw=1.35))
ax.text(5.65, -0.39, "Potentially less unresolved uncertainty; monotonic reduction is not guaranteed",
        ha="center", va="center", fontsize=8.0, color="#315A35")

fig.tight_layout(pad=0.2)
fig.savefig(OUT / "semantic_observability_continuum.pdf", bbox_inches="tight")
plt.close(fig)
