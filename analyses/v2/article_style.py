"""article_style.py — shared figure style for the article (palette taken from the PhD defence slides).

Figures are drawn at their printed size (full width 7.2 in, one column 3.5 in) so that font sizes are the
printed sizes; saved as 300-dpi PNG and as PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CRIMSON, PLUM, BLUE, SKY, PINK = "#c60b46", "#63003c", "#1f5fae", "#95bbea", "#eaa8bd"
AMBER, TEAL = "#c98a12", "#2a9d8f"
INK, INK2, GRID = "#1a1a1a", "#5a5a5a", "#e6e4e1"

RISK = {"low": BLUE, "intermediate": AMBER, "high": CRIMSON}          # validated: CVD dE >= 16
MODEL = {"SHERLOC": CRIMSON, "Clinical Cox": BLUE}
ARM = {"Panel": CRIMSON, "Encoder": BLUE, "Encoder gene": BLUE, "Clinical": AMBER, "Gene": TEAL}
FULL, HALF = 7.2, 3.5

plt.rcParams.update({
    "font.family": "Inter", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def panel_label(ax, letter, x=-0.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="right")


def grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def save(fig, path):
    from pathlib import Path
    path = Path(path)
    fig.savefig(path.with_suffix(".png")); fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
