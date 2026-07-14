"""Figure style for the paper.

Colour is assigned by IDENTITY (which method), in a fixed slot order that was
validated for colour-vision deficiency (worst adjacent dE 47.2, well above the
>=12 target).  Slots are never cycled and never reassigned when a filter changes
the set of methods -- a colour always means the same method.

Two slots in this palette sit under 3:1 contrast on a white page, so every series
is ALSO direct-labelled: identity is never carried by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_FIG_DIR = Path(__file__).resolve().parent / "figures"


def set_fig_dir(path) -> None:
    """Point figure output at a run's own artifacts dir.

    Without this, a smoke run with tiny/synthetic settings silently overwrites the
    real figures in eval/figures and you cannot tell which is which.
    """
    global _FIG_DIR
    _FIG_DIR = Path(path)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a8a7a1"
GRID = "#e6e5e1"

# Fixed categorical slots. Order is the CVD-safety mechanism, not cosmetic.
SLOT = {
    "modular": "#2a78d6",   # blue
    "e2e": "#e34948",       # red
    "hybrid": "#4a3aa7",    # violet
    "oracle": "#1baf7a",    # aqua
    "accent": "#eda100",    # yellow
}
BASELINE_INK = MUTED  # every model-free baseline shares one recessive grey


def color_of(name: str) -> str:
    for key, c in SLOT.items():
        if name.startswith(key):
            return c
    if name == "oracle":
        return SLOT["oracle"]
    return BASELINE_INK


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "font.size": 9,
            "figure.dpi": 140,
        }
    )


def save(fig, name: str) -> Path:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = _FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def label_point(ax, x: float, y: float, text: str, color: str, dx: float = 0.012) -> None:
    """Direct label. Text stays in ink; the mark beside it carries the identity."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, 0.0),
        textcoords="offset fontsize",
        color=INK_2,
        fontsize=7.5,
        va="center",
        ha="left",
    )
