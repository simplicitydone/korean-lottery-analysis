"""viz.py — one shared visual system for every chart (notebooks + web dashboard).

Palette and mark rules follow the project's data-viz design system (validated categorical
hues, recessive grid, thin marks). Import ``apply_style()`` once per notebook; use ``CAT`` for
categorical series colors in fixed order, ``SEQ`` for a single-hue magnitude ramp, and the
``INK``/``STATUS`` roles for text and state.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical hues — assigned in this fixed order, never cycled past slot 8.
CAT = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# Single-hue sequential ramp (blue, light -> dark) for continuous magnitude.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}


def apply_style() -> None:
    """Apply the shared matplotlib rcParams: recessive chrome, thin marks, system sans."""
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": ["Segoe UI", "Malgun Gothic", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.edgecolor": INK["baseline"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlecolor": INK["primary"],
        "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "text.color": INK["primary"],
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })
    # Korean glyphs render with a minus that isn't a unicode-minus box.
    mpl.rcParams["axes.unicode_minus"] = False


def reference_line(ax, y, label=None, color=None):
    """Draw a recessive dashed reference line (e.g. the uniform-expectation baseline)."""
    color = color or INK["muted"]
    ax.axhline(y, ls="--", lw=1.0, color=color, zorder=0)
    if label:
        ax.text(
            0.995, y, f" {label}", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=9, color=color,
        )


def savefig(fig, name: str):
    """Save a figure into reports/figures/ with the shared surface background."""
    from . import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    return path
