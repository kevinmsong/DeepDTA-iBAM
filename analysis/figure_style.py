"""Shared publication figure style for the IEEE Access manuscript.

Every figure in the paper is built through this module so the whole set reads as
one system: one font ladder, one line weight, one colorblind-safe palette, and
one export path that writes both a vector PDF for the typeset manuscript and a
600 dpi raster for submission systems that require one.

Design rules enforced here:

  * Sized to IEEE column width (3.5 in) or full text width (7.16 in) at final
    scale, so no figure is rescaled by ``includegraphics`` and no text ends up
    below about 7 pt.
  * Fonts embedded as Type 42 so text stays selectable and is never rasterised.
  * A colorblind-safe palette (Okabe-Ito), with colour never the only channel
    carrying meaning; callers pair it with ``MARKERS`` or ``LINESTYLES``.
  * No chart junk: top and right spines off, light axis-aligned grid only.

Usage::

    from figure_style import apply_style, figure, save, COLORS, MARKERS
    apply_style()
    fig, ax = figure(width="single", height=2.4)
    ...
    save(fig, "fig_name")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# IEEE two-column geometry, in inches.
COL_WIDTH = 3.5
FULL_WIDTH = 7.16

DPI = 600

# Okabe-Ito hues, ordered so that the first three are the set that survives all
# three common colour-vision deficiencies.  Blue and bluish green are the pair
# that collapses under tritanopia (CIE76 distance 10.9), so green is demoted out
# of the first three and the working trio is blue, vermillion, reddish purple,
# which holds a worst-case distance of 27.0 across normal vision, protanopia,
# deuteranopia and tritanopia.  Verified by analysis/check_colorblind.py.
# Callers key by role rather than hue so the mapping can change in one place.
COLORS = {
    "primary":   "#0072B2",   # blue
    "secondary": "#D55E00",   # vermillion
    "tertiary":  "#CC79A7",   # reddish purple
    "quaternary": "#E69F00",  # orange
    "accent":    "#009E73",   # bluish green
    "muted":     "#56B4E9",   # sky blue
    "dark":      "#000000",
    "grey":      "#7F7F7F",
    "light":     "#BFBFBF",
}
CYCLE = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
         COLORS["quaternary"], COLORS["accent"], COLORS["muted"]]

# Sequential colormap for the one heatmap in the paper.  Cividis is
# perceptually uniform and was constructed so that a viewer with deuteranomaly
# sees very nearly what a viewer with normal colour vision sees, and it rises
# monotonically in lightness, so it also survives greyscale printing.  Verified
# by analysis/check_colorblind.py.
SEQUENTIAL = "cividis"

# Redundant encoding channels, so a figure survives greyscale printing.
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def apply_style() -> None:
    mpl.rcParams.update({
        # --- export -------------------------------------------------------
        "figure.dpi": 200,          # on-screen; save() overrides for output
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,         # embed TrueType, keep text selectable
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # --- fonts --------------------------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "mathtext.fontset": "dejavusans",

        # --- lines and marks ---------------------------------------------
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),

        # --- axes ---------------------------------------------------------
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["light"],
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # --- legend -------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,

        "figure.constrained_layout.use": True,
    })


def figure(width: str = "single", height: float = 2.4, **kwargs):
    """Create a figure sized to the IEEE column or full text width."""
    w = COL_WIDTH if width == "single" else FULL_WIDTH
    return plt.subplots(figsize=(w, height), **kwargs)


def panel_labels(axes, labels=None, x: float = -0.16, y: float = 1.04) -> None:
    """Put (a), (b), ... on each panel of a multi-panel figure."""
    import numpy as np
    axes = np.atleast_1d(axes).ravel()
    labels = labels or [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        ax.text(x, y, lab, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str, outdir: Path | None = None) -> tuple[Path, Path]:
    """Write the figure as a vector PDF and a 600 dpi PNG."""
    outdir = Path(outdir) if outdir else RESULTS
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    return pdf, png


def check_min_font(fig, minimum: float = 6.5) -> list[str]:
    """Report any text artist rendering below the minimum point size."""
    small = []
    for t in fig.findobj(match=mpl.text.Text):
        if t.get_text().strip() and t.get_fontsize() < minimum:
            small.append(f"{t.get_text()[:30]!r} at {t.get_fontsize()}pt")
    return small
