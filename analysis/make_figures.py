"""
Publication figures for the J. Cheminform. submission.

Generates two figures that present the revision's two principal findings and
that have no counterpart in the original pipeline output:

  fig_forest_interpretability  Forest plot of per-complex residue contact AUROC
                               with analytic (Hanley-McNeil) intervals, stratified
                               by inhibitor binding mode, with fixed-effect and
                               random-effects pooled estimates.
  fig_retrieval_separation     Class-conditional nearest-anchor Tanimoto
                               distributions for the EGFR retrieval panel, showing
                               the disjoint support that makes fingerprint
                               baselines inadmissible.

Palette is the validated two-hue categorical pair (blue #2a78d6, orange #eb6834);
adjacent-pair separation is dE 24.7 under protanopia, so the figures remain
readable under colour-vision deficiency and in greyscale print.

Run from the submission directory:  python make_figures.py
"""

# --- path resolution -------------------------------------------------------
# Paths are anchored to this file, not the working directory, so the scripts
# run identically from a clean clone. LaTeX fragments are written to the
# manuscript directory when it is present (it is not part of this repository)
# and beside this script otherwise.
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_MS = _ROOT / "J_Cheminform_Submission"
OUT_DIR = _MS if _MS.is_dir() else _Path(__file__).resolve().parent


def _out(name):
    return str(OUT_DIR / name)
# ---------------------------------------------------------------------------

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from analysis_revision import hanley_mcneil_se, Z

RES = str(_ROOT / "results") + "/"
DATA = str(_ROOT / "data") + "/"

BLUE = "#2a78d6"      # categorical slot 1 -- type I
ORANGE = "#eb6834"    # categorical slot 2 -- type II
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.7,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 150,
})


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
def forest_plot():
    rows = list(csv.DictReader(open(RES + "interpretability_benchmark.csv")))
    mode = {"2HYY": "II", "4RJ3": "II", "1KE6": "I", "4WKQ": "I", "6YOJ": "I"}
    lig = {"1KE6": "LS2", "2HYY": "imatinib", "4RJ3": "3QS", "4WKQ": "erlotinib", "6YOJ": "P4N"}

    per = []
    for x in rows:
        a = float(x["residue_contact_auroc"])
        npos = int(x["n_contact_residues"])
        nneg = int(x["n_residues"]) - npos
        se = hanley_mcneil_se(a, npos, nneg)
        per.append((x["pdb_id"], x["protein"], a, se, npos, nneg, mode[x["pdb_id"]]))
    # type II first so the visual grouping reads top-down
    per.sort(key=lambda r: (r[6] != "II", -r[2]))

    a = np.array([p[2] for p in per])
    se = np.array([p[3] for p in per])
    w = 1 / se ** 2
    fe = (w * a).sum() / w.sum()
    se_fe = np.sqrt(1 / w.sum())
    q = (w * (a - fe) ** 2).sum()
    tau2 = max(0.0, (q - 4) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    re = (ws * a).sum() / ws.sum()
    se_re = np.sqrt(1 / ws.sum())
    i2 = max(0.0, (q - 4) / q) * 100

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    n = len(per)
    ypos = list(range(n))[::-1]

    for y, p in zip(ypos, per):
        pdb, prot, est, s, npos, nneg, m = p
        col = ORANGE if m == "II" else BLUE
        lo, hi = est - Z * s, est + Z * s
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
        # marker area proportional to inverse variance (standard forest convention)
        ax.scatter([est], [y], s=90 * (1 / s ** 2) / (1 / se.min() ** 2) + 26,
                   color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(-0.012, y, f"{pdb}  {prot}", ha="right", va="center",
                fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
        sig = "*" if lo > 0.5 else ""
        ax.text(1.012, y, f"{est:.3f} [{lo:.3f}, {hi:.3f}]{sig}", ha="left", va="center",
                fontsize=8, color=INK2, transform=ax.get_yaxis_transform())

    # pooled estimates as diamonds
    for i, (label, est, s) in enumerate(
            [("Fixed effect", fe, se_fe), ("Random effects", re, se_re)]):
        y = -1.1 - i * 0.85
        lo, hi = est - Z * s, est + Z * s
        ax.add_patch(Polygon([[lo, y], [est, y + 0.26], [hi, y], [est, y - 0.26]],
                             closed=True, facecolor=INK2, edgecolor=INK, lw=0.6, zorder=3))
        ax.text(-0.012, y, label, ha="right", va="center", fontsize=8.5,
                style="italic", color=INK, transform=ax.get_yaxis_transform())
        ax.text(1.012, y, f"{est:.3f} [{lo:.3f}, {hi:.3f}]", ha="left", va="center",
                fontsize=8, color=INK2, transform=ax.get_yaxis_transform())

    ax.axvline(0.5, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, -2.72, "chance", ha="center", va="bottom", fontsize=8, color=INK2)

    ax.set_xlim(0.25, 0.90)
    ax.set_ylim(-2.95, n - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Residue contact AUROC (95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))

    h1 = plt.Line2D([], [], color=ORANGE, marker="o", lw=1.6, ms=6,
                    markeredgecolor="white", label="Type II (DFG-out)")
    h2 = plt.Line2D([], [], color=BLUE, marker="o", lw=1.6, ms=6,
                    markeredgecolor="white", label="Type I (ATP-competitive)")
    ax.legend(handles=[h1, h2], loc="upper left", frameon=False,
              handletextpad=0.5, labelspacing=0.35, borderpad=0.1)

    fig.subplots_adjust(left=0.24, right=0.74, top=0.95, bottom=0.16)
    for ext in ("pdf", "png"):
        fig.savefig(_out(f"fig_forest_interpretability.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    return dict(fe=fe, re=re, i2=i2)


# ---------------------------------------------------------------------------
def separation_plot():
    rows = list(csv.DictReader(open(RES + "egfr_interpolation_ranked_candidates.csv")))
    y = np.array([1 if x["panel_role"] == "holdout" else 0 for x in rows])
    tan = np.array([float(x["NearestAnchorTanimoto"]) for x in rows])
    pos, neg = tan[y == 1], tan[y == 0]

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    bins = np.linspace(0, 1.0, 61)
    ax.hist(neg, bins=bins, color=BLUE, alpha=0.85, label=f"Decoys (n = {len(neg):,})",
            edgecolor="white", linewidth=0.4)
    ax.hist(pos, bins=bins, color=ORANGE, alpha=0.85, label=f"Positives (n = {len(pos)})",
            edgecolor="white", linewidth=0.4)

    ax.axvspan(neg.max(), pos.min(), color=GRID, alpha=0.55, zorder=0)
    top = ax.get_ylim()[1]
    mid = (neg.max() + pos.min()) / 2
    ax.annotate(f"empty interval\nwidth {pos.min() - neg.max():.3f}",
                xy=(mid, top * 0.62), ha="center", va="center",
                fontsize=8, color=INK2, linespacing=1.4)
    ax.axvline(0.40, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("selection cutoff\n(Tanimoto $\\geq$ 0.40)", xy=(0.40, top * 0.90),
                xytext=(0.53, top * 0.90), fontsize=8, color=INK,
                va="center", ha="left", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))

    ax.set_xlabel("Nearest-anchor ECFP Tanimoto similarity")
    ax.set_ylabel("Candidate count")
    ax.set_xlim(0, 1.0)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_out(f"fig_retrieval_separation.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    return dict(gap=pos.min() - neg.max())


if __name__ == "__main__":
    f = forest_plot()
    s = separation_plot()
    print(f"forest: fixed={f['fe']:.4f} random={f['re']:.4f} I2={f['i2']:.0f}%")
    print(f"separation: gap={s['gap']:.4f}")
    print("wrote fig_forest_interpretability.{pdf,png}, fig_retrieval_separation.{pdf,png}")
