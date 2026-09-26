"""Build the supplementary residual-diagnostics figure.

The Discussion argues that the affinity head should triage a series but should
not commit to a single compound, and rests that on limits of agreement spanning
-0.895 to 0.876. This figure is the evidence for it. The previous version was
produced by the case-study pipeline at 300 dpi on an earlier palette; this one
is built through the shared manuscript style.

Panels
------
(a) Bland-Altman agreement between predicted and measured KIBA score, with the
    mean bias and the 95% limits of agreement.
(b) Residual against measured value, with a binned median, which shows whether
    the error is centered across the range or drifts with potency.

With 5,913 held-out pairs a scatter would be a solid block, so both panels are
hexagon-binned and the count is carried on a shared color scale.

Source:  results/max_rmse_cluster_diffusion_member1_standard_predictions.csv
Output:  results/fig_residual_diagnostics.{pdf,png}

Run from anywhere:  python analysis/make_figure_residuals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import (COLORS, SEQUENTIAL, apply_style,  # noqa: E402
                          check_min_font, save)

# The supplement is single column on a 1 in margin page, so it is a little
# narrower than the IEEE two-column text width.
SUPP_WIDTH = 6.4


def main() -> None:
    import matplotlib.pyplot as plt

    apply_style()
    df = pd.read_csv(RESULTS / "max_rmse_cluster_diffusion_member1_standard_predictions.csv")
    obs, pred = df["target"].values, df["prediction"].values
    diff = pred - obs
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(SUPP_WIDTH, 2.6))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.085, right=0.885,
                          top=0.88, bottom=0.165)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # --- (a) Bland-Altman --------------------------------------------------
    mean_xy = (pred + obs) / 2
    hb = ax_a.hexbin(mean_xy, diff, gridsize=42, cmap=SEQUENTIAL,
                     mincnt=1, linewidths=0, zorder=2)
    for y, style, label in ((bias, "-", f"bias {bias:+.3f}"),
                            (lo, "--", f"95% limits {lo:.3f} to {hi:.3f}"),
                            (hi, "--", None)):
        ax_a.axhline(y, color=COLORS["secondary"], linewidth=1.0,
                     linestyle=style, zorder=3, label=label)
    ax_a.set_xlabel("Mean of predicted and measured KIBA score")
    ax_a.set_ylabel("Predicted $-$ measured")
    ax_a.set_title("(a) Agreement", fontsize=7.5)
    ax_a.legend(fontsize=6.5, frameon=True, framealpha=0.9, loc="lower right",
                borderpad=0.25, handlelength=1.5, labelspacing=0.25)

    # --- (b) residual against measured value -------------------------------
    hb_b = ax_b.hexbin(obs, diff, gridsize=42, cmap=SEQUENTIAL, mincnt=1,
                       linewidths=0, zorder=2)
    # Both panels use the count scale represented by the shared colorbar.
    from matplotlib.colors import Normalize
    count_norm = Normalize(vmin=1, vmax=max(hb.get_array().max(),
                                           hb_b.get_array().max()))
    hb.set_norm(count_norm)
    hb_b.set_norm(count_norm)
    ax_b.axhline(0.0, color=COLORS["dark"], linewidth=0.8, zorder=3)

    # Binned median, so systematic drift is visible through the density.
    edges = np.linspace(obs.min(), obs.max(), 13)
    centers, med = [], []
    for a, b in zip(edges, edges[1:]):
        m = (obs >= a) & (obs < b)
        if m.sum() >= 20:
            centers.append((a + b) / 2)
            med.append(float(np.median(diff[m])))
    ax_b.plot(centers, med, color=COLORS["secondary"], linewidth=1.2,
              marker="o", markersize=2.8, zorder=4, label="binned median")
    ax_b.set_xlabel("Measured KIBA score")
    ax_b.set_ylabel("Predicted $-$ measured")
    ax_b.set_title("(b) Drift across the range", fontsize=7.5)
    ax_b.legend(fontsize=6.5, frameon=True, framealpha=0.9, loc="upper right",
                borderpad=0.25, handlelength=1.5)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linewidth=0.4, alpha=0.4)

    cax = fig.add_axes([0.9, 0.165, 0.018, 0.715])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Pairs per bin", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.6)

    small = check_min_font(fig)
    if small:
        print("   warning, small text:", small[:3])
    print("  wrote", save(fig, "fig_residual_diagnostics")[0].name)
    print(f"  n={len(df):,}  bias {bias:+.4f}  limits {lo:.3f} to {hi:.3f}")


if __name__ == "__main__":
    main()
