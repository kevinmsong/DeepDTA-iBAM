"""Rebuild the supplementary scaffold-split ablation figure.

The original was produced by the full case-study pipeline at 300 dpi with an
earlier palette.  This rebuilds the same figure from the released aggregate
table, using the shared manuscript style so that it matches the main-text
figures in typography, palette and resolution.

Source:  results/ablation_table_wide.csv (three retraining seeds per variant)
Output:  results/fig_ablation_scaffold_summary.{pdf,png}

Run from anywhere:  python analysis/make_figure_scaffold_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import COLORS, FULL_WIDTH, apply_style, save  # noqa: E402

LABELS = {
    "abl_base": "Backbone only",
    "abl_full": "DeepDTA-iBAM",
    "abl_no_fusion": "No cross-attention",
    "abl_no_diffusion": "No diffusion head",
    "abl_no_ranking": "No ranking loss",
}


def main() -> None:
    import matplotlib.pyplot as plt

    apply_style()

    df = pd.read_csv(RESULTS / "ablation_table_wide.csv")
    df = df[df["split_type"] == "scaffold"].copy()
    df["label"] = df["model_name"].map(LABELS).fillna(df["model_name"])
    df = df.sort_values("CI_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 2.25))
    y = range(len(df))
    for i, row in df.iterrows():
        full = row["label"] == "DeepDTA-iBAM"
        c = COLORS["primary"] if full else COLORS["grey"]
        ax.plot([row["CI_mean"] - row["CI_sd"], row["CI_mean"] + row["CI_sd"]],
                [i, i], color=c, linewidth=1.4, solid_capstyle="butt", zorder=2)
        ax.plot([row["CI_mean"]], [i], marker="o" if full else "s",
                markersize=5 if full else 4, color=c, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel(r"Scaffold-split concordance index (mean $\pm$ 1 SD, 3 seeds)")
    ax.axvline(float(df.loc[df["label"] == "DeepDTA-iBAM", "CI_mean"].iloc[0]),
               color=COLORS["primary"], linestyle=":", linewidth=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax.grid(axis="y", visible=False)

    save(fig, "fig_ablation_scaffold_summary")
    print("wrote results/fig_ablation_scaffold_summary.{pdf,png}")
    print(df[["label", "CI_mean", "CI_sd", "CI_n"]].to_string(index=False))


if __name__ == "__main__":
    main()
