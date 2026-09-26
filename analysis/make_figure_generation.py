"""Build the main-text figure for the analog-generation capability.

Section IV-F reports the diffusion head's behavior entirely in numbers. This
figure shows it, and shows the limit alongside the capability, because the two
are the same fact seen twice: the head moves a long way from the seed in
fingerprint space while never leaving its skeleton.

Panels
------
(a) Fingerprint similarity of each proposal to the dasatinib seed, per
    generator. The diffusion head proposes the most distant analogs, which is
    the capability the other two baselines do not have.
(b) Drug-likeness against synthetic accessibility for the same molecules, with
    the seed marked. This is the price: diffusion proposals are less drug-like
    and harder to make than the fragment-swap baseline.
(c) Distinct-structure counts through the audit, by three definitions of
    "distinct". Canonical SMILES and Bemis-Murcko scaffolds track the set size;
    generic frameworks, which erase atom types and bond orders, stay at one.
    The apparent scaffold diversity is an artifact of how it was counted.

Sources
-------
  results/generation_comparison.csv          240 molecules, three generators
  results/generated_egfr_analogs_audited.csv the 60 surviving the audit

Output:  results/fig_generation.{pdf,png}

Run from anywhere:  python analysis/make_figure_generation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import (COLORS, FULL_WIDTH, MARKERS, apply_style,  # noqa: E402
                          check_min_font, save)

GENS = [("diffusion", "Diffusion head", "primary"),
        ("fragment_swap", "Fragment swap", "secondary"),
        ("random_edit", "Random atom edit", "tertiary")]

# Audit stages, from table_generated_audit.tex, which is built from the same
# released CSVs.
STAGES = ["As released", "Seed\nremoved", "Alert-\nfiltered"]
COUNTS = {
    "Canonical SMILES": [100, 99, 60],
    "Murcko scaffolds": [98, 97, 59],
    "Generic frameworks": [1, 1, 1],
}


def main() -> None:
    import matplotlib.pyplot as plt

    apply_style()
    df = pd.read_csv(RESULTS / "generation_comparison.csv")
    seed_qed, seed_sa = 0.465717, 2.649762     # seed_reference row of the summary

    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(FULL_WIDTH, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.42,
                          left=0.075, right=0.99, top=0.86, bottom=0.235)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- (a) distance from the seed ---------------------------------------
    rng = np.random.default_rng(0)
    for i, (key, label, col) in enumerate(GENS):
        v = df.loc[df.generator == key, "tanimoto"].values
        c = COLORS[col]
        ax_a.scatter(v, i + rng.uniform(-0.17, 0.17, len(v)), s=3.2,
                     color=c, alpha=0.45, linewidths=0, zorder=2)
        ax_a.plot([np.median(v)], [i], marker="|", markersize=11,
                  markeredgewidth=1.6, color=COLORS["dark"], zorder=4)
    ax_a.set_yticks(range(len(GENS)))
    ax_a.set_yticklabels([g[1] for g in GENS])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 1.06)
    ax_a.set_ylim(2.55, -0.75)
    ax_a.set_xlabel("ECFP similarity to the seed")
    ax_a.set_title("(a) How far each generator moves", fontsize=7.5)
    ax_a.annotate("seed returned\nin its own output", xy=(1.0, 0.0),
                  xytext=(0.46, -0.46), fontsize=6.5, ha="center",
                  arrowprops=dict(arrowstyle="->", linewidth=0.6,
                                  color=COLORS["grey"]))
    for s in ("top", "right", "left"):
        ax_a.spines[s].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax_a.grid(axis="y", visible=False)

    # --- (b) what that costs ----------------------------------------------
    for (key, label, col), mk in zip(GENS, MARKERS):
        sub = df[df.generator == key]
        ax_b.scatter(sub.SA, sub.QED, s=7, marker=mk, facecolor="none",
                     edgecolor=COLORS[col], linewidths=0.55, alpha=0.8,
                     label=label, zorder=2)
    ax_b.plot([seed_sa], [seed_qed], marker="*", markersize=9,
              color=COLORS["dark"], linestyle="none", label="Dasatinib seed",
              zorder=4)
    ax_b.set_xlabel("Synthetic accessibility (lower is easier)")
    ax_b.set_ylabel("QED (higher is more drug-like)")
    ax_b.set_title("(b) The price of moving further", fontsize=7.5)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(linewidth=0.4, alpha=0.5)
    ax_b.legend(fontsize=6.5, frameon=False, loc="lower left",
                handletextpad=0.2, borderpad=0.15, labelspacing=0.25)

    # --- (c) the diversity audit ------------------------------------------
    x = np.arange(len(STAGES))
    w = 0.26
    styles = [(COLORS["primary"], None), (COLORS["secondary"], None),
              (COLORS["tertiary"], "///")]
    for i, ((name, vals), (col, hatch)) in enumerate(zip(COUNTS.items(), styles)):
        bars = ax_c.bar(x + (i - 1) * w, vals, w, label=name, color=col,
                        edgecolor=COLORS["dark"], linewidth=0.5, hatch=hatch,
                        zorder=2)
        for b, v in zip(bars, vals):
            ax_c.text(b.get_x() + b.get_width() / 2, v + 2.5, str(v),
                      ha="center", va="bottom", fontsize=6.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(STAGES)
    ax_c.set_ylim(0, 152)
    ax_c.set_ylabel("Distinct structures")
    ax_c.set_title("(c) Diversity depends on how you count", fontsize=7.5)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax_c.grid(axis="x", visible=False)
    ax_c.legend(fontsize=6.5, frameon=False, loc="upper right",
                handlelength=1.1, handletextpad=0.3, borderpad=0.15,
                labelspacing=0.25)

    small = check_min_font(fig)
    if small:
        print("   warning, small text:", small[:3])
    print("  wrote", save(fig, "fig_generation")[0].name)
    for key, label, _ in GENS:
        v = df[df.generator == key]
        print(f"  {label:18s} n={len(v):3d}  median Tanimoto {v.tanimoto.median():.3f}"
              f"  QED {v.QED.mean():.3f}  SA {v.SA.mean():.3f}")


if __name__ == "__main__":
    main()
