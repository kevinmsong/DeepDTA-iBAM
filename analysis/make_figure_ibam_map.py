"""Build the main-text figure showing what an interpretable binding attention
map actually looks like.

The paper is named for the iBAM but reports it only through summary statistics.
This figure shows the map itself for one complex, beside the per-residue profile
derived from it and the same profile for a complex where attention does not
localize, so the reader sees both what the map looks like and how far it can be
trusted.

Panels
------
(a) The atom-by-residue attention matrix for 2HYY (ABL1 with imatinib), cropped
    to the window of the sequence that carries the signal.  This is the matrix
    defined in the System Architecture section, before any averaging.
(b) The per-residue profile for the same complex, which is the matrix averaged
    down its atom axis, with crystallographic contacts marked.
(c) The same profile for 4WKQ (EGFR with gefitinib), a type I complex whose
    contact AUROC interval includes chance.

Attention localizes in two of the five co-crystal complexes; showing only the
successful case would misrepresent that variation. The former explanation in
terms of binding mode was withdrawn because structural labels were unverified.

Sources
-------
  results/ibam_matrix_2HYY.npz, results/ibam_matrix_4WKQ.npz
      the matrices, from export_residue_level.py --dump-matrix
  results/interpretability_residue_level.csv
      the released per-residue profile, used for the curves in (b) and (c)
  results/interpretability_benchmark.csv
      the released per-complex metrics, used for the AUROC annotations

The AUROC values are read from the benchmark CSV rather than recomputed from
the profile, because that CSV is the artifact the forest plot and the manuscript
both quote.  The two released files come from separate runs of the same
checkpoint and their AUROCs differ by about 0.001, so recomputing here would put
a number on the figure that disagrees in the third decimal with the number in
the text.

Output:  results/fig_ibam_map.{pdf,png}

Run from anywhere:  python analysis/make_figure_ibam_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_style import (COLORS, FULL_WIDTH, SEQUENTIAL, apply_style,  # noqa: E402
                          check_min_font, save)

MAIN = "2HYY"        # ABL1 + imatinib, type II, contact AUROC 0.709
FOIL = "4WKQ"        # EGFR + gefitinib, interval includes chance
PAD = 12             # residues of context either side of the contact window


def load(pdb_id: str):
    d = np.load(RESULTS / f"ibam_matrix_{pdb_id}.npz")
    return {k: d[k] for k in d}


def released_profile(pdb_id: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "interpretability_residue_level.csv")
    return df[df.pdb_id == pdb_id].reset_index(drop=True)


def released_auroc(pdb_id: str) -> float:
    """Residue contact AUROC as the manuscript and the forest plot report it."""
    df = pd.read_csv(RESULTS / "interpretability_benchmark.csv")
    return float(df.loc[df.pdb_id == pdb_id, "residue_contact_auroc"].iloc[0])


def contact_window(contact: np.ndarray, pad: int = PAD):
    idx = np.flatnonzero(contact)
    return max(0, idx.min() - pad), min(len(contact), idx.max() + pad + 1)


def profile_panel(ax, prof: pd.DataFrame, lo: int, hi: int, title: str,
                  show_ylabel: bool) -> None:
    """Per-residue attention over a window, with contacts marked."""
    x = np.arange(lo, hi)
    y = prof.attention.values[lo:hi]
    c = prof.contact.values[lo:hi].astype(bool)

    # Attention spans more than two decades and a single non-contact residue
    # carries the largest weight in each complex, so a linear axis would flatten
    # the pocket region into the baseline.  A log axis shows both without
    # clipping anything away.
    floor = float(np.percentile(y, 1)) * 0.7
    ax.fill_between(x, floor, y, color=COLORS["light"], linewidth=0, zorder=1)
    ax.plot(x, y, color=COLORS["grey"], linewidth=0.7, zorder=2)
    # Contacts carry a marker as well as a color, so the panel survives
    # greyscale printing and color-vision deficiency.
    ax.plot(x[c], y[c], linestyle="none", marker="v", markersize=3.2,
            color=COLORS["secondary"], zorder=3,
            label=f"crystal contact (n={int(c.sum())})")

    ax.set_yscale("log")
    ax.set_xlim(lo, hi - 1)
    ax.set_ylim(floor, float(y.max()) * 3.2)
    ax.set_xlabel("Residue position")
    if show_ylabel:
        ax.set_ylabel("Mean attention")
    ax.set_title(title, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right",
              handletextpad=0.4, borderpad=0.2)


def main() -> None:
    import matplotlib.pyplot as plt

    apply_style()

    m = load(MAIN)
    prof_main = released_profile(MAIN)
    prof_foil = released_profile(FOIL)

    contact = m["contact"].astype(bool)
    lo, hi = contact_window(contact)
    labels = m["residue_label"]

    a_main = released_auroc(MAIN)
    a_foil = released_auroc(FOIL)

    # The shared style turns constrained layout on, which fights an explicitly
    # placed gridspec; this figure sets its own margins instead.
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(FULL_WIDTH, 3.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0],
                          height_ratios=[1.0, 1.0],
                          wspace=0.30, hspace=1.05,
                          left=0.072, right=0.985, top=0.885, bottom=0.145)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- (a) the map itself ------------------------------------------------
    sub = m["attention"][:, lo:hi]
    # Log color scale for the same reason the profiles use a log axis: the
    # weights span more than two decades, and on a linear scale one residue
    # saturates the map and everything else reads as empty.
    from matplotlib.colors import LogNorm
    im = ax_map.imshow(sub, aspect="auto", cmap=SEQUENTIAL,
                       interpolation="nearest",
                       norm=LogNorm(vmin=float(sub.min()), vmax=float(sub.max())),
                       extent=(lo - 0.5, hi - 0.5, sub.shape[0] - 0.5, -0.5))
    ax_map.grid(False)          # the shared style's grid would overlay the map
    ax_map.set_xlabel("Protein residue")
    ax_map.set_ylabel("Ligand graph atom")
    ax_map.set_title(f"(a) {str(m['protein'])} + imatinib: attention matrix",
                     fontsize=7.5)

    # Label a readable subset of residues, and mark the contacts beneath.
    ticks = np.linspace(lo, hi - 1, 7, dtype=int)
    ax_map.set_xticks(ticks)
    ax_map.set_xticklabels([labels[t] for t in ticks], rotation=45,
                           ha="right", fontsize=6.5)
    ax_map.set_yticks(np.arange(0, sub.shape[0], 6))
    ax_map.set_yticklabels([m["atom_symbol"][i]
                            for i in range(0, sub.shape[0], 6)], fontsize=6.5)
    shown = [p for p in np.flatnonzero(contact) if lo <= p < hi]
    ax_map.plot(shown, [sub.shape[0] + 0.9] * len(shown), linestyle="none",
                marker="v", markersize=2.8, color=COLORS["secondary"],
                clip_on=False, zorder=5)
    ax_map.tick_params(length=2)

    cb = fig.colorbar(im, ax=ax_map, pad=0.015, fraction=0.045)
    cb.set_label("Attention weight", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.6)

    # --- (b) and (c) the profiles -----------------------------------------
    profile_panel(ax_b, prof_main, lo, hi,
                  f"(b) ABL1 atom-averaged profile: AUROC {a_main:.3f}",
                  show_ylabel=True)
    lo_f, hi_f = contact_window(prof_foil.contact.values.astype(bool))
    profile_panel(ax_c, prof_foil, lo_f, hi_f,
                  f"(c) EGFR + gefitinib: AUROC {a_foil:.3f}",
                  show_ylabel=True)

    small = check_min_font(fig)
    if small:
        print("   warning, small text:", small[:3])
    print("  wrote", save(fig, "fig_ibam_map")[0].name)
    print(f"  {MAIN} window residues {lo}-{hi}, "
          f"{int(contact[lo:hi].sum())} of {int(contact.sum())} contacts shown")
    print(f"  AUROC  {MAIN} {a_main:.4f}   {FOIL} {a_foil:.4f}")


if __name__ == "__main__":
    main()
