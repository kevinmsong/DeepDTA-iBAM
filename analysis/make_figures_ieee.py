"""Regenerate every manuscript figure to publication standard.

All figures in the IEEE Access submission are built here so the set shares one
style: the font ladder, line weights, colorblind-safe palette and export path
defined in figure_style.py.  Each figure is written as a vector PDF for the
typeset manuscript and as a 600 dpi PNG for submission systems that need a
raster.

Figures produced (those whose inputs exist are built; the rest are skipped with
a message, so this script is safe to run at any point during the analysis):

  fig_localization_forest    per-complex residue contact AUROC with analytic
                             intervals, stratified by inhibitor binding mode
  fig_benchmark_validity     the two constructions that cannot measure:
                             atom-level contact saturation, and the disjoint
                             class supports of the similarity-defined panel
  fig_efficiency             throughput, the marginal cost of emitting
                             interaction maps, and scaling with protein length
  fig_interaction_content    what the interaction maps encode
  fig_docking                docking as an external retrieval reference

Run from anywhere:  python analysis/make_figures_ieee.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    COLORS, MARKERS, FULL_WIDTH, apply_style, figure, panel_labels, save,
    check_min_font,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

Z = 1.959964


def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """Analytic standard error of an AUROC (Hanley and McNeil, 1982)."""
    a = float(auroc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a))
    return float(np.sqrt(max(var, 0.0) / (n_pos * n_neg)))


def _skip(name: str, why: str) -> None:
    print(f"  [skip] {name}: {why}")


# ---------------------------------------------------------------------------
def fig_localization_forest() -> None:
    src = RESULTS / "interpretability_benchmark.csv"
    if not src.exists():
        return _skip("fig_localization_forest", f"missing {src.name}")
    df = pd.read_csv(src)

    mode = {"2HYY": "II", "4RJ3": "II", "6YOJ": "I", "4WKQ": "I", "1KE6": "I"}
    df["mode"] = df["pdb_id"].map(mode)
    df["se"] = [hanley_mcneil_se(r.residue_contact_auroc,
                                 int(r.n_contact_residues),
                                 int(r.n_residues) - int(r.n_contact_residues))
                for r in df.itertuples()]
    df["w"] = 1.0 / df["se"] ** 2
    df = df.sort_values("residue_contact_auroc").reset_index(drop=True)

    fixed = float((df["w"] * df["residue_contact_auroc"]).sum() / df["w"].sum())
    se_fixed = float(np.sqrt(1.0 / df["w"].sum()))

    # DerSimonian-Laird random effects
    q = float((df["w"] * (df["residue_contact_auroc"] - fixed) ** 2).sum())
    k = len(df)
    c = float(df["w"].sum() - (df["w"] ** 2).sum() / df["w"].sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1.0 / (df["se"] ** 2 + tau2)
    rand = float((wr * df["residue_contact_auroc"]).sum() / wr.sum())
    se_rand = float(np.sqrt(1.0 / wr.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0

    fig, ax = figure(width="single", height=2.9)
    for i, r in df.iterrows():
        col = COLORS["secondary"] if r["mode"] == "II" else COLORS["primary"]
        mk = MARKERS[1] if r["mode"] == "II" else MARKERS[0]
        lo, hi = r.residue_contact_auroc - Z * r.se, r.residue_contact_auroc + Z * r.se
        ax.plot([lo, hi], [i, i], color=col, lw=1.1, solid_capstyle="round")
        ax.plot([r.residue_contact_auroc], [i], mk, color=col,
                markersize=4.0, markeredgecolor="white", zorder=3)
        if lo > 0.5:
            ax.text(hi + 0.012, i, "*", va="center", fontsize=9, color=col)

    labels = [f"{r.protein} ({r.pdb_id})" for r in df.itertuples()]
    y = len(df)
    for est, se, lab in ((fixed, se_fixed, "Fixed effect"),
                         (rand, se_rand, "Random effects")):
        lo, hi = est - Z * se, est + Z * se
        ax.plot([lo, hi], [y, y], color=COLORS["dark"], lw=1.1)
        ax.plot([est], [y], "D", color=COLORS["dark"], markersize=4.0,
                markeredgecolor="white", zorder=3)
        labels.append(lab)
        y += 1

    ax.axvline(0.5, color=COLORS["grey"], lw=0.8, ls="--", zorder=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Residue contact AUROC")
    ax.set_xlim(0.2, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=COLORS["primary"], marker=MARKERS[0], ls="-",
               markersize=4, label="Type I (ATP-competitive)"),
        Line2D([], [], color=COLORS["secondary"], marker=MARKERS[1], ls="-",
               markersize=4, label="Type II (DFG-out)"),
        Line2D([], [], color=COLORS["dark"], marker="D", ls="none",
               markersize=4, label="Pooled"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=6.5)
    ax.set_title(f"$I^2$ = {i2:.0%}, $Q$ = {q:.1f} on {k-1} df", fontsize=7.5)

    small = check_min_font(fig)
    if small:
        print("   warning, small text:", small[:3])
    print("  wrote", save(fig, "fig_localization_forest")[0].name)


# ---------------------------------------------------------------------------
def fig_benchmark_validity() -> None:
    sat = RESULTS / "overlap_chance_reference.csv"
    sep = RESULTS / "egfr_interpolation_ranked_candidates.csv"
    if not sat.exists():
        return _skip("fig_benchmark_validity", f"missing {sat.name}")
    if not sep.exists():
        return _skip("fig_benchmark_validity", f"missing {sep.name}")

    s = pd.read_csv(sat)
    c = pd.read_csv(sep)

    fig, axes = matplotlib.pyplot.subplots(
        1, 2, figsize=(7.16, 2.8), constrained_layout=True)

    # (a) atom-level saturation
    ax = axes[0]
    order = s.sort_values("protein").reset_index(drop=True)
    ypos = np.arange(len(order))
    h = 0.36
    ax.barh(ypos - h / 2, order["atom_base_rate"], height=h,
            color=COLORS["secondary"], label="Ligand atoms")
    ax.barh(ypos + h / 2, order["residue_base_rate"], height=h,
            color=COLORS["primary"], label="Protein residues")
    for i, r in order.iterrows():
        ax.text(r["atom_base_rate"] - 0.02, i - h / 2,
                f"{r['atom_base_rate']:.3f}", va="center", ha="right",
                fontsize=6.5, color="white")
        ax.text(r["residue_base_rate"] + 0.02, i + h / 2,
                f"{r['residue_base_rate']:.3f}", va="center", ha="left",
                fontsize=6.5, color=COLORS["dark"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.protein} ({r.pdb_id})" for r in order.itertuples()],
                       fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of tokens in contact (base rate)")
    ax.set_xlim(0, 1.15)
    ax.axvline(1.0, color=COLORS["grey"], lw=0.8, ls=":")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=6.5)
    n_sat = int((order["atom_label_classes"] == 1).sum())
    ax.set_title(f"Atom labels are single-class in {n_sat} of {len(order)} complexes",
                 fontsize=7.5)

    # (b) disjoint class supports
    ax = axes[1]
    pos = c.loc[c["panel_role"] == "holdout", "NearestAnchorTanimoto"].dropna()
    neg = c.loc[c["panel_role"] != "holdout", "NearestAnchorTanimoto"].dropna()
    bins = np.linspace(0, 1.0, 51)
    ax.hist(neg, bins=bins, color=COLORS["primary"], alpha=0.85,
            label=f"Decoys (n={len(neg):,})")
    ax.hist(pos, bins=bins, color=COLORS["secondary"], alpha=0.85,
            label=f"Positives (n={len(pos):,})")
    gap_lo, gap_hi = float(neg.max()), float(pos.min())
    ax.axvspan(gap_lo, gap_hi, color=COLORS["grey"], alpha=0.25, zorder=0)
    ax.axvline(0.40, color=COLORS["dark"], lw=0.9, ls="--")
    ax.set_xlabel("Nearest-anchor ECFP Tanimoto similarity")
    ax.set_ylabel("Compounds")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_title(f"Empty interval of width {gap_hi - gap_lo:.3f}", fontsize=7.5)

    panel_labels(axes, x=-0.14)
    small = check_min_font(fig)
    if small:
        print("   warning, small text:", small[:3])
    print("  wrote", save(fig, "fig_benchmark_validity")[0].name)


# ---------------------------------------------------------------------------
def fig_efficiency() -> None:
    src = RESULTS / "efficiency_profile.json"
    if not src.exists():
        return _skip("fig_efficiency", f"missing {src.name}")
    prof = json.loads(src.read_text())

    fig, axes = matplotlib.pyplot.subplots(
        1, 3, figsize=(7.16, 2.3), constrained_layout=True)

    # (a) parameters by module
    ax = axes[0]
    by = prof["parameters"]["by_module"]
    pretty = {"gat": "Ligand graph\nencoder", "protein_adapter": "Protein adapter",
              "fusion": "Cross-attention\nfusion", "affinity_head": "Affinity head",
              "diffusion": "Diffusion head", "other": "Other"}
    items = [(k, v) for k, v in by.items() if v > 0]
    items.sort(key=lambda kv: -kv[1])
    names = [pretty.get(k, k.replace("_", " ")) for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float) / 1e6
    ax.barh(np.arange(len(vals)), vals, color=COLORS["primary"], height=0.6)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters (millions)")
    ax.set_title(f"{prof['parameters']['total_parameters']/1e6:.1f} M total",
                 fontsize=7.5)

    # (b) throughput by batch size, with and without maps
    ax = axes[1]
    tp = pd.DataFrame(prof["throughput"])
    for i, collect in enumerate([False, True]):
        sub = tp[tp["collect_attention"] == collect].sort_values("batch_size")
        if not len(sub):
            continue
        ax.plot(sub["batch_size"], sub["pairs_per_second"],
                marker=MARKERS[i], color=[COLORS["primary"], COLORS["secondary"]][i],
                label="With interaction maps" if collect else "Affinity only")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Pairs per batch")
    ax.set_ylabel("Pairs per second (CPU)")
    ax.legend(fontsize=6.5, loc="lower left", frameon=True, framealpha=0.9,
              borderpad=0.3, handlelength=1.6)
    ax.set_title("Inference throughput", fontsize=7.5)

    # (c) scaling with protein length
    ax = axes[2]
    lat = RESULTS / "efficiency_latency.csv"
    if lat.exists():
        d = pd.read_csv(lat)
        d1 = d[d["batch_size"] == 1]
        if len(d1) > 5:
            ax.plot(d1["max_protein_tokens"], 1000 * d1["seconds"], MARKERS[0],
                    color=COLORS["tertiary"], ls="none", markersize=2.5, alpha=0.6)
            x = d1["max_protein_tokens"].to_numpy(float)
            y = 1000 * d1["seconds"].to_numpy(float)
            if x.std() > 0:
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, a + b * xs, color=COLORS["dark"], lw=1.0, ls="--")
            ax.set_xlabel("Protein length (residues)")
            ax.set_ylabel("Latency per pair (ms)")
            ax.set_title("Scaling with target length", fontsize=7.5)

    panel_labels(axes, x=-0.22)
    print("  wrote", save(fig, "fig_efficiency")[0].name)


# ---------------------------------------------------------------------------
def fig_interaction_content() -> None:
    src = RESULTS / "contact_information_summary.json"
    stats = RESULTS / "contact_information_profile_stats.csv"
    npy = RESULTS / "panel_attention_residue.npy"
    if not src.exists():
        return _skip("fig_interaction_content", f"missing {src.name}")
    res = json.loads(src.read_text())
    st = pd.read_csv(stats) if stats.exists() else None

    fig, axes = matplotlib.pyplot.subplots(
        1, 3, figsize=(7.16, 2.6), constrained_layout=True)

    # (a) variance decomposition
    ax = axes[0]
    vd = res["variance_decomposition"]
    parts = [("Residue\nmain effect", vd["frac_residue_main_effect"]),
             ("Ligand\nmain effect", vd["frac_ligand_main_effect"]),
             ("Ligand x residue\ninteraction", vd["frac_ligand_by_residue_interaction"])]
    cols = [COLORS["primary"], COLORS["grey"], COLORS["secondary"]]
    ax.bar(np.arange(3), [100 * p for _, p in parts], color=cols, width=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([n for n, _ in parts], fontsize=6.5)
    ax.set_ylabel("Share of attention variance (%)")
    ax.set_title("Where the map's variation lives", fontsize=7.5)
    for i, (_, p) in enumerate(parts):
        ax.text(i, 100 * p, f"{100*p:.1f}", ha="center", va="bottom", fontsize=6.5)

    # (b) profile similarity to the panel mean
    ax = axes[1]
    if st is not None and "corr_to_mean_profile" in st:
        for lab, name, col in ((0, "Decoys", COLORS["primary"]),
                               (1, "Actives", COLORS["secondary"])):
            v = st.loc[st["label"] == lab, "corr_to_mean_profile"].dropna()
            if len(v):
                ax.hist(v, bins=40, color=col, alpha=0.8, label=name, density=True)
        ax.set_xlabel("Correlation with the panel mean profile")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("Every ligand gets nearly the same map", fontsize=7.5)

    # (c) what can be recovered from the map
    ax = axes[2]
    cl = res["classification"]
    # Colour encodes the three groups the caption distinguishes, not one hue per
    # bar: what the map supports, the controls it is tested against, and the
    # zero-shot rankers that are not comparable to either.  Every bar also
    # carries its own axis label, so colour is a grouping cue rather than the
    # only channel.
    bars = [
        ("Attention profile", cl["auroc_from_attention_profile"], COLORS["secondary"]),
        ("Profile, size removed",
         cl.get("auroc_from_profile_residualized_on_descriptors",
                cl.get("auroc_from_profile_residualised_on_descriptors", np.nan)),
         COLORS["secondary"]),
        ("Descriptors only", cl["auroc_from_descriptors"], COLORS["grey"]),
        ("Shuffled labels", cl["auroc_attention_shuffled_null"], COLORS["grey"]),
        ("Model score", cl["auroc_model_score"], COLORS["primary"]),
        ("Nearest-active ECFP", cl["auroc_ecfp_baseline"], COLORS["primary"]),
    ]
    y = np.arange(len(bars))
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62)
    for i, b in enumerate(bars):
        if np.isfinite(b[1]):
            ax.text(b[1] - 0.015, i, f"{b[1]:.3f}", va="center", ha="right",
                    fontsize=6.5, color="white")
    ax.axvline(0.5, color=COLORS["dark"], lw=0.8, ls="--")
    # Separate the supervised probes from the zero-shot rankers, which are not
    # comparable to them.
    ax.axhline(3.5, color=COLORS["grey"], lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Active vs decoy AUROC")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.set_title("Recovering binder status", fontsize=7.5)

    panel_labels(axes, x=-0.22)
    print("  wrote", save(fig, "fig_interaction_content")[0].name)


# ---------------------------------------------------------------------------
def fig_docking() -> None:
    src = RESULTS / "docking_retrieval_summary.json"
    scores = RESULTS / "docking" / "vina_scores.csv"
    if not src.exists():
        return _skip("fig_docking", f"missing {src.name}")
    res = json.loads(src.read_text())

    fig, axes = matplotlib.pyplot.subplots(
        1, 2, figsize=(7.16, 2.4), constrained_layout=True)

    # (a) ROC curves
    ax = axes[0]
    from sklearn.metrics import roc_curve
    scored_csv = RESULTS / "docking_panel_scored.csv"
    if scored_csv.exists():
        m = pd.read_csv(scored_csv)
        n_act = int(m["label"].sum())
        series = [("AutoDock Vina", m["vina"], COLORS["tertiary"]),
                  ("DeepDTA-iBAM", m["deepdta_ibam"], COLORS["primary"]),
                  ("Nearest-active ECFP", m["ecfp_nearest_active"], COLORS["secondary"])]
        for i, (name, s, col) in enumerate(series):
            fpr, tpr, _ = roc_curve(m["label"], s)
            auc = res["metrics"][["vina", "deepdta_ibam",
                                  "ecfp_nearest_active"][i]]["auroc"]
            ax.plot(fpr, tpr, color=col, ls=["-", "--", "-."][i],
                    label=f"{name} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], color=COLORS["grey"], lw=0.8, ls=":")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.set_title(f"In-domain EGFR retrieval "
                     f"({n_act} actives, {len(m) - n_act} decoys)", fontsize=7.5)

    # (b) generated analogs vs seed
    ax = axes[1]
    if scores.exists():
        d = pd.read_csv(scores)
        g = d[(d["set"] == "generated_analog") & (d["status"] == "ok")]
        seed = g[g["ident"] == "dasatinib_seed"]["vina_kcal"]
        rest = g[g["ident"] != "dasatinib_seed"]["vina_kcal"]
        if len(rest):
            ax.hist(rest, bins=20, color=COLORS["primary"], alpha=0.85,
                    label=f"Diffusion analogs (n={len(rest)})")
            if len(seed):
                ax.axvline(float(seed.iloc[0]), color=COLORS["secondary"],
                           lw=1.2, ls="--", label="Dasatinib seed")
            ax.set_xlabel("Vina score (kcal/mol, lower is better)")
            ax.set_ylabel("Analogs")
            ax.legend(fontsize=6.5, loc="upper left")
            ax.set_title("Orthogonal scoring of generated analogs", fontsize=7.5)

    panel_labels(axes, x=-0.14)
    print("  wrote", save(fig, "fig_docking")[0].name)


# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    """Block diagram of the architecture, drawn to the shared figure style.

    The previous version of this figure distinguished node classes by fill
    colour alone, using a green-versus-red pairing that is the classic
    red-green confusion, and carried no legend.  It was also a tall portrait
    panel placed in a single column, so its text printed at roughly 5 pt.  This
    version is landscape at full text width, encodes node class by border style
    as well as colour, and states the encoding in a legend.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = matplotlib.pyplot.subplots(figsize=(FULL_WIDTH, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 58)
    ax.axis("off")

    # class -> (facecolor, edgecolor, linestyle)
    # Node class is carried by border colour, border style and the legend.  The
    # three coloured classes use the blue/vermillion/purple trio that survives
    # all three colour-vision deficiencies; input is achromatic with a dashed
    # border, so it needs no hue at all.
    CLS = {
        "input":  ("#FFFFFF", COLORS["dark"],      "--"),
        "module": ("#CFE6F5", COLORS["primary"],   "-"),
        "state":  ("#FBDECB", COLORS["secondary"], "-"),
        "output": ("#F2DCE8", COLORS["tertiary"],  "-"),
    }

    def box(x, y, w, h, text, cls, bold=False, fs=6.6):
        fc, ec, ls = CLS[cls]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=0.9, linestyle=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3,
                fontweight="bold" if bold else "normal", linespacing=1.25)
        return (x, y, w, h)

    def arrow(a, b, side="v"):
        if side == "v":
            xy = (a[0] + a[2] / 2, a[1]); xytext = (b[0] + b[2] / 2, b[1] + b[3])
        else:
            xy = (a[0], a[1] + a[3] / 2); xytext = (b[0] + b[2], b[1] + b[3] / 2)
        ax.add_patch(FancyArrowPatch(
            xytext, xy, arrowstyle="-|>", mutation_scale=7,
            linewidth=0.8, color=COLORS["grey"], zorder=1,
            shrinkA=1.5, shrinkB=1.5))

    # --- ligand branch (left) ---------------------------------------------
    l1 = box(2, 46, 22, 8,  "Ligand SMILES", "input", bold=True)
    l2 = box(2, 34, 22, 8,  "RDKit atom-bond graph", "module")
    l3 = box(2, 22, 22, 8,  "Graph attention encoder\n6 layers, 8 heads", "module")
    l4 = box(2, 11, 22, 7,  "Atom tokens", "state")
    for a, b in ((l2, l1), (l3, l2), (l4, l3)):
        arrow(a, b)

    # --- protein branch (right) -------------------------------------------
    r1 = box(76, 46, 22, 8, "Protein sequence", "input", bold=True)
    r2 = box(76, 34, 22, 8, "Cached ESM-C\nresidue embeddings", "module")
    r3 = box(76, 22, 22, 8, "Protein adapter", "module")
    r4 = box(76, 11, 22, 7, "Residue tokens", "state")
    for a, b in ((r2, r1), (r3, r2), (r4, r3)):
        arrow(a, b)

    # --- fusion (centre) ---------------------------------------------------
    f = box(31, 22, 38, 12,
            "Bidirectional cross-attention fusion\n"
            r"3 blocks, gated: $A \leftarrow A + g_a\tilde{A}$",
            "module", bold=True, fs=7.0)
    arrow(f, l4, side="h")          # atom tokens -> fusion
    ax.add_patch(FancyArrowPatch(
        (r4[0], r4[1] + r4[3] / 2), (f[0] + f[2], f[1] + f[3] / 2),
        arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
        color=COLORS["grey"], zorder=1, shrinkA=1.5, shrinkB=1.5))

    shared = box(31, 11, 38, 7, "Shared state", "state", bold=True, fs=7.0)
    arrow(shared, f)

    # --- four heads from one state ----------------------------------------
    outs = [
        (1.8,  "Affinity\nprediction"),
        (26.2, "Interaction map\n(iBAM)"),
        (50.6, "Latent\nretrieval"),
        (75.0, "Analog generation\n(diffusion head)"),
    ]
    # Routed as a bus rather than four diagonals: a diagonal drawn behind the
    # rounded boxes leaves its arrowhead hidden under the border, which reads as
    # a stray stub above the outer boxes.
    PAD = 0.35          # the rounded boxstyle pad, so edges sit at y +/- PAD
    BUS = 9.1           # height of the horizontal distribution line
    boxes = [box(x, 0.0, 22.4, 7.5, label, "output") for x, label in outs]
    centers = [o[0] + o[2] / 2 for o in boxes]
    ax.plot([shared[0] + shared[2] / 2] * 2, [shared[1] - PAD, BUS],
            color=COLORS["grey"], linewidth=0.8, zorder=1)
    ax.plot([min(centers), max(centers)], [BUS, BUS],
            color=COLORS["grey"], linewidth=0.8, zorder=1)
    for o, cx in zip(boxes, centers):
        ax.add_patch(FancyArrowPatch(
            (cx, BUS), (cx, o[1] + o[3] + PAD),
            arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
            color=COLORS["grey"], zorder=1, shrinkA=0, shrinkB=0))

    ax.text(50, 55.5, "One trained checkpoint serves all four tasks",
            ha="center", va="center", fontsize=7.2, style="italic",
            color=COLORS["dark"])

    handles = [
        mpatches.Patch(facecolor=CLS["input"][0], edgecolor=CLS["input"][1],
                       linestyle="--", linewidth=0.9, label="Input"),
        mpatches.Patch(facecolor=CLS["module"][0], edgecolor=CLS["module"][1],
                       linewidth=0.9, label="Learned module"),
        mpatches.Patch(facecolor=CLS["state"][0], edgecolor=CLS["state"][1],
                       linewidth=0.9, label="Representation"),
        mpatches.Patch(facecolor=CLS["output"][0], edgecolor=CLS["output"][1],
                       linewidth=0.9, label="Task output"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.075),
              ncol=4, fontsize=6.6, frameon=False, handlelength=1.4,
              handleheight=0.9, columnspacing=1.4)

    small = check_min_font(fig)
    if small:
        print("   warning, small text:", small[:3])
    print("  wrote", save(fig, "fig_architecture")[0].name)


def main() -> None:
    apply_style()
    print("building figures")
    for fn in (fig_architecture, fig_localization_forest, fig_benchmark_validity,
               fig_efficiency, fig_interaction_content, fig_docking):
        try:
            fn()
        except Exception as exc:  # keep going, report what failed
            print(f"  [fail] {fn.__name__}: {type(exc).__name__}: {exc}")
    print("done")


if __name__ == "__main__":
    main()
