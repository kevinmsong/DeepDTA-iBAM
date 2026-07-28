"""
Revision analyses for the J. Cheminform. submission.

Runs four analyses that are not part of the original pipeline and writes the
LaTeX tables consumed by main.tex:

  1. Meta-analysis of residue contact AUROC across the 5-complex panel, using
     analytic Hanley-McNeil standard errors instead of a t-test on 5 values.
     Reports fixed-effect and random-effects (DerSimonian-Laird) pooling and
     per-complex confidence intervals.
  2. Class-conditional nearest-anchor similarity distributions for the EGFR
     retrieval panel, quantifying the separation that makes fingerprint
     baselines inadmissible.
  3. Welch tests on the archived component ablations (3 seeds per variant).
  4. Paired bootstrap and Mann-Whitney tests on the generator comparison.

Run from the submission directory:  python analysis_revision.py
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
import numpy as np
from math import erfc, sqrt
from scipy import stats

RES = str(_ROOT / "results") + "/"
DATA = str(_ROOT / "data") + "/"
Z = 1.959964


def hanley_mcneil_se(auroc, n_pos, n_neg):
    """Analytic standard error of AUROC (Hanley & McNeil, Radiology 1982)."""
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    var = (auroc * (1 - auroc)
           + (n_pos - 1) * (q1 - auroc ** 2)
           + (n_neg - 1) * (q2 - auroc ** 2)) / (n_pos * n_neg)
    return sqrt(var)


def two_sided_p(z):
    return erfc(abs(z) / sqrt(2))


# ---------------------------------------------------------------------------
# 1. Interpretability meta-analysis
# ---------------------------------------------------------------------------
def interpretability_meta():
    rows = list(csv.DictReader(open(RES + "interpretability_benchmark.csv")))
    notes = {d["pdb_id"]: d for d in json.load(open(DATA + "interpretability_benchmark.json"))}

    # Binding mode assigned from the panel annotations; imatinib (2HYY) and the
    # annotated DFG-out ligand (4RJ3) are type II, the remainder are type I.
    mode = {"2HYY": "II", "4RJ3": "II", "1KE6": "I", "4WKQ": "I", "6YOJ": "I"}

    per = []
    for x in rows:
        a = float(x["residue_contact_auroc"])
        npos = int(x["n_contact_residues"])
        nneg = int(x["n_residues"]) - npos
        se = hanley_mcneil_se(a, npos, nneg)
        per.append(dict(pdb=x["pdb_id"], prot=x["protein"], auroc=a, se=se,
                        npos=npos, nneg=nneg, lo=a - Z * se, hi=a + Z * se,
                        mode=mode[x["pdb_id"]]))

    a = np.array([p["auroc"] for p in per])
    se = np.array([p["se"] for p in per])
    w = 1.0 / se ** 2

    fe = float((w * a).sum() / w.sum())
    se_fe = sqrt(1.0 / w.sum())
    q = float((w * (a - fe) ** 2).sum())
    k = len(a)
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))

    ws = 1.0 / (se ** 2 + tau2)
    re = float((ws * a).sum() / ws.sum())
    se_re = sqrt(1.0 / ws.sum())

    out = dict(per=per, fe=fe, se_fe=se_fe, p_fe=two_sided_p((fe - 0.5) / se_fe),
               re=re, se_re=se_re, p_re=two_sided_p((re - 0.5) / se_re),
               q=q, i2=i2, tau2=tau2,
               p_het=two_sided_p(sqrt(max(q, 0))) if q > 0 else 1.0)

    with open(_out("table_interpretability_meta.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llrrccl}\n\\toprule\n")
        fh.write("PDB & Protein & Contacts & Non-contacts & AUROC & 95\\% CI & Binding mode \\\\\n\\midrule\n")
        for p in per:
            star = "$^{*}$" if p["lo"] > 0.5 else ""
            fh.write(f"{p['pdb']} & {p['prot']} & {p['npos']} & {p['nneg']} & "
                     f"{p['auroc']:.3f}{star} & {p['lo']:.3f}--{p['hi']:.3f} & Type {p['mode']} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{4}}{{l}}{{Fixed-effect pooled}} & {fe:.3f} & "
                 f"{fe - Z * se_fe:.3f}--{fe + Z * se_fe:.3f} & $p = {out['p_fe']:.4f}$ \\\\\n")
        fh.write(f"\\multicolumn{{4}}{{l}}{{Random-effects pooled}} & {re:.3f} & "
                 f"{re - Z * se_re:.3f}--{re + Z * se_re:.3f} & $p = {out['p_re']:.3f}$ \\\\\n")
        fh.write(f"\\multicolumn{{7}}{{l}}{{Heterogeneity: $Q = {q:.2f}$ (df $= 4$), "
                 f"$I^2 = {i2:.0f}\\%$, $\\tau^2 = {tau2:.4f}$}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return out


# ---------------------------------------------------------------------------
# 2. EGFR panel class-conditional similarity separation
# ---------------------------------------------------------------------------
def retrieval_separation():
    rows = list(csv.DictReader(open(RES + "egfr_interpolation_ranked_candidates.csv")))
    y = np.array([1 if x["panel_role"] == "holdout" else 0 for x in rows])
    tan = np.array([float(x["NearestAnchorTanimoto"]) for x in rows])
    pos, neg = tan[y == 1], tan[y == 0]
    return dict(pos_min=pos.min(), pos_med=float(np.median(pos)), pos_max=pos.max(),
                neg_min=neg.min(), neg_med=float(np.median(neg)), neg_max=neg.max(),
                gap=pos.min() - neg.max(), n_pos=len(pos), n_neg=len(neg),
                n_decoys_above_pos_min=int((neg >= pos.min()).sum()),
                n_pos_below_decoy_max=int((pos <= neg.max()).sum()))


# ---------------------------------------------------------------------------
# 3. Component ablation tests
# ---------------------------------------------------------------------------
LABEL = {"abl_no_fusion": "No cross-attention", "abl_no_ranking": "No ranking loss",
         "abl_no_diffusion": "No diffusion head", "abl_base": "Backbone only"}


def ablation_tests():
    r = {(x["model_name"], x["split_type"]): x
         for x in csv.DictReader(open(RES + "ablation_table_wide.csv"))}
    res = {}
    with open(_out("table_ablation_tests.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llccrc}\n\\toprule\n")
        fh.write("Split & Variant & CI (mean $\\pm$ SD) & $\\Delta$ vs full & Cohen's $d$ & $p$ \\\\\n\\midrule\n")
        for split in ["standard", "scaffold"]:
            f = r[("abl_full", split)]
            m1, s1, n1 = float(f["CI_mean"]), float(f["CI_sd"]), int(f["CI_n"])
            fh.write(f"{split.capitalize()} & Full model & {m1:.4f} $\\pm$ {s1:.4f} & -- & -- & -- \\\\\n")
            for v in ["abl_no_fusion", "abl_no_ranking", "abl_no_diffusion", "abl_base"]:
                g = r[(v, split)]
                m2, s2, n2 = float(g["CI_mean"]), float(g["CI_sd"]), int(g["CI_n"])
                t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
                pooled = sqrt((s1 ** 2 + s2 ** 2) / 2)
                d = (m1 - m2) / pooled
                res[(split, v)] = (m1 - m2, d, p)
                fh.write(f" & {LABEL[v]} & {m2:.4f} $\\pm$ {s2:.4f} & {m1 - m2:+.4f} & {d:+.2f} & {p:.2f} \\\\\n")
            if split == "standard":
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return res


# ---------------------------------------------------------------------------
# 4. Generator comparison
# ---------------------------------------------------------------------------
def generation_tests(n_boot=10000, seed=1337):
    rows = list(csv.DictReader(open(RES + "generation_comparison.csv")))
    g = {}
    for x in rows:
        g.setdefault(x["generator"], []).append(float(x["PredAffinity"]))
    g = {k: np.array(v) for k, v in g.items()}
    rng = np.random.default_rng(seed)
    d = g["diffusion"]
    res = {}
    with open(_out("table_generation_tests.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrccrc}\n\\toprule\n")
        fh.write("Comparison & $n$ & Mean affinity & $\\Delta$ vs diffusion & 95\\% CI of $\\Delta$ & $p$ \\\\\n\\midrule\n")
        fh.write(f"Diffusion & {len(d)} & {d.mean():.4f} & -- & -- & -- \\\\\n")
        for base in ["random_edit", "fragment_swap"]:
            b = g[base]
            obs = d.mean() - b.mean()
            boot = np.array([rng.choice(d, len(d), True).mean() - rng.choice(b, len(b), True).mean()
                             for _ in range(n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            _, p = stats.mannwhitneyu(d, b, alternative="greater")
            res[base] = (obs, lo, hi, p)
            name = base.replace("_", " ").capitalize()
            fh.write(f"{name} & {len(b)} & {b.mean():.4f} & {obs:+.4f} & "
                     f"{lo:+.4f} to {hi:+.4f} & {p:.4f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return res


if __name__ == "__main__":
    meta = interpretability_meta()
    sep = retrieval_separation()
    abl = ablation_tests()
    gen = generation_tests()

    print("INTERPRETABILITY META-ANALYSIS")
    print(f"  fixed-effect  {meta['fe']:.4f} (p={meta['p_fe']:.4f})")
    print(f"  random-effect {meta['re']:.4f} (p={meta['p_re']:.3f})  I2={meta['i2']:.0f}%")
    print(f"  individually significant: "
          f"{[p['pdb'] for p in meta['per'] if p['lo'] > 0.5]}")
    print("\nRETRIEVAL PANEL SEPARATION")
    print(f"  positives {sep['pos_min']:.3f}-{sep['pos_max']:.3f}, "
          f"decoys {sep['neg_min']:.3f}-{sep['neg_max']:.3f}, gap {sep['gap']:.3f}")
    print("\nGENERATION")
    for k, (o, lo, hi, p) in gen.items():
        print(f"  vs {k:14s} delta={o:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4g}")
    print("\nABLATION: min p =", min(v[2] for v in abl.values()).round(3))
