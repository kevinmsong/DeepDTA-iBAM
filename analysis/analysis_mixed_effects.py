"""
Residue-level mixed-effects analysis of attention localization.

The main text originally summarized the structural-localization panel by pooling
one AUROC per complex (Hanley--McNeil inverse-variance weighting) and then, post
hoc, noticing that the two complexes whose intervals excluded chance were both
type II DFG-out binders.  Pooling discards the residue-level structure, and the
2-vs-3 split is a comparison of five numbers.

This module replaces that with a model fit to the residues themselves:

    contact_ij ~ 1 + r_ij * mode_j            (fixed effects)
                   + (1 + r_ij | complex_j)   (random effects)

where contact_ij is the crystallographic contact label of residue i in complex
j, r_ij is that residue's within-complex attention rank, and mode_j is the
inhibitor binding mode.  The coefficient on r is the panel-level localization
effect with between-complex variation modelled rather than averaged away; the
r x mode interaction is the type I / type II hypothesis tested directly rather
than read off two significant complexes.

Attention is used on the within-complex rank scale, standardized, rather than
raw.  Residue attention is severely right-skewed (skewness 5.6 to 9.8,
excess kurtosis 40 to 107), so a raw-scale logit coefficient is driven by a
handful of extreme residues and does not track the rank statistic the benchmark
actually reports: on 6YOJ the raw-scale slope is positive (+0.073) while the
AUROC is 0.487.  On the rank scale the per-complex slopes reproduce the ordering
of the per-complex AUROCs exactly.  The raw-scale fit is retained as a
sensitivity analysis.

Fit by variational Bayes (statsmodels BinomialBayesMixedGLM), because a
5-cluster Laplace GLMM is not reliably identified.

The module also reports the exact permutation limit: with 2 type II complexes
among 5, only C(5,2) = 10 assignments of the binding-mode label exist, so the
smallest attainable one-sided permutation p value is 0.10.  No panel of this
size can establish the binding-mode effect at conventional thresholds,
whatever the true effect.

Outputs
-------
  table_mixed_effects.tex      fixed-effect estimates (main text)
  mixed_effects_summary.json   machine-readable results

Run from the submission directory:  python analysis_mixed_effects.py
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

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

SRC = str(_ROOT / "results" / "interpretability_residue_level.csv")
Z = 1.959964


def load():
    d = pd.read_csv(SRC)
    d["mode_II"] = (d["binding_mode"] == "II").astype(float)
    # Within-complex attention rank, standardized. See module docstring for why
    # the rank scale rather than the raw scale.
    d["attention_rank"] = d.groupby("pdb_id")["attention"].transform(
        lambda s: stats.rankdata(s) / len(s)
    )
    d["attention_r"] = d.groupby("pdb_id")["attention_rank"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=1)
    )
    return d


# ---------------------------------------------------------------------------
# 1. Mixed-effects logistic regression
# ---------------------------------------------------------------------------
def fit_glmm(d, predictor="attention_r"):
    """Random intercept and random slope of attention, both grouped by complex."""
    vcf = {"complex": "0 + C(pdb_id)",
           "complex_slope": f"0 + C(pdb_id):{predictor}"}
    model = BinomialBayesMixedGLM.from_formula(
        f"contact ~ {predictor} * mode_II", vcf, d, vcp_p=2.0, fe_p=2.0
    )
    res = model.fit_vb(verbose=False)

    names = list(res.model.exog_names)
    out = {}
    for i, nm in enumerate(names):
        m, s = float(res.fe_mean[i]), float(res.fe_sd[i])
        out[nm] = dict(mean=m, sd=s, lo=m - Z * s, hi=m + Z * s,
                       odds=float(np.exp(m)),
                       odds_lo=float(np.exp(m - Z * s)),
                       odds_hi=float(np.exp(m + Z * s)),
                       p=float(2 * stats.norm.sf(abs(m / s))))
    vc = {nm: float(np.exp(res.vcp_mean[i]))
          for i, nm in enumerate(res.model.vcp_names)}
    return res, out, vc


# ---------------------------------------------------------------------------
# 2. Exact permutation test on the binding-mode label
# ---------------------------------------------------------------------------
def permutation_limit(d):
    """
    Per-complex AUROC difference (type II minus type I), referred to the exact
    permutation distribution over all C(5,2) = 10 relabellings.
    """
    per = {p: roc_auc_score(g.contact, g.attention) for p, g in d.groupby("pdb_id")}
    pdbs = sorted(per)
    obs_ii = {p for p, g in d.groupby("pdb_id") if g.binding_mode.iloc[0] == "II"}

    def diff(sel):
        a = np.mean([per[p] for p in pdbs if p in sel])
        b = np.mean([per[p] for p in pdbs if p not in sel])
        return a - b

    obs = diff(obs_ii)
    null = [diff(set(c)) for c in combinations(pdbs, 2)]
    p_one = float(np.mean([v >= obs - 1e-12 for v in null]))
    return dict(per_complex=per, observed_diff=float(obs),
                null=[float(v) for v in null], n_perm=len(null),
                p_one_sided=p_one, min_attainable_p=1.0 / len(null))


# ---------------------------------------------------------------------------
# 3. Reproduction check against the published per-complex AUROCs
# ---------------------------------------------------------------------------
def reproduction_check(d):
    pub = pd.read_csv(str(_ROOT / "results" / "interpretability_benchmark.csv")).set_index("pdb_id")
    rows = []
    for p, g in d.groupby("pdb_id"):
        a = roc_auc_score(g.contact, g.attention)
        rows.append(dict(pdb=p, recomputed=float(a),
                         published=float(pub.loc[p, "residue_contact_auroc"])))
    return rows


# ---------------------------------------------------------------------------
def main():
    d = load()
    res, fe, vc = fit_glmm(d, "attention_r")
    _, fe_raw, vc_raw = fit_glmm(d, "attention_z")

    # Refit with type II as the reference level, so the attention-rank slope
    # within type II complexes is read off directly with its own interval
    # rather than summed from two correlated coefficients.
    d2 = d.copy()
    d2["mode_II"] = 1.0 - d2["mode_II"]          # now indicates type I
    _, fe_ii, _ = fit_glmm(d2, "attention_r")
    slope_ii = fe_ii["attention_r"]

    perm = permutation_limit(d)
    repro = reproduction_check(d)

    def pfmt(p):
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    keys = ["Intercept", "attention_r", "mode_II", "attention_r:mode_II"]
    label = {
        "Intercept": "Intercept (type I complexes)",
        "attention_r": "Attention rank (per SD)",
        "mode_II": "Binding mode II",
        "attention_r:mode_II": "Attention rank $\\times$ mode II",
    }
    with open(_out("table_mixed_effects.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrcc}\n\\toprule\n")
        fh.write("Term & Coefficient & Odds ratio & 95\\% interval (OR) & $p$ \\\\\n")
        fh.write("\\midrule\n")
        for k in keys:
            v = fe[k]
            fh.write(f"{label[k]} & {v['mean']:+.3f} & {v['odds']:.2f} & "
                     f"{v['odds_lo']:.2f}--{v['odds_hi']:.2f} & "
                     f"{pfmt(v['p'])} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\textit{{Attention-rank slope, type II}} & "
                 f"{slope_ii['mean']:+.3f} & {slope_ii['odds']:.2f} & "
                 f"{slope_ii['odds_lo']:.2f}--{slope_ii['odds_hi']:.2f} & "
                 f"{pfmt(slope_ii['p'])} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{5}}{{l}}{{Random effects by complex: "
                 f"intercept SD {vc.get('complex', float('nan')):.2f}, "
                 f"attention-slope SD {vc.get('complex_slope', float('nan')):.2f}}} \\\\\n")
        fh.write(f"\\multicolumn{{5}}{{l}}{{$n = {len(d)}$ residues in 5 complexes, "
                 f"{int(d.contact.sum())} annotated contacts}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    summary = dict(fixed_effects=fe, random_effects=vc,
                   slope_type_II=slope_ii,
                   fixed_effects_rawscale=fe_raw, random_effects_rawscale=vc_raw,
                   permutation=perm,
                   reproduction=repro, n_residues=int(len(d)),
                   n_contacts=int(d.contact.sum()))
    json.dump(summary, open("mixed_effects_summary.json", "w"), indent=2)

    print("MIXED-EFFECTS LOGISTIC REGRESSION (contact ~ r * mode + (1+r|complex))")
    for k in keys:
        v = fe[k]
        print(f"  {k:22s} b={v['mean']:+.4f} (SD {v['sd']:.4f})  "
              f"OR={v['odds']:.3f} [{v['odds_lo']:.3f},{v['odds_hi']:.3f}]  "
              f"p={v['p']:.4f}")
    print(f"  {'slope within type II':22s} b={slope_ii['mean']:+.4f} "
          f"(SD {slope_ii['sd']:.4f})  OR={slope_ii['odds']:.3f} "
          f"[{slope_ii['odds_lo']:.3f},{slope_ii['odds_hi']:.3f}]  "
          f"p={slope_ii['p']:.4g}")
    print(f"  random SDs: {vc}")
    print(f"  n = {len(d)} residues, {int(d.contact.sum())} contacts")
    print("  sensitivity, raw attention scale: "
          f"slope b={fe_raw['attention_z']['mean']:+.4f} "
          f"(p={fe_raw['attention_z']['p']:.3f}), "
          f"interaction b={fe_raw['attention_z:mode_II']['mean']:+.4f} "
          f"(p={fe_raw['attention_z:mode_II']['p']:.3f})")

    print("\nEXACT PERMUTATION LIMIT ON THE BINDING-MODE SPLIT")
    print(f"  per-complex AUROC: "
          + ", ".join(f"{k} {v:.3f}" for k, v in sorted(perm['per_complex'].items())))
    print(f"  observed type II - type I difference: {perm['observed_diff']:+.4f}")
    print(f"  exact one-sided p = {perm['p_one_sided']:.2f} "
          f"over {perm['n_perm']} relabellings")
    print(f"  smallest attainable p at this panel size = "
          f"{perm['min_attainable_p']:.2f}")

    print("\nREPRODUCTION CHECK vs PUBLISHED PER-COMPLEX AUROC")
    for r in repro:
        print(f"  {r['pdb']}  recomputed {r['recomputed']:.4f}  "
              f"published {r['published']:.4f}  "
              f"delta {r['recomputed'] - r['published']:+.4f}")


if __name__ == "__main__":
    main()
