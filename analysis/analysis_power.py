"""
Power calculation for the structural-localization panel.

The manuscript asserts that "roughly 25 to 30 complexes would be needed to
separate an AUROC of 0.585 from 0.5 at 80% power".  This module derives that
number from the observed panel so a reader can check it, and reports the
sample sizes required for the binding-mode-stratified design as well.

Design being powered
--------------------
Complexes are the unit of replication.  Each complex j supplies one residue
contact AUROC A_j estimated from its own residues, so A_j carries a within-
complex sampling variance s_j^2 (Hanley--McNeil) on top of a between-complex
variance tau^2.  The panel estimate is the random-effects pooled AUROC, whose
variance with k complexes is approximately

    Var(A_pooled) = (tau^2 + s_bar^2) / k

where s_bar^2 is the typical within-complex sampling variance.  Testing
H0: A = 0.5 against a true value A1 at two-sided level alpha with power
1 - beta therefore needs

    k >= (z_{1-alpha/2} + z_{1-beta})^2 * (tau^2 + s_bar^2) / (A1 - 0.5)^2

Both variance components are estimated from the five observed complexes, so
this is a design calculation conditional on the observed panel, not an
independent prior.

Because tau^2 is estimated on 4 degrees of freedom it is itself very
imprecise, which is why the requirement is quoted as a range rather than a
single integer.  The range reported below spans the requirement computed from
the DerSimonian--Laird tau^2 and from the naive complex-level variance.

Outputs
-------
  table_power.tex          required panel size against target effect
  power_summary.json

Run from the submission directory:  python analysis_power.py
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
from math import ceil, sqrt

import numpy as np
from scipy import stats

RES = str(_ROOT / "results") + "/"
ALPHA = 0.05
POWER = 0.80


def hanley_mcneil_se(auroc, n_pos, n_neg):
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    var = (auroc * (1 - auroc)
           + (n_pos - 1) * (q1 - auroc ** 2)
           + (n_neg - 1) * (q2 - auroc ** 2)) / (n_pos * n_neg)
    return sqrt(var)


def variance_components():
    """tau^2 (DerSimonian--Laird) and mean within-complex variance."""
    rows = list(csv.DictReader(open(RES + "interpretability_benchmark.csv")))
    a, s2 = [], []
    for x in rows:
        auc = float(x["residue_contact_auroc"])
        npos = int(x["n_contact_residues"])
        nneg = int(x["n_residues"]) - npos
        a.append(auc)
        s2.append(hanley_mcneil_se(auc, npos, nneg) ** 2)
    a = np.array(a)
    s2 = np.array(s2)
    w = 1.0 / s2
    fe = (w * a).sum() / w.sum()
    q = (w * (a - fe) ** 2).sum()
    k = len(a)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    return dict(auroc=a.tolist(), s2=s2.tolist(), s2_bar=float(s2.mean()),
                tau2=float(tau2), q=float(q), k=k,
                fixed_effect=float(fe),
                naive_var=float(a.var(ddof=1)))


def required_k(effect, var_total, alpha=ALPHA, power=POWER):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) ** 2 * var_total / effect ** 2


def attained_power(k, effect, var_total, alpha=ALPHA):
    z_a = stats.norm.ppf(1 - alpha / 2)
    se = sqrt(var_total / k)
    return float(stats.norm.sf(z_a - effect / se))


def attained_power_t(k, effect, var_total, alpha=ALPHA):
    """
    Small-sample version: one-sample t test on k complex-level AUROCs, with the
    variance estimated from the same k complexes.  Power is computed from the
    noncentral t distribution.  This is the honest calculation when k is small
    enough that the normal approximation flatters the design, which is exactly
    the regime the panel sits in.
    """
    df = k - 1
    ncp = effect / sqrt(var_total / k)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp))


def required_k_t(effect, var_total, alpha=ALPHA, power=POWER, kmax=200):
    for k in range(3, kmax + 1):
        if attained_power_t(k, effect, var_total, alpha) >= power:
            return k
    return float("inf")


def main():
    vc = variance_components()
    tau2, s2_bar = vc["tau2"], vc["s2_bar"]
    var_re = tau2 + s2_bar               # random-effects variance per complex
    var_naive = vc["naive_var"]          # complex-level variance, ignores s_j^2

    effect = 0.585 - 0.5
    k_re = required_k(effect, var_re)
    k_naive = required_k(effect, var_naive)
    k_re_t = required_k_t(effect, var_re)
    k_naive_t = required_k_t(effect, var_naive)
    k_lo = min(ceil(k_re), ceil(k_naive))
    k_hi = max(k_re_t, k_naive_t)

    # Stratified design: the binding-mode contrast needs each stratum powered
    # for a difference between two group means, doubling the variance.
    delta_mode = 0.2013                  # observed type II minus type I
    k_per_stratum = required_k(delta_mode, 2 * var_re)

    # Grid for the table.
    grid = [5, 10, 15, 20, 25, 30, 40, 50]

    with open(_out("table_power.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rcccc}\n\\toprule\n")
        fh.write("Complexes $k$ & SE of pooled AUROC & 95\\% CI half-width & "
                 "Power (normal) & Power ($t$) \\\\\n\\midrule\n")
        for k in grid:
            se = sqrt(var_re / k)
            fh.write(f"{k} & {se:.4f} & {1.96 * se:.4f} & "
                     f"{attained_power(k, effect, var_re):.2f} & "
                     f"{attained_power_t(k, effect, var_re):.2f} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{5}}{{l}}{{Variance components from the observed "
                 f"panel: $\\tau^2 = {tau2:.5f}$, "
                 f"$\\bar{{s}}^2 = {s2_bar:.5f}$}} \\\\\n")
        fh.write(f"\\multicolumn{{5}}{{l}}{{Required for 80\\% power at "
                 f"$\\alpha = 0.05$: $k = {k_lo}$ to {k_hi} complexes}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    summary = dict(variance_components=vc, effect=effect,
                   var_random_effects=var_re, var_naive=var_naive,
                   k_random_effects=k_re, k_naive=k_naive,
                   k_random_effects_t=k_re_t, k_naive_t=k_naive_t,
                   k_range=[k_lo, k_hi],
                   delta_binding_mode=delta_mode,
                   k_per_stratum=k_per_stratum,
                   stratified={f"{d:.3f}": [ceil(required_k(d, 2 * var_re)),
                                            required_k_t(d, 2 * var_re)]
                               for d in [delta_mode, 0.15, 0.10, 0.05]},
                   power_grid={k: attained_power(k, effect, var_re) for k in grid},
                   power_grid_t={k: attained_power_t(k, effect, var_re) for k in grid})
    json.dump(summary, open("power_summary.json", "w"), indent=2)

    print("VARIANCE COMPONENTS (5-complex panel)")
    print(f"  per-complex AUROC        {[round(x, 3) for x in vc['auroc']]}")
    print(f"  mean within-complex s^2  {s2_bar:.5f}  (SE {sqrt(s2_bar):.4f})")
    print(f"  between-complex tau^2    {tau2:.5f}  (SD {sqrt(tau2):.4f})")
    print(f"  naive complex-level var  {var_naive:.5f}  (SD {sqrt(var_naive):.4f})")
    print(f"  total per-complex var    {var_re:.5f}")
    print("\nREQUIRED PANEL SIZE, 80% POWER, alpha = 0.05, effect = 0.085")
    print(f"  normal approx, RE variance    k = {k_re:.1f}  -> {ceil(k_re)}")
    print(f"  normal approx, naive variance k = {k_naive:.1f}  -> {ceil(k_naive)}")
    print(f"  noncentral t, RE variance     k -> {k_re_t}")
    print(f"  noncentral t, naive variance  k -> {k_naive_t}")
    print(f"  defensible range              {k_lo} to {k_hi} complexes")
    print(f"\n  power of the actual panel (k = 5): "
          f"{attained_power(5, effect, var_re):.2f} (normal), "
          f"{attained_power_t(5, effect, var_re):.2f} (t)")
    print("\nSTRATIFIED (type I vs type II) DESIGN, per stratum")
    print("  the observed 0.201 gap is a post hoc selected difference and is")
    print("  upward biased, so it gives a floor rather than a design target")
    for delta in [delta_mode, 0.15, 0.10, 0.05]:
        kk = required_k(delta, 2 * var_re)
        kk_t = required_k_t(delta, 2 * var_re)
        tag = " (observed)" if delta == delta_mode else ""
        print(f"  delta = {delta:.3f}{tag:11s}  k = {ceil(kk)} to {kk_t} per stratum")


if __name__ == "__main__":
    main()
