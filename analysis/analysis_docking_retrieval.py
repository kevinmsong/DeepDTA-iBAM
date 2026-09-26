"""Retrieval metrics for the docking baseline against the learned scorer.

Consumes the per-ligand Vina scores written by analysis_docking_baseline.py and
compares three rankers on the identical in-domain EGFR candidate set:

  * AutoDock Vina, an established structure-based scoring function;
  * DeepDTA-iBAM, the learned scorer;
  * a nearest-active ECFP similarity baseline.

Vina reports binding free energy, so its score is negated before ranking.

All three see the same candidates, so their errors are correlated and comparing
marginal confidence intervals would misstate how well the differences are
resolved.  Head-to-head differences therefore use a paired bootstrap over the
shared candidate set.

Also summarizes the docking of the audited diffusion analogs against the
dasatinib seed, which is the orthogonal scorer the generation comparison
previously lacked.

Run from anywhere:  python analysis/analysis_docking_retrieval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCORES = RESULTS / "docking" / "vina_scores.csv"
PANEL = RESULTS / "dude_egfr_panel_scored.csv"
OUT_JSON = RESULTS / "docking_retrieval_summary.json"
OUT_TEX = RESULTS / "table_docking_retrieval.tex"

SEED = 1337
N_BOOT = 10000


def ecfp_nearest_active(smiles: list[str], y: np.ndarray) -> np.ndarray:
    """Leave-one-out nearest-active ECFP similarity within this candidate set.

    Each candidate is scored by its greatest Tanimoto similarity to an active
    other than itself, using a within-subset active reference. These known
    active labels provide reference information unavailable to the other rankers.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)

    active_pos = [i for i, lab in enumerate(y) if lab == 1 and fps[i] is not None]
    scores = np.zeros(len(smiles), dtype=float)
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        refs = [fps[j] for j in active_pos if j != i]
        if not refs:
            continue
        scores[i] = max(DataStructs.BulkTanimotoSimilarity(fp, refs))
    return scores


def expected_rank_labels(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Expected label at each rank over all orderings within equal-score ties.

    Threshold enrichment/recovery and BEDROC are linear in ranked labels,
    so block means give their exact averages over possible tied orderings.
    """
    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    sorted_labels = y[order].astype(float)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(y)])
    block_means = np.add.reduceat(sorted_labels, starts) / sizes
    return np.repeat(block_means, sizes)


def enrichment(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    n = max(1, int(round(len(y) * frac)))
    labels = expected_rank_labels(y, s)
    return (float(labels[:n].sum()) / n) / (y.mean() + 1e-12)


def recovery(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    n = max(1, int(round(len(y) * frac)))
    labels = expected_rank_labels(y, s)
    return float(labels[:n].sum()) / max(1.0, float(y.sum()))


def bedroc(y: np.ndarray, s: np.ndarray, alpha: float = 20.0) -> float:
    n = len(y)
    n_act = int(y.sum())
    if n_act == 0 or n_act == n:
        return float("nan")
    labels = expected_rank_labels(y, s)
    ranks = np.arange(1, n + 1)
    ra = n_act / n
    s_sum = float(np.sum(labels * np.exp(-alpha * ranks / n)))
    rie = s_sum / (ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    return float((rie * ra * np.sinh(alpha / 2) /
                  (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra)))
                 + 1.0 / (1 - np.exp(alpha * (1 - ra))))


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "bedroc20": bedroc(y, s),
        "ef_1pct": enrichment(y, s, 0.01),
        "ef_5pct": enrichment(y, s, 0.05),
        "recovery_10pct": recovery(y, s, 0.10),
    }


def paired_bootstrap(y, a, b, n_boot=N_BOOT, seed=SEED) -> dict:
    """Difference in AUROC (a minus b) on resamples of the shared candidates."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p value for no difference.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta_auroc_mean": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(max(p, 1.0 / max(len(diffs), 1))),
        "n_resamples": int(len(diffs)),
    }


def main() -> None:
    if not SCORES.exists():
        raise SystemExit(f"missing {SCORES}; run analysis_docking_baseline.py first")

    raw_scores = pd.read_csv(SCORES)
    d = raw_scores.copy()
    # A resumed run can append a ligand more than once; keep the first success.
    d = d[d["status"] == "ok"].drop_duplicates(subset=["set", "ident", "smiles"], keep="first")

    out: dict = {"seed": SEED, "n_bootstrap": N_BOOT,
                 "score_convention": "Vina reports binding free energy; negated for ranking",
                 "rank_tie_handling": "Enrichment, recovery, and BEDROC average over all orderings within tied-score groups; AUROC and average precision use scikit-learn tie handling."}

    # ------------------------------------------------------------ panel
    dp = d[d["set"] == "egfr_panel"]
    panel = pd.read_csv(PANEL)
    m = panel.merge(dp[["ident", "smiles", "vina_kcal"]], on=["ident", "smiles"], how="inner")
    attempted = set(map(tuple, raw_scores.loc[raw_scores["set"] == "egfr_panel",
                                             ["ident", "smiles"]].to_numpy()))
    successful = set(map(tuple, dp[["ident", "smiles"]].to_numpy()))

    out["panel"] = {
        "n_candidates_total": int(len(panel)),
        "n_docked": int(len(m)),
        "n_not_docked": int(len(panel) - len(m)),
        "n_attempted": len(attempted),
        "n_failed": len(attempted - successful),
        "n_actives_docked": int(m["label"].sum()),
        "n_decoys_docked": int((1 - m["label"]).sum()),
        "active_docking_rate": float(m["label"].sum() / max(panel["label"].sum(), 1)),
        "decoy_docking_rate": float((1 - m["label"]).sum() /
                                    max((1 - panel["label"]).sum(), 1)),
        "base_rate": float(m["label"].mean()),
    }

    y = m["label"].to_numpy(int)

    # The released ecfp_score is a leave-one-out nearest-active similarity
    # computed against all 300 panel actives. Recompute using only the actives
    # that were docked: the rankers share the same candidates, and the fingerprint
    # method uses a within-subset active reference. It still uses known active
    # labels unavailable to the other rankers. Here all 300 actives were docked,
    # so the reference sets coincide. Retain the full-panel score for comparison.
    ecfp_within = ecfp_nearest_active(m["smiles"].astype(str).tolist(), y)

    rankers = {
        "vina": -m["vina_kcal"].to_numpy(float),
        "deepdta_ibam": m["model_score"].to_numpy(float),
        "ecfp_nearest_active": ecfp_within,
        "ecfp_full_panel_reference": m["ecfp_score"].to_numpy(float),
    }
    out["metrics"] = {k: metrics(y, v) for k, v in rankers.items()}

    # Release the per-candidate scores so the figure and any re-analysis use
    # exactly the rankings these metrics were computed from.
    scored = m[["ident", "smiles", "label"]].copy()
    for k, v in rankers.items():
        scored[k] = v
    scored.to_csv(RESULTS / "docking_panel_scored.csv", index=False)

    out["paired"] = {
        "vina_minus_model": paired_bootstrap(y, rankers["vina"], rankers["deepdta_ibam"]),
        "ecfp_minus_model": paired_bootstrap(y, rankers["ecfp_nearest_active"],
                                             rankers["deepdta_ibam"]),
        "ecfp_minus_vina": paired_bootstrap(y, rankers["ecfp_nearest_active"],
                                            rankers["vina"]),
    }

    # ------------------------------------------------- generated analogs
    dg = d[d["set"] == "generated_analog"]
    if len(dg):
        seed_row = dg[dg["ident"] == "dasatinib_seed"]
        analogs = dg[dg["ident"] != "dasatinib_seed"]["vina_kcal"].to_numpy(float)
        g = {"n_analogs_docked": int(len(analogs))}
        if len(analogs):
            g.update({
                "analog_mean_kcal": float(analogs.mean()),
                "analog_sd_kcal": float(analogs.std(ddof=1)) if len(analogs) > 1 else 0.0,
                "analog_best_kcal": float(analogs.min()),
                "analog_median_kcal": float(np.median(analogs)),
            })
        if len(seed_row):
            sv = float(seed_row["vina_kcal"].iloc[0])
            g["dasatinib_seed_kcal"] = sv
            if len(analogs):
                g["n_analogs_better_than_seed"] = int((analogs < sv).sum())
                g["frac_analogs_better_than_seed"] = float((analogs < sv).mean())
                g["mean_delta_vs_seed"] = float(analogs.mean() - sv)
        out["generated_analogs"] = g

    # ---------------------------------------------------------- timing
    tim = d[d["set"] == "egfr_panel"]["seconds"].astype(float)
    if len(tim):
        out["docking_cost"] = {
            "median_seconds_per_ligand": float(tim.median()),
            "mean_seconds_per_ligand": float(tim.mean()),
            "note": "wall clock under 14-way parallelism on CPU",
        }

    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------- table
    nl = "\\\\"
    name = {"vina": "AutoDock Vina", "deepdta_ibam": "DeepDTA-iBAM",
            "ecfp_nearest_active": "Nearest-active ECFP",
            "ecfp_full_panel_reference": "ECFP, full-panel reference"}
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Ranker & AUROC & AUPRC & BEDROC20 & EF 1\% & EF 5\% & Rec.\ 10\% " + nl,
           r"\midrule"]
    for k in ("vina", "deepdta_ibam", "ecfp_nearest_active"):
        v = out["metrics"][k]
        tex.append(f"{name[k]} & {v['auroc']:.3f} & {v['auprc']:.3f} & "
                   f"{v['bedroc20']:.3f} & {v['ef_1pct']:.2f} & {v['ef_5pct']:.2f} & "
                   f"{v['recovery_10pct']:.3f} " + nl)
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    OUT_TEX.write_text("\n".join(tex))

    p = out["panel"]
    print(f"docked {p['n_docked']} of {p['n_candidates_total']} candidates "
          f"({p['n_actives_docked']} actives, {p['n_decoys_docked']} decoys); "
          f"base rate {p['base_rate']:.4f}")
    print(f"  active docking rate {p['active_docking_rate']:.3f}, "
          f"decoy docking rate {p['decoy_docking_rate']:.3f}")
    print("\n--- retrieval ---")
    for k in ("vina", "deepdta_ibam", "ecfp_nearest_active", "ecfp_full_panel_reference"):
        v = out["metrics"][k]
        print(f"  {name[k]:26s} AUROC {v['auroc']:.4f}  AUPRC {v['auprc']:.4f}  "
              f"EF1% {v['ef_1pct']:.2f}  Rec@10% {v['recovery_10pct']:.3f}")
    print("\n--- paired differences (AUROC) ---")
    for k, v in out["paired"].items():
        lo, hi = v["ci95"]
        print(f"  {k:22s} {v['delta_auroc_mean']:+.4f} "
              f"(95% CI {lo:+.4f} to {hi:+.4f}, p = {v['p_two_sided']:.4g})")
    if "generated_analogs" in out:
        g = out["generated_analogs"]
        print("\n--- generated analogs, docked ---")
        print(f"  {g['n_analogs_docked']} analogs, mean "
              f"{g.get('analog_mean_kcal', float('nan')):.3f} kcal/mol, "
              f"seed {g.get('dasatinib_seed_kcal', float('nan')):.3f}")
        if "frac_analogs_better_than_seed" in g:
            print(f"  better than the seed: {g['n_analogs_better_than_seed']} "
                  f"({g['frac_analogs_better_than_seed']:.1%})")
    print(f"\nwrote {OUT_JSON.name}, {OUT_TEX.name}")


if __name__ == "__main__":
    main()
