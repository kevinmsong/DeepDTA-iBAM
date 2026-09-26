"""Archived exploratory analysis of ligand-specific information in EGFR maps.

Current probe estimates are produced by revalidate_contact_probes.py and saved
in results/contact_information_revalidated.json. That script uses nested
cross-validation for ridge penalty selection and fits descriptor residualization
inside each outer training fold. This original script is retained to reproduce
the archived estimates; running it overwrites contact_information_summary.json.

The panel contains 300 assay-defined actives and 3,000 property-matched DUD-E
decoys against one fixed target. Decoys are presumed inactive, not experimentally
confirmed nonbinders. Attention profiles are model outputs, not measured contacts.

Three analyses are retained:

1. Descriptive variance decomposition and correlations among ligand profiles.
   The residue main effect describes the panel-mean profile at this fixed target;
   it does not establish that the target alone determines those values.
2. Ridge regression to the model's predicted affinity, not measured affinity.
   The best penalty is selected on the same five folds used for reporting, so
   the archived R2 is subject to selection optimism.
3. Fixed-penalty logistic probes for active/decoy labels, using stratified
   five-fold cross-validation with training-fold scaling. This is not nested
   cross-validation. The original descriptor residualization uses the full panel
   before cross-validation, so that estimate is exploratory. Residualization
   removes linear descriptor associations, not every possible property effect.

Consumes exports from export_panel_attention.py. DUD-E decoy bias and chemical
dependence within random candidate folds limit interpretation of all probes.

Current validation: python analysis/revalidate_contact_probes.py
Archived analysis:  python analysis/analysis_contact_information.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
NPY = RESULTS / "panel_attention_residue.npy"
IDX = RESULTS / "panel_attention_index.csv"
META = RESULTS / "panel_attention_meta.json"
OUT_JSON = RESULTS / "contact_information_summary.json"
OUT_CSV = RESULTS / "contact_information_profile_stats.csv"

SEED = 1337
N_FOLDS = 5
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------- descriptors
def descriptor_matrix(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """Cheap physicochemical descriptors, the trivial-feature control."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    names = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "Rings",
             "HeavyAtoms", "FracCSP3", "FormalCharge"]
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append([np.nan] * len(names))
            continue
        rows.append([
            Descriptors.MolWt(m),
            Crippen.MolLogP(m),
            rdMolDescriptors.CalcTPSA(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcNumRings(m),
            m.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(m),
            Chem.GetFormalCharge(m),
        ])
    X = np.asarray(rows, dtype=float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, names


# -------------------------------------------------------------- decomposition
def variance_decomposition(A: np.ndarray) -> dict:
    """Two-way decomposition of an N-ligand by R-residue attention matrix.

    Rows are attention distributions over residues, so the ligand main effect is
    near zero by construction and the informative comparison is between the
    residue main effect, which any ligand would produce, and the ligand-by-
    residue interaction, which is the only ligand-specific component.
    """
    N, R = A.shape
    grand = A.mean()
    res_means = A.mean(axis=0)        # (R,) profile any ligand would give
    lig_means = A.mean(axis=1)        # (N,) near-constant for distributions

    ss_total = float(((A - grand) ** 2).sum())
    ss_residue = float(N * ((res_means - grand) ** 2).sum())
    ss_ligand = float(R * ((lig_means - grand) ** 2).sum())
    ss_inter = float(ss_total - ss_residue - ss_ligand)

    return {
        "n_ligands": int(N),
        "n_residues": int(R),
        "ss_total": ss_total,
        "frac_residue_main_effect": ss_residue / ss_total if ss_total else float("nan"),
        "frac_ligand_main_effect": ss_ligand / ss_total if ss_total else float("nan"),
        "frac_ligand_by_residue_interaction": ss_inter / ss_total if ss_total else float("nan"),
    }


def profile_similarity(A: np.ndarray, rng: np.random.Generator) -> dict:
    mean_profile = A.mean(axis=0)
    # Correlation of each ligand profile with the panel mean.
    Ac = A - A.mean(axis=1, keepdims=True)
    mc = mean_profile - mean_profile.mean()
    num = Ac @ mc
    den = np.linalg.norm(Ac, axis=1) * np.linalg.norm(mc)
    corr_to_mean = num / np.where(den == 0, np.nan, den)

    # Pairwise correlation on a random subsample, which is enough to
    # characterize the distribution and avoids an N-by-N matrix.
    k = min(400, A.shape[0])
    sel = rng.choice(A.shape[0], size=k, replace=False)
    S = np.corrcoef(A[sel])
    iu = np.triu_indices(k, k=1)
    pair = S[iu]

    return {
        "corr_to_mean_profile_mean": float(np.nanmean(corr_to_mean)),
        "corr_to_mean_profile_min": float(np.nanmin(corr_to_mean)),
        "corr_to_mean_profile_p05": float(np.nanpercentile(corr_to_mean, 5)),
        "pairwise_corr_mean": float(np.nanmean(pair)),
        "pairwise_corr_p05": float(np.nanpercentile(pair, 5)),
        "pairwise_corr_min": float(np.nanmin(pair)),
        "pairwise_subsample": int(k),
    }, corr_to_mean


# ------------------------------------------------------------------ modelling
def _residualise(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Remove from X everything linearly predictable from the controls C."""
    Cs = StandardScaler().fit_transform(C)
    Cs = np.hstack([Cs, np.ones((len(Cs), 1))])
    beta, *_ = np.linalg.lstsq(Cs, X, rcond=None)
    return X - Cs @ beta


def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=0.1, max_iter=5000,
                           class_weight="balanced", random_state=seed),
    )
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def cv_r2(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    best = -np.inf
    for a in RIDGE_ALPHAS:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=a, random_state=seed))
        pred = cross_val_predict(pipe, X, y, cv=cv)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        best = max(best, 1.0 - ss_res / ss_tot)
    return float(best)


def main() -> None:
    if not NPY.exists():
        raise SystemExit(f"missing {NPY}; run analysis/export_panel_attention.py first")

    A = np.load(NPY).astype(np.float64)
    idx = pd.read_csv(IDX)
    meta = json.loads(META.read_text()) if META.exists() else {}
    if len(idx) != A.shape[0]:
        raise SystemExit("index and profile matrix are not aligned")

    y = idx["label"].to_numpy(int)
    affinity = idx["model_pred_rerun"].to_numpy(float)
    rng = np.random.default_rng(SEED)

    out: dict = {"seed": SEED, "n_folds": N_FOLDS, "export_meta": meta}

    # 1 --------------------------------------------------------- invariance
    out["variance_decomposition"] = variance_decomposition(A)
    sim, corr_to_mean = profile_similarity(A, rng)
    out["profile_similarity"] = sim

    # Keep only residues with some variation across ligands; constant columns
    # carry no information and inflate the feature count.
    col_sd = A.std(axis=0)
    keep = col_sd > 0
    Xatt = A[:, keep]
    out["n_residue_features_used"] = int(keep.sum())

    # 2 ------------------------------------------- affinity from the profile
    y_shuf = rng.permutation(affinity)
    out["affinity_regression"] = {
        "target": "model predicted affinity (panel, EGFR-conditioned)",
        "r2_from_attention_profile": cv_r2(Xatt, affinity),
        "r2_shuffled_null": cv_r2(Xatt, y_shuf),
    }

    # 3 ------------------------------ binder vs non-binder from the profile
    Xdesc, desc_names = descriptor_matrix(idx["smiles"].astype(str).tolist())
    lab_shuf = rng.permutation(y)

    # Does the map add anything beyond bulk physicochemical properties?  A
    # profile could be a proxy for ligand size, so compare descriptors alone
    # against descriptors plus the profile, and also residualize the profile on
    # the descriptors so the probe can only use what the descriptors do not
    # already explain.
    Xboth = np.hstack([Xdesc, Xatt])
    Xresid = _residualise(Xatt, Xdesc)

    out["classification"] = {
        "base_rate": float(y.mean()),
        "auroc_from_attention_profile": cv_auroc(Xatt, y),
        "auroc_from_descriptors": cv_auroc(Xdesc, y),
        "auroc_from_descriptors_plus_profile": cv_auroc(Xboth, y),
        "auroc_from_profile_residualised_on_descriptors": cv_auroc(Xresid, y),
        "auroc_attention_shuffled_null": cv_auroc(Xatt, lab_shuf),
        "auroc_model_score": float(roc_auc_score(y, idx["model_score"])),
        "auroc_ecfp_baseline": float(roc_auc_score(y, idx["ecfp_score"])),
        "descriptor_names": desc_names,
        "note": ("the profile and descriptor rows are supervised probes fitted "
                 "with cross-validation; the model-score and ECFP rows are "
                 "zero-shot rankers and are not directly comparable to them"),
    }

    # Per-ligand profile statistics, released so the figure can be rebuilt.
    stats = pd.DataFrame({
        "ident": idx["ident"],
        "role": idx["role"],
        "label": y,
        "model_pred": affinity,
        "corr_to_mean_profile": corr_to_mean,
        "profile_max": A.max(axis=1),
        "profile_entropy": -(np.where(A > 0, A * np.log(np.where(A > 0, A, 1)), 0.0)).sum(axis=1),
    })
    stats.to_csv(OUT_CSV, index=False)

    OUT_JSON.write_text(json.dumps(out, indent=2))

    vd = out["variance_decomposition"]
    print("--- variance decomposition of the ligand-by-residue attention matrix ---")
    print(f"  residue main effect (target-determined) : {vd['frac_residue_main_effect']:.4%}")
    print(f"  ligand main effect                      : {vd['frac_ligand_main_effect']:.4%}")
    print(f"  ligand x residue interaction            : {vd['frac_ligand_by_residue_interaction']:.4%}")
    print("\n--- profile similarity across ligands ---")
    print(f"  mean correlation to the panel mean profile : {sim['corr_to_mean_profile_mean']:.4f}")
    print(f"  5th percentile                             : {sim['corr_to_mean_profile_p05']:.4f}")
    print(f"  mean pairwise correlation                  : {sim['pairwise_corr_mean']:.4f}")
    ar = out["affinity_regression"]
    print("\n--- affinity as a linear function of the profile ---")
    print(f"  cross-validated R^2 from profile : {ar['r2_from_attention_profile']:.4f}")
    print(f"  shuffled-target null             : {ar['r2_shuffled_null']:.4f}")
    cl = out["classification"]
    print("\n--- binder vs non-binder from the profile ---")
    print("  supervised probes (5-fold cross-validated):")
    print(f"    attention profile                : {cl['auroc_from_attention_profile']:.4f}")
    print(f"    descriptors only                 : {cl['auroc_from_descriptors']:.4f}")
    print(f"    descriptors + profile            : {cl['auroc_from_descriptors_plus_profile']:.4f}")
    print(f"    profile residualized on desc.    : {cl['auroc_from_profile_residualised_on_descriptors']:.4f}")
    print(f"    shuffled-label null              : {cl['auroc_attention_shuffled_null']:.4f}")
    print("  zero-shot rankers (not comparable to the probes above):")
    print(f"    model affinity score             : {cl['auroc_model_score']:.4f}")
    print(f"    nearest-active ECFP              : {cl['auroc_ecfp_baseline']:.4f}")
    print(f"\nwrote {OUT_JSON.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
