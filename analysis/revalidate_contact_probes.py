"""Revalidate saved EGFR attention probes without retraining the base model.

Ridge penalties are selected by an inner five-fold CV inside each outer
training fold. Descriptor residualization is fitted on outer training rows
only and then applied to their held-out rows. Archived inputs are unchanged.
Run from anywhere with the project Python environment.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEED = 1337
FOLDS = 5
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
DESCRIPTORS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "Rings",
               "HeavyAtoms", "FracCSP3", "FormalCharge"]
INPUT_NAMES = ["panel_attention_residue.npy", "panel_attention_index.csv",
               "panel_attention_meta.json", "dude_egfr_panel_scored.csv",
               "contact_information_summary.json"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptors_and_atoms(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    rows, atoms = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}; no global imputation allowed")
        # The graph builder uses MolFromSmiles/GetNumAtoms, which preserves
        # certain explicit hydrogens. The archived index column is misnamed.
        atoms.append(mol.GetNumAtoms())
        rows.append([Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                     rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcNumHBD(mol),
                     rdMolDescriptors.CalcNumHBA(mol),
                     rdMolDescriptors.CalcNumRotatableBonds(mol),
                     rdMolDescriptors.CalcNumRings(mol), mol.GetNumHeavyAtoms(),
                     rdMolDescriptors.CalcFractionCSP3(mol), Chem.GetFormalCharge(mol)])
    result = np.asarray(rows, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite descriptors")
    return result, np.asarray(atoms)


def check_alignment(A: np.ndarray, idx: pd.DataFrame, panel: pd.DataFrame,
                    meta: dict, atoms: np.ndarray) -> dict:
    assert A.shape == (len(idx), meta["n_residue_positions"])
    assert len(idx) == len(panel) == meta["n_ligands"]
    assert idx["smiles"].is_unique
    same = {}
    for column in ["role", "ident", "smiles", "label", "model_score", "ecfp_score"]:
        same[column] = bool(idx[column].equals(panel[column]))
        assert same[column], f"Panel/index disagreement in {column}"
    assert np.array_equal(idx["label"].to_numpy(), (idx["role"] == "active").astype(int))
    atom_match = bool(np.array_equal(atoms, idx["n_heavy_atoms_padded"].to_numpy()))
    assert atom_match, "Graph atom counts disagree with indexed molecular structures"
    delta = idx["model_pred_rerun"].to_numpy() - panel["model_score"].to_numpy()
    assert np.max(np.abs(delta)) < 1e-4, "Export predictions do not match independent panel scores"
    assert np.isfinite(A).all() and np.min(A) >= 0
    sum_error = np.max(np.abs(A.sum(axis=1) - 1.0))
    assert sum_error < 1e-5, "Attention profiles are not normalized distributions"
    # Independently instantiate the actual export sampler. All candidates share
    # one target; stable sorting of equal lengths must preserve their row order.
    import sys
    sys.path.insert(0, str(ROOT))
    from data.datasets import TokenBudgetBatchSampler
    sampler = TokenBudgetBatchSampler([meta["sequence_length"]] * len(idx),
                                      max_pairs_per_batch=7,
                                      protein_token_budget=7 * meta["sequence_length"],
                                      shuffle=False, seed=SEED)
    order = [i for batch in sampler for i in batch]
    assert order == list(range(len(idx))), "Export sampler changes row order"
    return {"panel_index_columns_exact_match": same,
            "unique_smiles": True,
            "duplicate_identifier_rows_beyond_first": int(idx["ident"].duplicated().sum()),
            "duplicate_identifier_records": idx.loc[idx["ident"].duplicated(False),
                                                    ["role", "ident", "smiles", "label"]].to_dict("records"),
            "all_graph_atom_counts_match_structures": atom_match,
            "atom_column_note": "Archived n_heavy_atoms_padded is graph atom count (RDKit GetNumAtoms), including retained explicit hydrogens in six molecules.",
            "max_absolute_rerun_vs_saved_affinity_difference": float(np.max(np.abs(delta))),
            "rerun_vs_saved_affinity_correlation": float(np.corrcoef(
                idx["model_pred_rerun"], panel["model_score"])[0, 1]),
            "max_profile_sum_error": float(sum_error),
            "export_sampler_preserves_constant_target_row_order": True,
            "limitation": "Saved profile rows have no embedded ligand IDs; alignment is supported by export code, row identities, atom counts, and independent prediction agreement."}


def logistic_oof(X: np.ndarray, y: np.ndarray, splits: list,
                 controls: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    predictions = np.full(len(y), np.nan)
    fold_info = []
    for fold, (train, test) in enumerate(splits):
        assert not np.intersect1d(train, test).size
        train_X, test_X = X[train], X[test]
        if controls is not None:
            control_scaler = StandardScaler().fit(controls[train])
            train_C = np.column_stack([control_scaler.transform(controls[train]),
                                       np.ones(len(train))])
            test_C = np.column_stack([control_scaler.transform(controls[test]),
                                      np.ones(len(test))])
            beta, *_ = np.linalg.lstsq(train_C, train_X, rcond=None)
            train_X = train_X - train_C @ beta
            test_X = test_X - test_C @ beta
        pipe = make_pipeline(StandardScaler(), LogisticRegression(
            penalty="l2", C=0.1, max_iter=5000, class_weight="balanced", random_state=SEED))
        pipe.fit(train_X, y[train])
        predictions[test] = pipe.predict_proba(test_X)[:, 1]
        fold_info.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                          "auroc": float(roc_auc_score(y[test], predictions[test])),
                          "solver_iterations": int(pipe[-1].n_iter_[0])})
    assert np.isfinite(predictions).all()
    return predictions, fold_info


def ridge_oof(X: np.ndarray, y: np.ndarray, splits: list) -> tuple[np.ndarray, list, list]:
    predictions = np.full(len(y), np.nan)
    fold_info, inner_assignments = [], []
    for fold, (train, test) in enumerate(splits):
        assert not np.intersect1d(train, test).size
        inner = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(train))
        for inner_fold, (_, validation) in enumerate(inner):
            inner_assignments.extend({"outer_fold": fold, "row_index": int(row),
                                      "inner_validation_fold": inner_fold}
                                     for row in train[validation])
        estimator = make_pipeline(StandardScaler(), Ridge(random_state=SEED))
        search = GridSearchCV(estimator, {"ridge__alpha": ALPHAS}, scoring="r2",
                              cv=inner, n_jobs=1, refit=True, error_score="raise")
        search.fit(X[train], y[train])
        predictions[test] = search.predict(X[test])
        fold_info.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                          "selected_alpha": float(search.best_params_["ridge__alpha"]),
                          "inner_mean_validation_r2": search.cv_results_["mean_test_score"].tolist(),
                          "outer_r2": float(r2_score(y[test], predictions[test]))})
        log(f"Ridge outer fold {fold + 1}: alpha {fold_info[-1]['selected_alpha']}, R2 {fold_info[-1]['outer_r2']:.6f}")
    assert np.isfinite(predictions).all()
    return predictions, fold_info, inner_assignments


def assign_folds(n: int, splits: list) -> np.ndarray:
    assigned = np.full(n, -1, dtype=int)
    for fold, (_, test) in enumerate(splits):
        assert (assigned[test] == -1).all()
        assigned[test] = fold
    assert (assigned >= 0).all()
    return assigned


def main() -> None:
    started = time.time()
    input_hashes = {name: digest(RES / name) for name in INPUT_NAMES}
    A = np.load(RES / INPUT_NAMES[0]).astype(np.float64)
    idx = pd.read_csv(RES / INPUT_NAMES[1])
    meta = json.loads((RES / INPUT_NAMES[2]).read_text())
    panel = pd.read_csv(RES / INPUT_NAMES[3])
    archive = json.loads((RES / INPUT_NAMES[4]).read_text())
    C, atoms = descriptors_and_atoms(idx["smiles"].tolist())
    alignment = check_alignment(A, idx, panel, meta, atoms)
    log(f"Alignment checks passed for {A.shape[0]} ligands by {A.shape[1]} residues")
    labels = idx["label"].to_numpy(int)
    affinity = idx["model_pred_rerun"].to_numpy(float)
    # Preserve the archived shuffled controls exactly: the old script first
    # consumed a 400-row sample for the pairwise correlation diagnostic.
    rng = np.random.default_rng(SEED)
    rng.choice(len(idx), size=min(400, len(idx)), replace=False)
    shuffled_affinity = rng.permutation(affinity)
    shuffled_labels = rng.permutation(labels)
    classification_splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(A, labels))
    null_splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(A, shuffled_labels))
    regression_splits = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(A))
    oof = idx.copy()
    oof.insert(0, "row_index", np.arange(len(idx)))
    oof["classification_fold"] = assign_folds(len(idx), classification_splits)
    oof["classification_shuffled_fold"] = assign_folds(len(idx), null_splits)
    oof["regression_fold"] = assign_folds(len(idx), regression_splits)
    oof["shuffled_label"] = shuffled_labels
    oof["shuffled_predicted_affinity"] = shuffled_affinity
    classification, fold_details = {}, {}
    jobs = [("attention_profile", A, labels, classification_splits, None),
            ("descriptors", C, labels, classification_splits, None),
            ("descriptors_plus_profile", np.column_stack([C, A]), labels, classification_splits, None),
            ("profile_residualised_on_descriptors", A, labels, classification_splits, C),
            ("attention_shuffled_null", A, shuffled_labels, null_splits, None)]
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*penalty.*")
        for name, X, y, splits, controls in jobs:
            pred, detail = logistic_oof(X, y, splits, controls)
            key = "auroc_" + (name if name == "attention_shuffled_null" else "from_" + name)
            classification[key] = float(roc_auc_score(y, pred))
            fold_details[name] = detail
            oof["logistic_" + name] = pred
            log(f"{name}: OOF AUROC {classification[key]:.9f}")
        ridge_pred, ridge_detail, inner = ridge_oof(A, affinity, regression_splits)
        shuffled_pred, shuffled_detail, shuffled_inner = ridge_oof(A, shuffled_affinity, regression_splits)
        assert inner == shuffled_inner
    oof["ridge_predicted_affinity"] = ridge_pred
    oof["ridge_shuffled_affinity"] = shuffled_pred
    regression = {"target": "model predicted affinity, not experimental affinity",
                  "r2_from_attention_profile": float(r2_score(affinity, ridge_pred)),
                  "r2_shuffled_null": float(r2_score(shuffled_affinity, shuffled_pred))}
    classification["base_rate"] = float(labels.mean())
    classification["auroc_model_score"] = float(roc_auc_score(labels, idx["model_score"]))
    classification["auroc_ecfp_baseline"] = float(roc_auc_score(labels, idx["ecfp_score"]))
    comparison = {"classification": {}, "affinity_regression": {}}
    for group, values in [("classification", classification), ("affinity_regression", regression)]:
        for key, value in values.items():
            old = archive[group].get(key)
            if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                comparison[group][key] = {"archived": old, "revalidated": value, "change": value - old}
    # Independently call the archived implementation in the current numerical
    # environment. Saved metrics need not agree bit-for-bit across dependency
    # versions and threaded linear algebra; unchanged methods must agree here.
    from analysis_contact_information import cv_auroc, _residualise
    reference = {}
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*penalty.*")
        for name, features, target in [
            ("auroc_from_attention_profile", A, labels),
            ("auroc_from_descriptors", C, labels),
            ("auroc_from_descriptors_plus_profile", np.column_stack([C, A]), labels),
            ("auroc_attention_shuffled_null", A, shuffled_labels)]:
            reference[name] = cv_auroc(features, target)
            assert abs(classification[name] - reference[name]) < 1e-12, name
        reference["auroc_from_profile_residualised_on_descriptors"] = cv_auroc(_residualise(A, C), labels)
    assert input_hashes == {name: digest(RES / name) for name in INPUT_NAMES}
    output = {
        "seed": SEED, "outer_folds": FOLDS, "inner_folds": FOLDS,
        "n_ligands": len(idx), "n_actives": int(labels.sum()), "n_residue_features": A.shape[1],
        "thread_limit": 2, "input_sha256": input_hashes, "script_sha256": digest(Path(__file__)),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "rdkit": rdkit.__version__},
        "alignment_checks": alignment,
        "methods": {"logistic": "Shuffled stratified 5-fold CV, seed 1337; fixed L2 C=0.1, balanced class weights; StandardScaler fit only on outer training rows.",
                    "residualization": "Fit descriptor scaling and ordinary least-squares profile-on-descriptor regression with intercept on each outer training fold; transform held-out rows using those fitted coefficients; fit residual scaling and logistic probe on training residuals only.",
                    "ridge": "Shuffled outer 5-fold KFold, seed 1337, matching archived folds; within each training fold use shuffled 5-fold KFold seed 1337 to select alpha by mean validation R2 from [1,10,100,1000]; scaling refit within every inner training fold; refit selected pipeline on all outer training rows.",
                    "metrics": "AUROC and R2 calculated once over all concatenated out-of-fold predictions; per-fold metrics are also retained.",
                    "nulls": "One fixed permutation each of labels and predicted affinity, preserving the archived script's RNG state consumption; nested penalty selection repeated for the regression null."},
        "descriptor_names": DESCRIPTORS,
        "affinity_regression": regression, "classification": classification,
        "classification_fold_details": fold_details,
        "ridge_fold_details": ridge_detail, "ridge_shuffled_fold_details": shuffled_detail,
        "comparison_with_archive": comparison,
        "archived_method_current_environment_classification": reference,
        "limitations": ["The base checkpoint and attention features remain fixed; no new model training or inference was performed.",
                        "Regression targets are model predictions, not measured affinities.",
                        "Random candidate folds do not establish generalization to new chemical series or targets.",
                        "DUD-E active/decoy labels retain decoy-selection and property biases; decoys are not experimentally established inactive compounds.",
                        "One set of folds and one shuffled null per task do not characterize uncertainty across repetitions.",
                        "The supervised probes use panel labels for training, whereas the saved model and fingerprint rankings use different information and are not directly comparable."],
        "seconds": round(time.time() - started, 3)}
    oof.to_csv(RES / "contact_information_revalidated_oof.csv", index=False)
    pd.DataFrame(inner).to_csv(RES / "contact_information_revalidated_inner_folds.csv", index=False)
    (RES / "contact_information_revalidated.json").write_text(json.dumps(output, indent=2) + "\n")
    log(f"Completed: nested ridge R2 {regression['r2_from_attention_profile']:.9f}; shuffled {regression['r2_shuffled_null']:.9f}")


if __name__ == "__main__":
    main()
