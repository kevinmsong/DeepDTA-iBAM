"""
Score the assay-defined EGFR panel with DeepDTA-iBAM and with an ECFP baseline.

This is the in-domain, non-circular counterpart to the EGFR panel disqualified
in the main text, and the comparison the study otherwise lacks: the H1 panel is
admissible but out of domain for a kinase-trained model, so it cannot say how
the model compares against ligand-similarity search on its own target family.

Both rankers are scored on the identical candidate set, so their errors are
correlated and the head-to-head difference is tested with a paired bootstrap
rather than by comparing marginal intervals.

Reads results/dude_egfr_panel.csv (from build_dude_egfr_panel.py) and writes
  results/dude_egfr_panel_scored.csv
  table_kinase_panel.tex
  kinase_panel_results.json

Run from the submission directory:  python score_kinase_panel.py
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

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = _ROOT
sys.path.insert(0, str(ROOT))
SUB = Path(__file__).resolve().parent
os.chdir(ROOT)

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score, average_precision_score

RDLogger.DisableLog("rdApp.*")

from config_profiles import get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import make_unlabeled_prediction_loader, predict_unlabeled
from case_studies_results_generation import load_publication_ensemble

PANEL = ROOT / "results" / "dude_egfr_panel.csv"
OUT_CSV = ROOT / "results" / "dude_egfr_panel_scored.csv"
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
SEED = 1337
N_BOOT = 10000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def enrichment(y, s, frac):
    n = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-s)[:n]
    hits = float(y[idx].sum())
    return (hits / n) / (y.mean() + 1e-12)


def recovery(y, s, frac):
    n = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-s)[:n]
    return float(y[idx].sum()) / max(1.0, float(y.sum()))


def main():
    rows = list(csv.DictReader(open(PANEL)))
    y = np.array([1 if r["role"] == "active" else 0 for r in rows])
    smiles = [r["smiles"] for r in rows]
    log(f"panel: {int(y.sum())} actives, {int((1-y).sum())} decoys")

    # ---------------- ECFP nearest-active baseline (leave-one-out) ---------
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [MFPGEN.GetFingerprint(m) for m in mols]
    afps = [f for f, lab in zip(fps, y) if lab == 1]
    ecfp = []
    ai = 0
    for f, lab in zip(fps, y):
        if lab == 1:
            others = afps[:ai] + afps[ai + 1:]
            ai += 1
        else:
            others = afps
        ecfp.append(max(DataStructs.BulkTanimotoSimilarity(f, others)))
    ecfp = np.array(ecfp)
    log(f"ECFP baseline AUROC {roc_auc_score(y, ecfp):.4f}")

    # ---------------- model ------------------------------------------------
    seq = json.load(open("results/cache/egfr_interpolation/"
                         "egfr_interpolation_proteins_manifest.json"))
    seq = next(iter(seq["items"].values()))["sequence"]

    cfg = get_config_profile("max_rmse_cluster_diffusion")
    cfg.build_caches_on_start = False
    cfg.ensemble_size = 1

    log("building ligand caches")
    iso, gc_, pc_ = build_isolated_caches(
        cfg, smiles, [seq], "results/cache/dude_egfr_panel",
        force_rebuild=False, cache_prefix="dude_egfr_panel")
    iso.num_workers = 0          # Windows spawn guard; see probe_score.py
    models, norm = load_publication_ensemble(iso, checkpoint_dir=cfg.checkpoint_dir)

    df = pd.DataFrame({"compound_iso_smiles": smiles,
                       "target_sequence": [seq] * len(smiles)})
    loader = make_unlabeled_prediction_loader(df, gc_, pc_, iso)
    t = time.time()
    payload = predict_unlabeled(models, loader, iso, norm, collect_attention=False)
    model = np.asarray(payload["predictions"], dtype=float)
    log(f"scored {len(model)} molecules in {time.time()-t:.0f}s")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["role", "ident", "smiles", "label",
                    "model_score", "ecfp_score"])
        for r, lab, ms, es in zip(rows, y, model, ecfp):
            w.writerow([r["role"], r["ident"], r["smiles"], int(lab),
                        f"{ms:.6f}", f"{es:.6f}"])

    # ---------------- metrics and paired bootstrap -------------------------
    res = {}
    for name, s in (("DeepDTA-iBAM", model), ("Nearest-active ECFP", ecfp)):
        res[name] = dict(
            auroc=float(roc_auc_score(y, s)),
            auprc=float(average_precision_score(y, s)),
            ef1=enrichment(y, s, 0.01), ef5=enrichment(y, s, 0.05),
            recovery10=recovery(y, s, 0.10))

    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            diffs[b] = np.nan
            continue
        diffs[b] = roc_auc_score(yy, model[idx]) - roc_auc_score(yy, ecfp[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs = res["DeepDTA-iBAM"]["auroc"] - res["Nearest-active ECFP"]["auroc"]
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    paired = dict(observed=float(obs), lo=float(lo), hi=float(hi),
                  p=float(max(p, 1.0 / len(diffs))), n_boot=int(len(diffs)))

    out = dict(n_actives=int(y.sum()), n_decoys=int((1 - y).sum()),
               metrics=res, paired_bootstrap=paired)
    json.dump(out, open(_out("kinase_panel_results.json"), "w"), indent=2)

    with open(_out("table_kinase_panel.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        fh.write("Method & AUROC & AUPRC & EF1\\% & EF5\\% & Rec.@10\\% \\\\\n")
        fh.write("\\midrule\n")
        for name in ("DeepDTA-iBAM", "Nearest-active ECFP"):
            m = res[name]
            fh.write(f"{name} & {m['auroc']:.3f} & {m['auprc']:.3f} & "
                     f"{m['ef1']:.2f} & {m['ef5']:.2f} & "
                     f"{m['recovery10']:.3f} \\\\\n")
        def sgn(v):
            """Typeset signs as math so they render as minus, not hyphen."""
            return ("$-$" if v < 0 else "$+$") + f"{abs(v):.4f}"

        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{6}}{{l}}{{Paired bootstrap, model minus "
                 f"fingerprint: {sgn(paired['observed'])} "
                 f"(95\\% CI {sgn(paired['lo'])} to {sgn(paired['hi'])}, "
                 f"$p = {paired['p']:.4f}$)}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    log("RESULTS")
    for k, v in res.items():
        log(f"  {k:22s} AUROC {v['auroc']:.4f}  AUPRC {v['auprc']:.4f}  "
            f"EF1% {v['ef1']:.2f}  Rec@10% {v['recovery10']:.3f}")
    log(f"  paired difference {paired['observed']:+.4f} "
        f"[{paired['lo']:+.4f}, {paired['hi']:+.4f}] p={paired['p']:.4g}")


if __name__ == "__main__":
    main()
