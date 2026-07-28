"""
Build the in-domain, assay-defined EGFR retrieval panel from DUD-E.

Why not the local ZINC archive. The first attempt at this panel matched decoys
from the repository's own lead-like ZINC tranches. That archive spans roughly
207 to 320 Da at low lipophilicity, whereas the 6,060 ChEMBL EGFR actives
average 507 Da and cLogP 4.9, so only 1 active fell inside the pool's property
envelope. Matching regardless would have paired 507 Da actives with 253 Da
decoys and let molecular weight alone carry the ranking, which is the property
confound that DUD-E and LIT-PCBA exist to prevent. We therefore use DUD-E's
own EGFR target, whose decoys are property-matched to its actives by
construction.

What this panel does and does not fix. Its actives are defined by assay
outcome, not by similarity to a reference ligand, so it is free of the
selection-rule circularity diagnosed in the main text and a fingerprint
baseline is an admissible comparator on it. It is also in domain: EGFR is a
kinase, so unlike the H1 panel it tests the model on the target family it was
trained for. It does not, however, escape the separate and well documented
DUD-E decoy bias: DUD-E requires decoys to be topologically dissimilar to
actives, which advantages any fingerprint method. We therefore report the same
class-conditional similarity diagnostic used on the circular panel, so the
reader can see how much separation this construction carries.

Outputs
  results/dude_egfr_panel.csv
  results/dude_egfr_panel_diagnostic.json

Run from the submission directory:  python build_dude_egfr_panel.py
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
import random
import urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score, average_precision_score

RDLogger.DisableLog("rdApp.*")

ROOT = _ROOT
RES = ROOT / "results"
SEED = 1337
N_ACTIVES = 300
N_DECOYS = 3000

BASE = "https://dude.docking.org/targets/egfr/"
PANEL = RES / "dude_egfr_panel.csv"
DIAG = RES / "dude_egfr_panel_diagnostic.json"
RAW = RES / "downloads"

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fetch(name):
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"dude_egfr_{name}"
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(BASE + name,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = r.read().decode("utf-8", "replace")
    dest.write_text(text, encoding="utf-8")
    return text


def parse(text):
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        out.append((parts[0], parts[-1]))
    return out


def desc(m):
    return np.array([Descriptors.MolWt(m), Crippen.MolLogP(m),
                     rdMolDescriptors.CalcNumHBD(m),
                     rdMolDescriptors.CalcNumHBA(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m),
                     Chem.GetFormalCharge(m)], dtype=float)


def main():
    rng = random.Random(SEED)
    acts = parse(fetch("actives_final.ism"))
    decs = parse(fetch("decoys_final.ism"))
    print(f"DUD-E EGFR: {len(acts)} actives, {len(decs)} decoys")

    acts = [a for a in acts if Chem.MolFromSmiles(a[0])]
    decs = [d for d in decs if Chem.MolFromSmiles(d[0])]
    if len(acts) > N_ACTIVES:
        acts = rng.sample(acts, N_ACTIVES)
    if len(decs) > N_DECOYS:
        decs = rng.sample(decs, N_DECOYS)

    rows = [dict(role="active", ident=i, smiles=s) for s, i in acts] + \
           [dict(role="decoy", ident=i, smiles=s) for s, i in decs]
    with open(PANEL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["role", "ident", "smiles"])
        w.writeheader(); w.writerows(rows)
    print(f"panel: {len(acts)} actives, {len(decs)} decoys -> {PANEL.name}")

    mols = [Chem.MolFromSmiles(r["smiles"]) for r in rows]
    y = np.array([1 if r["role"] == "active" else 0 for r in rows])
    fps = [MFPGEN.GetFingerprint(m) for m in mols]

    # property balance
    d = np.array([desc(m) for m in mols])
    names = ["MW", "cLogP", "HBD", "HBA", "RotB", "charge"]
    balance = {}
    for k, n in enumerate(names):
        a, b = d[y == 1, k], d[y == 0, k]
        pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)) or 1e-9
        balance[n] = dict(active=round(float(a.mean()), 2),
                          decoy=round(float(b.mean()), 2),
                          std_diff=round(float((a.mean() - b.mean()) / pooled), 3))

    # similarity to the reference binders used by the circular panel
    refs = [MFPGEN.GetFingerprint(Chem.MolFromSmiles(s))
            for s in (GEFITINIB, ERLOTINIB)]
    simref = np.array([max(DataStructs.TanimotoSimilarity(f, q) for q in refs)
                       for f in fps])
    pos, neg = simref[y == 1], simref[y == 0]
    gap = float(pos.min() - neg.max())
    lo, hi = max(pos.min(), neg.min()), min(pos.max(), neg.max())

    # ECFP nearest-active baseline, leave-one-out on actives
    afps = [f for f, lab in zip(fps, y) if lab == 1]
    ecfp, ai = [], 0
    for f, lab in zip(fps, y):
        if lab == 1:
            others = afps[:ai] + afps[ai + 1:]; ai += 1
        else:
            others = afps
        ecfp.append(max(DataStructs.BulkTanimotoSimilarity(f, others)))
    ecfp = np.array(ecfp)

    out = dict(
        n_actives=int(y.sum()), n_decoys=int((1 - y).sum()),
        property_balance=balance,
        worst_abs_std_diff=round(max(abs(v["std_diff"]) for v in balance.values()), 3),
        similarity_to_reference=dict(
            positives=[round(float(pos.min()), 4), round(float(pos.max()), 4)],
            decoys=[round(float(neg.min()), 4), round(float(neg.max()), 4)],
            gap=round(gap, 4), supports_disjoint=bool(gap > 0),
            overlap_region=[round(float(lo), 4), round(float(hi), 4)],
            n_positives_in_overlap=int(((pos >= lo) & (pos <= hi)).sum()),
            n_decoys_in_overlap=int(((neg >= lo) & (neg <= hi)).sum())),
        ecfp_baseline=dict(auroc=round(float(roc_auc_score(y, ecfp)), 4),
                           auprc=round(float(average_precision_score(y, ecfp)), 4)),
    )
    json.dump(out, open(DIAG, "w"), indent=2)

    print("\nPROPERTY BALANCE (active vs decoy, standardised difference)")
    for n, v in balance.items():
        print(f"  {n:6s} {v['active']:8.2f}  {v['decoy']:8.2f}  {v['std_diff']:+.3f}")
    print(f"  worst |std diff| {out['worst_abs_std_diff']}")
    s = out["similarity_to_reference"]
    print("\nSIMILARITY TO REFERENCE BINDERS")
    print(f"  positives {s['positives']}, decoys {s['decoys']}")
    print(f"  supports disjoint: {s['supports_disjoint']} (gap {s['gap']})")
    print(f"  overlap {s['overlap_region']}: {s['n_positives_in_overlap']} positives,"
          f" {s['n_decoys_in_overlap']} decoys")
    print(f"\nECFP nearest-active baseline: AUROC {out['ecfp_baseline']['auroc']}, "
          f"AUPRC {out['ecfp_baseline']['auprc']}")


if __name__ == "__main__":
    main()
