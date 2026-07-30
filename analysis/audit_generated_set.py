"""
Audit of the EGFR-conditioned diffusion analog set.

Three problems were identified in the released set of "100 unique valid
diffusion analogs" and in the Table S9 gallery built from it:

  1. The dasatinib seed itself is present as a generated row (Tanimoto 1.000),
     so the set contains 99 analogs, not 100.
  2. Uniqueness was assessed by canonical SMILES, which counts single-atom
     relabellings of one scaffold as distinct molecules and therefore
     overstates chemical diversity.
  3. A subset of the decoded structures carry motifs that RDKit sanitizes
     without complaint but that no medicinal chemist would advance: O--O
     peroxide linkages closed inside rings, low-valent phosphorus
     (the C=[PH] phosphaalkene), and aliphatic N--N / N--O ring bonds
     (cyclic hydrazines and hydroxylamines, several fused to the peroxides).

This script re-audits the set from the released CSV, applies an explicit
structural-alert filter, and reports scaffold-level diversity by the
Bemis--Murcko framework rather than by canonical-SMILES uniqueness.

Outputs
-------
  generated_egfr_analogs_audited.csv    surviving analogs, ranked
  table_generated_audit.tex             audit summary (Table S8)
  table_generated_top10.tex             top 10 survivors (Table S9)
  audit_summary.json                    machine-readable counts

Run from the submission directory:  python audit_generated_set.py
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

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

RES = str(_ROOT / "results") + "/"
SRC = RES + "generated_egfr_analogs_100.csv"

# Dasatinib, the fixed starting structure for the local-design benchmark.
DASATINIB = "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"

# ---------------------------------------------------------------------------
# Structural alerts.  Each is a motif that survives RDKit sanitization but is
# not a credible synthetic target in a kinase-inhibitor analog series.
# ---------------------------------------------------------------------------
ALERTS = [
    ("Peroxide O--O",        "[OX2]-[OX2]"),
    ("Low-valent P",         "[#15;!$([PX4])]"),
    ("Ring hydrazine N--N",  "[NX3;R]-[NX3;R]"),
    ("Ring hydroxylamine N--O", "[NX3;R]-[OX2;R]"),
]
ALERT_PATTERNS = [(name, Chem.MolFromSmarts(sma)) for name, sma in ALERTS]


def alerts_for(mol):
    return [name for name, patt in ALERT_PATTERNS if mol.HasSubstructMatch(patt)]


def scaffold(mol):
    """Bemis--Murcko framework as canonical SMILES ('' for acyclic molecules)."""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core.GetNumAtoms() else ""


def generic_scaffold(mol):
    """Bemis--Murcko graph framework: ring/linker topology, atom types erased."""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if not core.GetNumAtoms():
        return ""
    return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))


def main():
    rows = list(csv.DictReader(open(SRC)))
    seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(DASATINIB))

    records = []
    invalid = 0
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            invalid += 1
            continue
        canon = Chem.MolToSmiles(mol)
        records.append(dict(
            row=r, mol=mol, canon=canon,
            is_seed=(canon == seed_canon),
            alerts=alerts_for(mol),
            scaffold=scaffold(mol),
            generic=generic_scaffold(mol),
        ))

    n_raw = len(rows)
    n_valid = len(records)
    n_unique_canon = len({x["canon"] for x in records})
    n_seed = sum(x["is_seed"] for x in records)

    # --- stage 1: remove the seed -----------------------------------------
    stage1 = [x for x in records if not x["is_seed"]]

    # --- stage 2: remove structural alerts --------------------------------
    # Cross-tabulated against the released PAINS/BRENK "AlertFree" flag, to
    # record which of these motifs the published filter already caught.  BRENK
    # catches the peroxides and the phosphine; the cyclic hydrazines and
    # hydroxylamines pass it, so the released alert-free rate does not bound
    # the problem.
    alert_counts = {name: 0 for name, _ in ALERTS}
    alert_passed_brenk = {name: 0 for name, _ in ALERTS}
    for x in stage1:
        for a in x["alerts"]:
            alert_counts[a] += 1
            if float(x["row"]["AlertFree"]):
                alert_passed_brenk[a] += 1
    n_any_alert = sum(1 for x in stage1 if x["alerts"])
    n_alert_passed_brenk = sum(1 for x in stage1
                               if x["alerts"] and float(x["row"]["AlertFree"]))
    stage2 = [x for x in stage1 if not x["alerts"]]

    # --- diversity, at each stage ------------------------------------------
    def diversity(rs):
        return dict(n=len(rs),
                    canonical=len({x["canon"] for x in rs}),
                    scaffolds=len({x["scaffold"] for x in rs}),
                    generic=len({x["generic"] for x in rs}))

    div_raw = diversity(records)
    div_1 = diversity(stage1)
    div_2 = diversity(stage2)

    # --- properties of the surviving set -----------------------------------
    def stat(rs, key, cast=float):
        vals = [cast(x["row"][key]) for x in rs]
        return sum(vals) / len(vals) if vals else float("nan")

    surviving = dict(
        n=len(stage2),
        qed=stat(stage2, "QED"),
        sa=stat(stage2, "SA"),
        tanimoto=stat(stage2, "tanimoto"),
        affinity=stat(stage2, "PredAffinity"),
        lipinski=stat(stage2, "LipinskiPass", lambda v: float(v)),
        alertfree=stat(stage2, "AlertFree", lambda v: float(v)),
    )

    # most populated scaffolds among survivors
    scaf_pop = {}
    for x in stage2:
        scaf_pop.setdefault(x["scaffold"], []).append(x)
    top_scaffolds = sorted(scaf_pop.items(), key=lambda kv: -len(kv[1]))[:5]

    # ---------------------------------------------------------------------
    # Write the audited CSV
    # ---------------------------------------------------------------------
    fields = list(rows[0].keys()) + ["canonical_smiles", "murcko_scaffold",
                                     "murcko_generic_scaffold"]
    with open(RES + "generated_egfr_analogs_audited.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for x in sorted(stage2, key=lambda x: -float(x["row"]["PredAffinity"])):
            out = dict(x["row"])
            out["canonical_smiles"] = x["canon"]
            out["murcko_scaffold"] = x["scaffold"]
            out["murcko_generic_scaffold"] = x["generic"]
            w.writerow(out)

    # ---------------------------------------------------------------------
    # Table: audit summary
    # ---------------------------------------------------------------------
    with open(_out("table_generated_audit.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        fh.write("Stage & Molecules & Unique canonical & Murcko scaffolds "
                 "& Generic frameworks \\\\\n\\midrule\n")
        fh.write(f"As released & {div_raw['n']} & {div_raw['canonical']} & "
                 f"{div_raw['scaffolds']} & {div_raw['generic']} \\\\\n")
        fh.write(f"Seed removed & {div_1['n']} & {div_1['canonical']} & "
                 f"{div_1['scaffolds']} & {div_1['generic']} \\\\\n")
        fh.write(f"Alert-filtered & {div_2['n']} & {div_2['canonical']} & "
                 f"{div_2['scaffolds']} & {div_2['generic']} \\\\\n")
        fh.write("\\midrule\n\\multicolumn{3}{l}{\\textit{Structural alerts "
                 "removed (molecules may carry more than one)}} & "
                 "\\multicolumn{2}{r}{\\textit{Passed PAINS/BRENK}} \\\\\n")
        for name, _ in ALERTS:
            fh.write(f"\\quad {name} & {alert_counts[name]} & & "
                     f"\\multicolumn{{2}}{{r}}{{{alert_passed_brenk[name]}}} \\\\\n")
        fh.write(f"\\quad Any alert & {n_any_alert} & & "
                 f"\\multicolumn{{2}}{{r}}{{{n_alert_passed_brenk}}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------------
    # Table: top 10 survivors, no SMILES column
    # ---------------------------------------------------------------------
    ranked = sorted(stage2, key=lambda x: (
        -float(x["row"]["LipinskiPass"]), -float(x["row"]["AlertFree"]),
        -float(x["row"]["QED"]), float(x["row"]["SA"]),
        -float(x["row"]["PredAffinity"])))[:10]
    with open(_out("table_generated_top10.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rlrrrrrcc}\n\\toprule\n")
        fh.write("\\# & Murcko framework (rings) & MW & cLogP & QED & SA & "
                 "Tanimoto & Lip. & Alert-free \\\\\n\\midrule\n")
        for i, x in enumerate(ranked, 1):
            m = x["mol"]
            nring = Chem.rdMolDescriptors.CalcNumRings(m)
            narom = Chem.rdMolDescriptors.CalcNumAromaticRings(m)
            frame = f"{nring} rings ({narom} aromatic)"
            r = x["row"]
            fh.write(f"{i} & {frame} & {float(r['MW']):.1f} & "
                     f"{float(r['cLogP']):.2f} & {float(r['QED']):.3f} & "
                     f"{float(r['SA']):.2f} & {float(r['tanimoto']):.3f} & "
                     f"{'yes' if float(r['LipinskiPass']) else 'no'} & "
                     f"{'yes' if float(r['AlertFree']) else 'no'} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------------
    # Structure gallery for the ten tabulated survivors, so the supplementary
    # table can carry properties and the figure can carry chemistry, with the
    # SMILES living only in the CSV artifact.
    # ---------------------------------------------------------------------
    # Emitted as vector PDF (preferred by the manuscript) plus a 600 dpi PNG
    # fallback sized for the full text width, so neither is resolution limited.
    try:
        from rdkit.Chem import Draw

        mols = [x["mol"] for x in ranked]
        legends = [f"{i}. QED {float(x['row']['QED']):.2f}, "
                   f"SA {float(x['row']['SA']):.1f}, "
                   f"Tanimoto {float(x['row']['tanimoto']):.2f}"
                   for i, x in enumerate(ranked, 1)]

        svg = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(320, 330),
                                   legends=legends, useSVG=True)
        svg = svg.data if hasattr(svg, "data") else str(svg)
        import fitz
        doc = fitz.open("svg", svg.encode("utf-8"))
        with open(RES + "fig_generated_gallery_audited.pdf", "wb") as fh:
            fh.write(doc.convert_to_pdf())
        print("\n[fig] results/fig_generated_gallery_audited.pdf (vector)")

        # 600 dpi raster at 6.5 in wide => 3900 px across 5 columns.
        png = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(780, 800),
                                   legends=legends, useSVG=False,
                                   returnPNG=False)
        png.save(RES + "fig_generated_gallery_audited.png", dpi=(600, 600))
        print(f"[fig] results/fig_generated_gallery_audited.png "
              f"({png.width}x{png.height} px, 600 dpi at 6.5 in)")
    except Exception as exc:  # pragma: no cover - drawing backend optional
        print(f"\n[warn] gallery rendering skipped: {exc}")

    summary = dict(
        n_raw=n_raw, n_invalid=invalid, n_valid=n_valid,
        n_unique_canonical=n_unique_canon, n_seed_rows=n_seed,
        alert_counts=alert_counts, n_any_alert=n_any_alert,
        alert_passed_brenk=alert_passed_brenk,
        n_alert_passed_brenk=n_alert_passed_brenk,
        diversity=dict(released=div_raw, seed_removed=div_1, filtered=div_2),
        surviving=surviving,
        top_scaffold_sizes=[len(v) for _, v in top_scaffolds],
    )
    json.dump(summary, open(_out("audit_summary.json"), "w"), indent=2)

    # ---------------------------------------------------------------------
    print("GENERATED-SET AUDIT")
    print(f"  released rows              {n_raw}")
    print(f"  RDKit-invalid              {invalid}")
    print(f"  unique canonical SMILES    {n_unique_canon}")
    print(f"  dasatinib seed rows        {n_seed}")
    print(f"  carrying >=1 alert         {n_any_alert} "
          f"({n_alert_passed_brenk} of them passed PAINS/BRENK)")
    for name, _ in ALERTS:
        print(f"    {name:26s} {alert_counts[name]:3d} "
              f"(passed BRENK: {alert_passed_brenk[name]})")
    print(f"  surviving analogs          {len(stage2)}")
    print("\nDIVERSITY")
    for label, d in [("as released", div_raw), ("seed removed", div_1),
                     ("alert-filtered", div_2)]:
        print(f"  {label:15s} n={d['n']:3d}  canonical={d['canonical']:3d}  "
              f"Murcko={d['scaffolds']:3d}  generic={d['generic']:3d}")
    print("\nSURVIVING-SET MEANS")
    print(f"  QED {surviving['qed']:.3f}  SA {surviving['sa']:.3f}  "
          f"Tanimoto {surviving['tanimoto']:.3f}  affinity {surviving['affinity']:.4f}")
    print(f"  Lipinski {surviving['lipinski']:.2f}  alert-free {surviving['alertfree']:.2f}")
    print("\nLARGEST MURCKO SCAFFOLD CLASSES (survivors)")
    for s, v in top_scaffolds:
        print(f"  {len(v):3d}  {s[:70]}")


if __name__ == "__main__":
    main()
