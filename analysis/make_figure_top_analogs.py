"""Draw dasatinib and the five highest-scoring audited diffusion analogs.

All ranking and affinity values come from the saved generation comparison,
including its returned seed, so displayed differences do not mix scoring
passes. No predictions or docking calculations are rerun. Exact saved SMILES
and docking identifiers are retained in the source CSV and provenance JSON.

Run: python analysis/make_figure_top_analogs.py [--width-mm 177.53]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import fitz
from matplotlib import font_manager
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdDepictor, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Contrib.SA_Score import sascorer

from audit_generated_set import ALERT_PATTERNS, DASATINIB
from figure_style import DPI

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STEM = "fig_top_analogs"
SOURCES = {
    "audited": RESULTS / "generated_egfr_analogs_audited.csv",
    "comparison": RESULTS / "generation_comparison.csv",
    "released": RESULTS / "generated_egfr_analogs_100.csv",
    "docking": RESULTS / "docking" / "vina_scores.csv",
    "summary": RESULTS / "table_generation_comparison_summary.csv",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable saved SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def unique_match(rows, key, description):
    matches = [(i + 1, row) for i, row in enumerate(rows)
               if canonical(row["smiles"]) == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one {description} match, found {len(matches)}")
    return matches[0]


def topology(mol):
    """All-carbon, single-bond connectivity for consistent depiction only."""
    simple = Chem.Mol(mol)
    Chem.RemoveStereochemistry(simple)
    for atom in simple.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    for bond in simple.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    simple.UpdatePropertyCache(strict=False)
    return simple


def select_records():
    sources = {name: read_csv(path) for name, path in SOURCES.items()}
    comparison = [r for r in sources["comparison"] if r["generator"] == "diffusion"]
    seed_key = canonical(DASATINIB)
    _, seed = unique_match(comparison, seed_key, "same-pass seed")
    candidates = sources["audited"]
    if len(candidates) != 60 or len({canonical(r["smiles"]) for r in candidates}) != 60:
        raise ValueError("The audited source must contain 60 unique molecules")
    # Verify the released motif audit, including seed exclusion, for all 60.
    for row in candidates:
        mol = Chem.MolFromSmiles(row["smiles"])
        if canonical(row["smiles"]) == seed_key or any(
                mol.HasSubstructMatch(pattern) for _, pattern in ALERT_PATTERNS):
            raise ValueError("An audited candidate is the seed or retains an excluded motif")
    ranked = sorted(candidates, key=lambda r: (-float(r["PredAffinity"]),
                                              canonical(r["smiles"])))[:5]
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = fingerprint.GetFingerprint(Chem.MolFromSmiles(seed["smiles"]))
    docking = [r for r in sources["docking"] if r["set"] == "generated_analog"]
    rows, checks = [], []
    for rank, saved in enumerate([seed] + ranked):
        key = canonical(saved["smiles"])
        comparison_record, same_pass = unique_match(comparison, key, "diffusion comparison")
        released_record, released = unique_match(sources["released"], key, "released analog")
        docking_record, dock = unique_match(docking, key, "docking")
        audited_record = (unique_match(candidates, key, "audited analog")[0] if rank else None)
        # The canonical join must not silently substitute a tautomer, isomer,
        # alternative raw SMILES, different prediction, or failed docking job.
        if not (saved["smiles"] == same_pass["smiles"] == released["smiles"] == dock["smiles"]):
            raise ValueError("Canonical match uses a different saved SMILES")
        if dock["status"] != "ok":
            raise ValueError("Selected docking result is not successful")
        for name in ("PredAffinity", "tanimoto", "QED", "SA", "AlertFree", "LipinskiPass"):
            if not (float(saved[name]) == float(same_pass[name]) == float(released[name])):
                raise ValueError(f"Saved sources disagree for {dock['ident']}: {name}")
        mol = Chem.MolFromSmiles(saved["smiles"])
        calculated = {
            "QED": QED.qed(mol), "SA": sascorer.calculateScore(mol),
            "tanimoto": DataStructs.TanimotoSimilarity(seed_fp, fingerprint.GetFingerprint(mol)),
        }
        errors = {name: float(value - float(saved[name])) for name, value in calculated.items()}
        if max(abs(value) for value in errors.values()) > 1e-10:
            raise ValueError("Saved descriptors failed independent recomputation")
        stereo = [{"type": str(info.type), "specified": str(info.specified),
                   "atom_or_bond_index": int(info.centeredOn),
                   "controlling_atoms": [int(i) for i in info.controllingAtoms]}
                  for info in Chem.FindPotentialStereo(mol)]
        row = {
            "panel": chr(97 + rank), "affinity_rank": rank or "seed",
            "saved_id": dock["ident"], "smiles": saved["smiles"],
            "canonical_isomeric_smiles": key,
            "PredAffinity": same_pass["PredAffinity"],
            "delta_KIBA_vs_same_pass_seed": float(same_pass["PredAffinity"]) - float(seed["PredAffinity"]),
            "tanimoto": same_pass["tanimoto"], "QED": same_pass["QED"], "SA": same_pass["SA"],
            "vina_kcal_mol": dock["vina_kcal"], "docking_status": dock["status"],
            "LipinskiPass": same_pass["LipinskiPass"], "AlertFree": same_pass["AlertFree"],
            "comparison_diffusion_record_1based": comparison_record,
            "released_record_1based": released_record,
            "audited_record_1based": audited_record,
            "docking_generated_analog_record_1based": docking_record,
            "potential_stereo": json.dumps(stereo, separators=(",", ":")),
        }
        rows.append(row)
        checks.append({"saved_id": dock["ident"], "descriptor_recomputation_errors": errors,
                       "potential_stereo": stereo, "raw_smiles_match_all_sources": True})
    reference = next(r for r in sources["summary"] if r["Generator"] == "seed_reference")
    reconciliation = {
        "displayed_same_pass_seed": float(seed["PredAffinity"]),
        "unused_separate_seed_reference": float(reference["PredAffinity mean"]),
        "explanation": "generation_comparison.csv scores all candidates, including the returned seed, in one pass. case_studies_results_generation.py then scores the seed separately for the summary. The figure uses only the candidate-pass scores for affinities and differences.",
        "source_code": "case_studies_results_generation.py:2942-2964",
    }
    return rows, checks, reconciliation


def molecule_svg(mol, seed, width, height):
    # All six saved structures have the same atom connectivity. Align them to
    # the seed for visual comparison, without changing elements, bonds, charge,
    # hydrogen counts, or source stereochemistry.
    drawn = Chem.Mol(mol)
    matches = topology(seed).GetSubstructMatches(topology(drawn), uniquify=False,
                                                useChirality=False, maxMatches=10000)
    if not matches or len(matches) >= 10000 or drawn.GetNumAtoms() != seed.GetNumAtoms():
        raise ValueError("A complete shared-topology depiction mapping was not found")
    mapping = min(matches)  # deterministic geometry choice; no metric is used
    conformer = Chem.Conformer(drawn.GetNumAtoms())
    conformer.Set3D(False)
    for atom, reference in enumerate(mapping):
        conformer.SetAtomPosition(atom, seed.GetConformer().GetAtomPosition(reference))
    drawn.RemoveAllConformers()
    drawn.AddConformer(conformer)
    scale = 3
    drawer = rdMolDraw2D.MolDraw2DSVG(round(width * scale), round(height * scale))
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = 0.7 * scale
    opts.fixedFontSize = 8 * scale
    opts.minFontSize = 8 * scale
    opts.maxFontSize = 8 * scale
    opts.padding = 0.03
    opts.fontFile = font_manager.findfont(font_manager.FontProperties(family="Arial"))
    opts.unspecifiedStereoIsUnknown = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, drawn)
    drawer.FinishDrawing()
    if canonical(Chem.MolToSmiles(drawn, isomericSmiles=True)) != canonical(Chem.MolToSmiles(mol, isomericSmiles=True)):
        raise ValueError("Depiction changed source chemical identity")
    return drawer.GetDrawingText(), list(mapping)


def draw_figure(rows, width_mm):
    width = width_mm / 25.4 * 72
    margin, gutter, row_height, top = 3, 13, 163, 27
    panel_width = (width - 2 * margin - gutter) / 2
    height = top + 3 * row_height + 2
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    for name, weight in (("Arial", "normal"), ("ArialBold", "bold")):
        font_file = font_manager.findfont(font_manager.FontProperties(family="Arial", weight=weight))
        page.insert_font(fontname=name, fontfile=font_file)

    def text(x, y, value, *, size=8, bold=False, color=(0, 0, 0)):
        page.insert_text((x, y), value, fontsize=size,
                         fontname="ArialBold" if bold else "Arial", color=color)

    text(margin, 12, "Dasatinib and the five highest-scoring audited analogs", size=9, bold=True)
    text(margin, 23, "Ranked by saved model-predicted EGFR KIBA score among 60 audit survivors", size=8)
    seed = Chem.MolFromSmiles(rows[0]["smiles"])
    rdDepictor.Compute2DCoords(seed, canonOrient=True)
    svgs, mappings = {}, {}
    for i, row in enumerate(rows):
        column, line = i % 2, i // 2
        x, y = margin + column * (panel_width + gutter), top + line * row_height
        ident = row["saved_id"]
        heading = (f"({row['panel']}) Dasatinib seed" if i == 0
                   else f"({row['panel']}) Rank {i}: {ident}")
        text(x, y + 10, heading, size=8.5, bold=True)
        mol = Chem.MolFromSmiles(row["smiles"])
        rect = fitz.Rect(x, y + 15, x + panel_width, y + 115)
        svg, mapping = molecule_svg(mol, seed, rect.width, rect.height)
        svg_doc = fitz.open(stream=svg.encode("utf-8"), filetype="svg")
        mol_pdf = fitz.open(stream=svg_doc.convert_to_pdf(), filetype="pdf")
        page.show_pdf_page(rect, mol_pdf, 0)
        svg_doc.close()
        mol_pdf.close()
        svgs[ident], mappings[ident] = svg, mapping
        affinity, delta = float(row["PredAffinity"]), row["delta_KIBA_vs_same_pass_seed"]
        text(x, y + 124, f"Predicted KIBA {affinity:.3f}   |   ΔKIBA {delta:+.3f}", bold=True)
        text(x, y + 135, f"ECFP Tanimoto {float(row['tanimoto']):.3f}   |   QED {float(row['QED']):.3f}   |   SA {float(row['SA']):.2f}")
        text(x, y + 146, f"Vina {float(row['vina_kcal_mol']):.3f} kcal/mol")
        lipinski = "pass" if float(row["LipinskiPass"]) else "fail"
        alerts = "pass" if float(row["AlertFree"]) else "fail"
        text(x, y + 157, f"Lipinski: {lipinski}   |   PAINS/BRENK: {alerts}", size=7.5)
        if line < 2:
            page.draw_line((x, y + row_height - 1), (x + panel_width, y + row_height - 1),
                           color=(0.75, 0.75, 0.75), width=0.4)
    pdf = RESULTS / f"{STEM}.pdf"
    doc.save(pdf, garbage=4, deflate=True)
    page.get_pixmap(dpi=DPI, alpha=False).save(RESULTS / f"{STEM}.png")
    # A small inspection raster is separate from the submission-resolution PNG.
    preview = ROOT / "IEEE_Access_Submission" / "tmp" / "pdfs" / f"{STEM}_preview.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(dpi=160, alpha=False).save(preview)
    font_sizes = sorted({round(span["size"], 3) for block in page.get_text("dict")["blocks"]
                         if "lines" in block for line in block["lines"] for span in line["spans"]})
    assert not page.get_images(full=True), "The PDF should contain vector drawings, not image objects"
    dimensions = {"width_mm": width_mm, "height_mm": height / 72 * 25.4,
                  "raster_dpi": DPI, "text_font_sizes_pt": font_sizes,
                  "structure_atom_font_size_pt": 8,
                  "layout": "two columns by three rows", "embedded_raster_images": 0,
                  "depiction_topology_mappings": mappings}
    doc.close()
    return dimensions, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-mm", type=float, default=177.53)
    args = parser.parse_args()
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()}
    rows, checks, reconciliation = select_records()
    dimensions, preview = draw_figure(rows, args.width_mm)
    source_csv = RESULTS / f"{STEM}_source.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unchanged = all(sha256(ROOT / path) == value for path, value in hashes.items())
    if not unchanged:
        raise RuntimeError("A source artifact changed during figure generation")
    provenance = {
        "selection": "Five highest saved PredAffinity values among the 60 nonseed diffusion analogs surviving the released four-motif audit. Ties would be resolved by canonical isomeric SMILES.",
        "ranking_metric": "Saved EGFR model prediction in KIBA-score units; higher first",
        "metric_join": "Unique exact raw SMILES and canonical isomeric SMILES matches across released, audited, comparison, and successful generated-analog docking rows",
        "source_id": "The displayed analog_000 to analog_004 identifiers are the saved docking identifiers, not new molecule names.",
        "seed_reconciliation": reconciliation,
        "stereochemistry": "All five analogs have two unspecified potential tetrahedral centers according to FindPotentialStereo, including dependent ring stereochemistry. No stereoisomers were assigned. Drawings preserve the source SMILES; unknown stereochemistry is shown with RDKit's unspecifiedStereoIsUnknown option.",
        "audit_scope": "The four-motif audit does not imply Lipinski or PAINS/BRENK compliance. Those original saved flags are displayed separately.",
        "docking_limit": "Vina values are saved docking scores for the archived 3D preparations. Source SMILES do not specify analog stereochemistry; these scores do not define or compare all possible stereoisomers.",
        "fingerprint": {"type": "Morgan/ECFP", "radius": 2, "bits": 2048,
                        "include_chirality": False, "reference": "dasatinib seed"},
        "rdkit_version": rdBase.rdkitVersion, "recomputed_predictions": False,
        "source_sha256": hashes, "original_sources_unchanged": unchanged,
        "independent_checks": checks, "export": dimensions,
        "source_csv": str(source_csv.relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
    }
    (RESULTS / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(RESULTS / f"{STEM}.{ext}") for ext in ("pdf", "png")],
                      "preview": str(preview), "dimensions": dimensions,
                      "seed_reconciliation": reconciliation}, indent=2))


if __name__ == "__main__":
    main()
