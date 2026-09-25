"""Prepare the EGFR (4WKQ) receptor and search box for AutoDock Vina.

Splits the cached 4WKQ co-crystal into a protein-only receptor and the
crystallographic erlotinib ligand (residue IRE), converts the receptor to
PDBQT with Open Babel, and writes the search box centred on the ligand
centroid.

Run from anywhere:  python analysis/prep_docking_receptor.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "results" / "downloads" / "interpretability_pdb" / "4WKQ.pdb"
OUT = ROOT / "results" / "docking"
OUT.mkdir(parents=True, exist_ok=True)

LIGAND_RESNAME = "IRE"
BOX_PADDING = 8.0  # angstrom beyond the crystallographic ligand extent


def main() -> None:
    protein_lines, ligand_lines = [], []
    for line in PDB.read_text().splitlines():
        rec = line[:6]
        if rec == "ATOM  ":
            protein_lines.append(line)
        elif rec == "HETATM" and line[17:20].strip() == LIGAND_RESNAME:
            ligand_lines.append(line)

    if not ligand_lines:
        raise SystemExit(f"no {LIGAND_RESNAME} ligand found in {PDB}")

    rec_pdb = OUT / "receptor_4wkq.pdb"
    rec_pdb.write_text("\n".join(protein_lines) + "\nEND\n")
    lig_pdb = OUT / "crystal_ligand_ire.pdb"
    lig_pdb.write_text("\n".join(ligand_lines) + "\nEND\n")

    coords = [
        (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        for l in ligand_lines
        if l[76:78].strip() != "H"
    ]
    xs, ys, zs = zip(*coords)
    center = [
        round((min(a) + max(a)) / 2.0, 3) for a in (xs, ys, zs)
    ]
    size = [
        round(max(max(a) - min(a) + 2 * BOX_PADDING, 18.0), 3) for a in (xs, ys, zs)
    ]

    rec_pdbqt = OUT / "receptor_4wkq.pdbqt"
    subprocess.run(
        ["obabel", str(rec_pdb), "-O", str(rec_pdbqt), "-xr", "-p", "7.4", "--partialcharge", "gasteiger"],
        check=True,
        capture_output=True,
    )

    box = {
        "pdb_id": "4WKQ",
        "ligand_resname": LIGAND_RESNAME,
        "n_ligand_heavy_atoms": len(coords),
        "center_x": center[0], "center_y": center[1], "center_z": center[2],
        "size_x": size[0], "size_y": size[1], "size_z": size[2],
        "box_padding_angstrom": BOX_PADDING,
        "receptor_pdbqt": rec_pdbqt.name,
    }
    (OUT / "docking_box.json").write_text(json.dumps(box, indent=2))

    n_rec_atoms = sum(1 for l in rec_pdbqt.read_text().splitlines() if l.startswith(("ATOM", "HETATM")))
    print(f"receptor atoms in pdbqt : {n_rec_atoms}")
    print(f"ligand heavy atoms      : {len(coords)}")
    print(f"box center              : {center}")
    print(f"box size                : {size}")


if __name__ == "__main__":
    main()
