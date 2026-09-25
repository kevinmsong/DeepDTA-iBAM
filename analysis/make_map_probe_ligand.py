"""Write probe ligands covering every Vina atom type a drug-like library uses.

Vina derives which affinity maps to precompute from the atom types present in
the ligands it is given, and Vina 1.2 types an atom by element *and* by its
hydrogen-bonding role (for example N_D for a donor nitrogen, N_A for an
acceptor, O_DA for a hydroxyl oxygen that is both).  A synthetic probe holding
one bare atom per element therefore misses most of the types a real library
needs.

These probes are small real molecules chosen so their union covers the donor,
acceptor, polar and hydrophobic roles for C, N, O, S and P, plus the four
halogens.  They are only used to precompute maps once with
`vina --write_maps`; they are never docked or scored.

Precomputing the maps turns each subsequent docking run from roughly 21 s into
roughly 2 s on this machine, because the grid is no longer rebuilt per ligand.

Run from anywhere:  python analysis/make_map_probe_ligand.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCK_DIR = ROOT / "results" / "docking"
PROBE_DIR = DOCK_DIR / "map_probes"

# name -> SMILES.  Between them these cover:
#   C_H / C_P, N_D / N_A / N_P, O_D / O_A / O_DA / O_P,
#   S_P, P_P, F_H, Cl_H, Br_H, I_H
PROBES = {
    "aminoethanol": "OCCN",
    "pyridine": "c1ccncc1",
    "acetamide": "CC(=O)N",
    "dimethylsulfide": "CSC",
    "methanesulfonamide": "CS(=O)(=O)N",
    "phosphate": "OP(=O)(O)O",
    "halobenzene": "Fc1cc(Cl)c(Br)c(I)c1",
    "imidazole": "c1c[nH]cn1",
    "anisole": "COc1ccccc1",
    "cyclohexane": "C1CCCCC1",
    "thiophenol": "Sc1ccccc1",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
}


def main() -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    RDLogger.DisableLog("rdApp.*")
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for old in PROBE_DIR.glob("*.pdbqt"):
        old.unlink()

    written = []
    for name, smiles in PROBES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  skipped {name}: unparseable")
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 1337
        if AllChem.EmbedMolecule(mol, params) != 0:
            print(f"  skipped {name}: embedding failed")
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
        setups = MoleculePreparation().prepare(mol)
        if not setups:
            print(f"  skipped {name}: no Meeko setup")
            continue
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            print(f"  skipped {name}: {err}")
            continue
        path = PROBE_DIR / f"{name}.pdbqt"
        path.write_text(pdbqt)
        written.append(path)

    types = set()
    for path in written:
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                types.add(line[77:].strip())
    print(f"wrote {len(written)} probe ligands to {PROBE_DIR}")
    print(f"AutoDock types present: {' '.join(sorted(types))}")


if __name__ == "__main__":
    main()
