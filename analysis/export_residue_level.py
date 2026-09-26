"""
Export residue-level attention and contact labels for the 5-complex panel.

The main-text meta-analysis pools one AUROC per complex.  That summary throws
away the residue-level structure and forces the type I / type II question to be
asked post hoc on 2 complexes against 3.  This script re-runs the same saved
checkpoint over the same five complexes and writes one row per residue, so the
hypothesis can be tested directly in a mixed-effects model with complex as a
random effect and binding mode as a fixed effect.

Reads the cached PDB files and the cached ligand-graph / ESM-C protein caches
built by run_interpretability_benchmark.py, so no network access is required.

Output
------
  results/interpretability_residue_level.csv
      pdb_id, protein, binding_mode, residue_index, residue_label,
      attention, attention_z, contact

With --dump-matrix, additionally writes the full atom-by-residue attention
matrix per complex, which is the interaction map itself rather than the
per-residue profile derived from it:

  results/ibam_matrix_{pdb_id}.npz
      attention (n_atoms x n_residues), atom_symbol, residue_label, contact,
      pdb_id, protein, binding_mode, smiles

Run from the submission directory:  python export_residue_level.py
                                    python export_residue_level.py --dump-matrix

Reproducibility note
--------------------
This script is deterministic within one environment: rerunning it reproduces
its own output bit for bit.  Across environments it is not, because the forward
pass uses non-deterministic reductions whose order depends on the hardware and
the library build.  Rerunning the released checkpoint on a CPU workstation
reproduces the released per-residue attention to 1.1e-3 in absolute value, which
moves per-complex residue contact AUROC by at most 0.0016 and leaves every value
the manuscript reports at three decimals unchanged.  The released CSV is kept as
the artifact of record, so the published statistics stay tied to the run that
produced them.
"""

from __future__ import annotations

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


import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = _ROOT
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from config_profiles import get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import make_unlabeled_prediction_loader, predict_unlabeled
from case_studies_results_generation import (
    _attention_arrays_from_batches,
    choose_best_ligand_occurrence,
    extract_chain_residues,
    load_publication_ensemble,
    parse_pdb_records,
    residue_contact_mask,
)

PDB_DIR = ROOT / "results" / "downloads" / "interpretability_pdb"
CACHE_DIR = ROOT / "results" / "cache"
OUT = ROOT / "results" / "interpretability_residue_level.csv"

# Binding mode as annotated in the main text: imatinib (2HYY) and the annotated
# DFG-out ligand (4RJ3) are type II, the remaining three are type I
# ATP-competitive hinge binders.
PANEL = [
    ("6YOJ", "P4N", "FAK1", "I"),
    ("4WKQ", "IRE", "EGFR", "I"),
    ("2HYY", "STI", "ABL1", "II"),
    ("1KE6", "LS2", "CDK2", "I"),
    ("4RJ3", "3QS", "VEGFR2", "II"),
]
CUTOFF = 4.5


def cached_smiles(tag: str) -> str:
    manifest = json.loads(
        (CACHE_DIR / tag / f"{tag}_graphs_manifest.json").read_text(encoding="utf-8")
    )
    items = manifest["items"]
    return next(iter(items.values()))["smiles"]


def graph_atom_symbols(smiles: str, n_atoms: int) -> list:
    """Element symbols for the ligand's graph atoms, in model token order.

    The rows of the attention matrix are graph atoms built by RDKit from the
    SMILES, not the PDB atom records, and nothing guarantees those two orders
    agree.  Labelling rows with PDB atom names would therefore imply a
    correspondence that does not exist, so rows are labelled by element and
    graph index instead.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return [f"{i}" for i in range(n_atoms)]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != n_atoms:
        return [f"{i}" for i in range(n_atoms)]
    return [f"{a.GetSymbol()}{i}" for i, a in enumerate(mol.GetAtoms())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-matrix", action="store_true",
                        help="also write the full atom-by-residue matrix per "
                             "complex to results/ibam_matrix_{pdb_id}.npz")
    args = parser.parse_args()

    config = get_config_profile("max_rmse_cluster_diffusion")
    config.build_caches_on_start = False
    # Main-text analyses use the primary member with seed 1337, which is the
    # checkpoint released with the repository.
    config.ensemble_size = 1

    frames = []
    for pdb_id, resname, protein, mode in PANEL:
        tag = f"ibam_{pdb_id}_{resname}".lower()
        print(f"[complex] {pdb_id} {protein} + {resname} (type {mode})", flush=True)

        pdb_text = (PDB_DIR / f"{pdb_id}.pdb").read_text(encoding="utf-8")
        protein_atoms, ligand_groups = parse_pdb_records(pdb_text, ligand_resname=resname)
        _, ligand_atoms, chain_id = choose_best_ligand_occurrence(protein_atoms, ligand_groups)
        residues = extract_chain_residues(protein_atoms, chain_id)
        sequence = "".join(r["aa"] for r in residues)

        ligand_smiles = cached_smiles(tag)
        isolated_config, graph_cache, protein_cache = build_isolated_caches(
            config, [ligand_smiles], [sequence], str(CACHE_DIR / tag),
            force_rebuild=False, cache_prefix=tag,
        )
        models, normalizer = load_publication_ensemble(
            isolated_config, checkpoint_dir=config.checkpoint_dir
        )
        pair_df = pd.DataFrame([{"compound_iso_smiles": ligand_smiles,
                                 "target_sequence": sequence}])
        loader = make_unlabeled_prediction_loader(
            pair_df, graph_cache, protein_cache, isolated_config
        )
        payload = predict_unlabeled(
            models, loader, isolated_config, normalizer, collect_attention=True
        )
        atom_to_residue, _ = _attention_arrays_from_batches(payload["attention"])

        n_atoms, n_res = len(ligand_atoms), len(residues)
        atom_to_residue = atom_to_residue[:n_atoms, :n_res]

        # Same residue score the benchmark ranks on: mean ligand->residue attention.
        scores = atom_to_residue.mean(axis=0)
        contacts = residue_contact_mask(residues, ligand_atoms, cutoff=CUTOFF)

        if args.dump_matrix:
            labels = [f"{r['aa']}{r['position']}" for r in residues]
            npz = ROOT / "results" / f"ibam_matrix_{pdb_id}.npz"
            np.savez_compressed(
                npz,
                attention=atom_to_residue.astype(np.float32),
                atom_symbol=np.array(graph_atom_symbols(ligand_smiles, n_atoms)),
                residue_label=np.array(labels),
                contact=contacts.astype(np.int8),
                pdb_id=pdb_id, protein=protein, binding_mode=mode,
                smiles=ligand_smiles,
            )
            print(f"  wrote {npz.name}  ({n_atoms} atoms x {n_res} residues)",
                  flush=True)

        z = (scores - scores.mean()) / scores.std(ddof=1)
        frames.append(pd.DataFrame({
            "pdb_id": pdb_id,
            "protein": protein,
            "binding_mode": mode,
            "residue_index": np.arange(n_res),
            "residue_label": [f"{r['aa']}{r['position']}" for r in residues],
            "attention": scores,
            "attention_z": z,
            "contact": contacts.astype(int),
        }))
        print(f"  residues {n_res}, contacts {int(contacts.sum())}", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"\n[out] {OUT}  ({len(df)} residues, "
          f"{int(df['contact'].sum())} contacts)")


if __name__ == "__main__":
    main()
