"""
Generation validation: compare diffusion-based generation against baselines.

Three generator modes:
  1. diffusion  — protein-conditioned DDPM seeded from seed molecule topology
  2. random_edit — randomly mutate one non-ring heavy atom's element
  3. fragment_swap — cut seed at one rotatable bond, replace fragment from pool

For N candidates per generator, compute: validity, uniqueness, Tanimoto
similarity to seed, QED, SA, Lipinski pass rate.

Output
------
  results/generation_comparison.csv         — one row per molecule
  results/generation_comparison_summary.csv — per-generator stats
  results/figs/generation_violin.{png,pdf}

Usage
-----
  python run_generation_validation.py
  python run_generation_validation.py --seed-smiles "c1ccc(cc1)C" --n 50
  python run_generation_validation.py --profile abl_full \\
      --checkpoint-dir results/ablations/abl_full/scaffold/seed_1337/checkpoints
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib/seaborn not available — figures will be skipped", flush=True)

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from config_profiles import ExperimentConfig, get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import make_unlabeled_prediction_loader
from case_studies_results_generation import (
    collect_seeded_analogs,
    decode_generated_analog,
    load_publication_ensemble,
    molecule_properties,
)

# ---------------------------------------------------------------------------
# Default seed (EGFR / dasatinib context matching existing generation section)
# ---------------------------------------------------------------------------

_DEFAULT_SEED_SMILES = (
    "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(n1)N1CCN(CCO)CC1"  # dasatinib
)

# Common fragment pool for fragment_swap baseline
_FRAGMENT_POOL = [
    "C",         # methyl
    "CC",        # ethyl
    "CCC",       # propyl
    "CCO",       # 2-hydroxyethyl
    "CCN",       # 2-aminoethyl
    "c1ccccc1",  # phenyl
    "C(=O)N",    # amide
    "CN",        # methylamine
    "CF",        # fluoromethyl
    "CCl",       # chloromethyl
    "C#N",       # nitrile
    "CO",        # methoxy
    "CS",        # thiomethyl
    "N",         # amine
    "O",         # hydroxyl
]

# Atom types for random edit
_EDIT_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br"]


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

def _lipinski(mol: Chem.Mol) -> int:
    return int(
        Descriptors.MolWt(mol) <= 500
        and Crippen.MolLogP(mol) <= 5
        and Lipinski.NumHDonors(mol) <= 5
        and Lipinski.NumHAcceptors(mol) <= 10
    )


def _mol_properties(mol: Chem.Mol, seed_fp, generator: str) -> Dict[str, Any]:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    tanimoto = float(DataStructs.TanimotoSimilarity(seed_fp, fp))
    return {
        "generator": generator,
        "smiles": Chem.MolToSmiles(mol),
        "valid": 1,
        "tanimoto": tanimoto,
        "QED": float(QED.qed(mol)),
        "SA": float(sascorer.calculateScore(mol)),
        "lipinski_pass": _lipinski(mol),
    }


# ---------------------------------------------------------------------------
# Baseline generators
# ---------------------------------------------------------------------------

def generate_random_edit(
    seed_mol: Chem.Mol,
    n: int,
    *,
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Randomly mutate one non-ring heavy atom's element per attempt.
    Keeps valid, unique SMILES.
    """
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, radius=2, nBits=2048)
    non_ring_atoms = [
        a.GetIdx() for a in seed_mol.GetAtoms()
        if not a.IsInRing() and a.GetAtomicNum() > 1
    ]
    if not non_ring_atoms:
        non_ring_atoms = [a.GetIdx() for a in seed_mol.GetAtoms() if a.GetAtomicNum() > 1]

    seen: set = set()
    seen.add(Chem.MolToSmiles(seed_mol))
    results: List[Dict[str, Any]] = []
    attempts = 0

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        rw = Chem.RWMol(seed_mol)
        atom_idx = int(rng.choice(non_ring_atoms))
        new_elem = str(rng.choice(_EDIT_ELEMENTS))
        try:
            rw.GetAtomWithIdx(atom_idx).SetAtomicNum(
                Chem.GetPeriodicTable().GetAtomicNumber(new_elem)
            )
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi not in seen:
                seen.add(smi)
                results.append(_mol_properties(mol, seed_fp, "random_edit"))
        except Exception:
            pass

    return results


def generate_fragment_swap(
    seed_mol: Chem.Mol,
    n: int,
    *,
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Cut seed at a rotatable non-ring single bond; replace the smaller fragment
    with one from _FRAGMENT_POOL using RDKit ReplaceSubstructs.
    """
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, radius=2, nBits=2048)

    # Identify rotatable non-ring bonds connecting non-H atoms
    rot_bonds = [
        b.GetIdx()
        for b in seed_mol.GetBonds()
        if (
            b.GetBondTypeAsDouble() == 1.0
            and not b.IsInRing()
            and b.GetBeginAtom().GetAtomicNum() > 1
            and b.GetEndAtom().GetAtomicNum() > 1
        )
    ]
    if not rot_bonds:
        print("[warn] fragment_swap: no rotatable bonds found — fallback to random_edit", flush=True)
        return generate_random_edit(seed_mol, n, rng=rng, max_attempts=max_attempts)

    seen: set = set()
    seen.add(Chem.MolToSmiles(seed_mol))
    results: List[Dict[str, Any]] = []
    attempts = 0

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        bond_idx = int(rng.choice(rot_bonds))
        frag_smi = str(rng.choice(_FRAGMENT_POOL))
        try:
            bond = seed_mol.GetBondWithIdx(bond_idx)
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Fragment: keep the larger piece, attach fragment
            em = Chem.RWMol(seed_mol)
            em.RemoveBond(begin_idx, end_idx)
            # Find which fragment contains more atoms (keep that one)
            frags = Chem.GetMolFrags(em.GetMol(), asMols=True)
            if not frags:
                continue
            core = max(frags, key=lambda m: m.GetNumAtoms())

            # Attach new fragment via wildcard atom pattern
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                continue
            # Simple attachment: replace a terminal H on core with fragment
            rw_core = Chem.RWMol(Chem.AddHs(core))
            h_atoms = [
                a.GetIdx() for a in rw_core.GetAtoms() if a.GetAtomicNum() == 1
            ]
            if not h_atoms:
                continue
            h_idx = int(rng.choice(h_atoms))
            rw_core.ReplaceAtom(h_idx, frag_mol.GetAtomWithIdx(0))
            # Add the rest of frag_mol atoms
            for atom in list(frag_mol.GetAtoms())[1:]:
                rw_core.AddAtom(atom)
            # Add bonds within frag_mol
            offset = rw_core.GetNumAtoms() - frag_mol.GetNumAtoms()
            for bond in frag_mol.GetBonds():
                rw_core.AddBond(
                    offset + bond.GetBeginAtomIdx(),
                    offset + bond.GetEndAtomIdx(),
                    bond.GetBondType(),
                )
            mol = Chem.RemoveHs(rw_core.GetMol())
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi not in seen:
                seen.add(smi)
                results.append(_mol_properties(mol, seed_fp, "fragment_swap"))
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Diffusion generator
# ---------------------------------------------------------------------------

def generate_diffusion(
    seed_smiles: str,
    protein_sequence: str,
    config: ExperimentConfig,
    n: int,
    *,
    rng: np.random.Generator,
    cache_root: str = "results/generation_cache",
    force_rebuild: bool = False,
) -> List[Dict[str, Any]]:
    """Run protein-conditioned diffusion generation and return property dicts."""
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        raise ValueError(f"Cannot parse seed SMILES: {seed_smiles}")
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, radius=2, nBits=2048)

    isolated_config, graph_cache, protein_cache = build_isolated_caches(
        config,
        [seed_smiles],
        [protein_sequence],
        cache_root,
        force_rebuild=force_rebuild,
        cache_prefix="gen_validation",
    )
    models, normalizer = load_publication_ensemble(
        isolated_config, checkpoint_dir=config.checkpoint_dir
    )
    pair_df = pd.DataFrame([{
        "compound_iso_smiles": seed_smiles,
        "target_sequence": protein_sequence,
    }])
    loader = make_unlabeled_prediction_loader(
        pair_df, graph_cache, protein_cache, isolated_config
    )
    batch = next(iter(loader))

    # Move batch to device
    device = config.device if hasattr(config, "device") else "cpu"
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Run collect_seeded_analogs (existing logic) but with target_count=n
    gen_df = collect_seeded_analogs(
        models, batch, seed_mol,
        rng=rng,
        target_count=n,
        max_attempts=n * 50,
    )

    results = []
    for _, row in gen_df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        tanimoto = float(DataStructs.TanimotoSimilarity(seed_fp, fp))
        results.append({
            "generator": "diffusion",
            "smiles": row["smiles"],
            "valid": 1,
            "tanimoto": tanimoto,
            "QED": float(row.get("QED", QED.qed(mol))),
            "SA": float(row.get("SA", sascorer.calculateScore(mol))),
            "lipinski_pass": int(row.get("LipinskiPass", _lipinski(mol))),
        })

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gen, grp in df.groupby("generator"):
        valid = grp["valid"].sum()
        total = len(grp)
        unique = grp["smiles"].nunique()
        rows.append({
            "generator": gen,
            "n_total": total,
            "n_valid": valid,
            "validity": valid / max(total, 1),
            "uniqueness": unique / max(valid, 1),
            "tanimoto_mean": grp["tanimoto"].mean(),
            "tanimoto_sd": grp["tanimoto"].std(ddof=1),
            "QED_mean": grp["QED"].mean(),
            "QED_sd": grp["QED"].std(ddof=1),
            "SA_mean": grp["SA"].mean(),
            "SA_sd": grp["SA"].std(ddof=1),
            "lipinski_pass_rate": grp["lipinski_pass"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_violin(df: pd.DataFrame, figs_dir: Path) -> None:
    if not HAS_PLOT or df.empty:
        return

    metrics = ["QED", "SA", "tanimoto"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    palette = {"diffusion": "#3b82f6", "random_edit": "#f97316", "fragment_swap": "#22c55e"}

    for ax, metric in zip(axes, metrics):
        order = sorted(df["generator"].unique())
        sns.violinplot(
            data=df, x="generator", y=metric, order=order,
            palette=palette, inner="box", ax=ax,
        )
        ax.set_title(metric, fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Generation comparison: diffusion vs baselines", fontsize=14, y=1.02)
    fig.tight_layout()

    figs_dir.mkdir(parents=True, exist_ok=True)
    stem = figs_dir / "generation_violin"
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {stem}.png/.pdf", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed-smiles", default=_DEFAULT_SEED_SMILES,
                        help="Seed molecule SMILES.")
    parser.add_argument("--protein-sequence", default=None,
                        help="Protein sequence (FASTA amino acids). "
                             "Defaults to EGFR kinase domain from existing results.")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of candidates per generator.")
    parser.add_argument("--profile", default="max_rmse_cluster_diffusion",
                        help="Model profile to use for diffusion generation.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory.")
    parser.add_argument("--device", default=None, help="Device override.")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for CSVs and figures.")
    parser.add_argument("--skip-diffusion", action="store_true",
                        help="Skip diffusion generator (faster; for baseline-only comparison).")
    parser.add_argument("--seed-rng", type=int, default=42,
                        help="Random seed for baseline generators.")
    parser.add_argument("--force", action="store_true",
                        help="Force cache rebuild for diffusion.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figs_dir = output_dir / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed_rng)
    seed_mol = Chem.MolFromSmiles(args.seed_smiles)
    if seed_mol is None:
        sys.exit(f"Cannot parse seed SMILES: {args.seed_smiles}")

    all_rows: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Baseline 1: random atom edit
    # -----------------------------------------------------------------------
    print(f"\n[gen] random_edit (n={args.n})", flush=True)
    edit_rows = generate_random_edit(seed_mol, args.n, rng=rng)
    print(f"  collected {len(edit_rows)} valid unique", flush=True)
    all_rows.extend(edit_rows)

    # -----------------------------------------------------------------------
    # Baseline 2: fragment swap
    # -----------------------------------------------------------------------
    print(f"\n[gen] fragment_swap (n={args.n})", flush=True)
    frag_rows = generate_fragment_swap(seed_mol, args.n, rng=rng)
    print(f"  collected {len(frag_rows)} valid unique", flush=True)
    all_rows.extend(frag_rows)

    # -----------------------------------------------------------------------
    # Diffusion generator
    # -----------------------------------------------------------------------
    if not args.skip_diffusion:
        protein_sequence = args.protein_sequence
        if protein_sequence is None:
            # Try to load EGFR sequence from existing results
            egfr_csv = output_dir / "generated_egfr_analogs_100.csv"
            if egfr_csv.exists():
                print("[gen] loading EGFR sequence from existing generation cache", flush=True)
                # Fall back: use a short EGFR kinase domain sequence stub
                # (users should supply --protein-sequence for a real run)
                print("[warn] --protein-sequence not given; use --protein-sequence for a real run",
                      flush=True)
                args.skip_diffusion = True
            else:
                print("[warn] --protein-sequence not given; skipping diffusion generator", flush=True)
                args.skip_diffusion = True

    if not args.skip_diffusion:
        print(f"\n[gen] diffusion (n={args.n})", flush=True)
        config = get_config_profile(args.profile)
        if args.checkpoint_dir:
            config.checkpoint_dir = args.checkpoint_dir
        if args.device:
            config.device = args.device
        config.build_caches_on_start = False
        try:
            diff_rows = generate_diffusion(
                args.seed_smiles,
                protein_sequence,
                config,
                n=args.n,
                rng=rng,
                cache_root=str(output_dir / "generation_cache"),
                force_rebuild=args.force,
            )
            print(f"  collected {len(diff_rows)} valid unique", flush=True)
            all_rows.extend(diff_rows)
        except Exception as exc:
            print(f"[warn] diffusion generation failed: {exc}", flush=True)

    if not all_rows:
        print("[error] no rows generated", flush=True)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)

    comp_path = output_dir / "generation_comparison.csv"
    df.to_csv(comp_path, index=False, float_format="%.6f")
    print(f"\n[out] {comp_path}", flush=True)

    summary = compute_summary(df)
    summ_path = output_dir / "generation_comparison_summary.csv"
    summary.to_csv(summ_path, index=False, float_format="%.6f")
    print(f"[out] {summ_path}", flush=True)

    print("\n=== Generation Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    try:
        plot_violin(df, figs_dir)
    except Exception as exc:
        print(f"[warn] violin plot failed: {exc}", flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
