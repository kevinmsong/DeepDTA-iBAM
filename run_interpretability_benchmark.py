"""
Interpretability benchmark for DeepDTA-iBAM.

For each protein-ligand complex in data/interpretability_benchmark.json:
  1. Download PDB and extract contact residues / atoms.
  2. Run model with attention collection.
  3. Compute enrichment metrics (AUROC, top-k overlap).
  4. Run perturbation tests: mask top-k vs random residues/atoms, measure score drop.
  5. Export per-complex heatmaps and residue barplots.
  6. Write summary CSV and boxplot figure.

Output
------
  results/interpretability_benchmark.csv
  results/figs/ibam_{pdb_id}_heatmap.{png,pdf}
  results/figs/ibam_{pdb_id}_barplot.{png,pdf}
  results/figs/ibam_summary_boxplot.{png,pdf}

Usage
-----
  python run_interpretability_benchmark.py
  python run_interpretability_benchmark.py --profile abl_full --checkpoint-dir results/ablations/abl_full/scaffold/seed_1337/checkpoints
  python run_interpretability_benchmark.py --topk 10 --force
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch

# Plotting (graceful fallback)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib/seaborn not available — figures will be skipped", flush=True)

from config_profiles import ExperimentConfig, get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import (
    load_ensemble,
    make_unlabeled_prediction_loader,
    predict_unlabeled,
)
from utils.metrics import auroc

# Import PDB parsing and attention utilities from the case studies module
from case_studies_results_generation import (
    _attention_arrays_from_batches,
    atom_contact_mask,
    choose_best_ligand_occurrence,
    compute_structure_alignment_metrics,
    extract_chain_residues,
    load_publication_ensemble,
    parse_pdb_records,
    residue_contact_mask,
)

# ---------------------------------------------------------------------------
# PDB download
# ---------------------------------------------------------------------------

_RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_pdb(pdb_id: str, cache_dir: Path, force: bool = False) -> str:
    """Download and cache a PDB file. Returns file text."""
    pdb_path = cache_dir / f"{pdb_id}.pdb"
    if pdb_path.exists() and not force:
        return pdb_path.read_text(encoding="utf-8")
    url = _RCSB_PDB_URL.format(pdb_id=pdb_id)
    print(f"  [pdb] downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    pdb_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# SMILES from first HETATM atom element heuristic + RDKit (best-effort)
# ---------------------------------------------------------------------------

def smiles_from_pdb_ligand(
    pdb_id: str, ligand_resname: str, pdb_text: str
) -> Optional[str]:
    """Fetch canonical SMILES from RCSB CCD for a given ligand residue name."""
    url = f"https://files.rcsb.org/ligands/download/{ligand_resname}_ideal.sdf"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            sdf_text = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"  [warn] failed to fetch SDF for {ligand_resname}: {exc}", flush=True)
        return None
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier()
        suppl.SetData(sdf_text)
        mol = next(iter(suppl), None)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as exc:
        print(f"  [warn] RDKit conversion failed for {ligand_resname}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Perturbation testing
# ---------------------------------------------------------------------------

@torch.no_grad()
def _masked_predict(
    models: Sequence[Any],
    base_batch: Dict[str, Any],
    config: ExperimentConfig,
    normalizer: Any,
    *,
    atom_mask_indices: Optional[List[int]] = None,
    residue_mask_indices: Optional[List[int]] = None,
) -> float:
    """
    Run ensemble prediction on a batch with specific atom/residue positions
    zeroed out.  Returns the ensemble-mean predicted affinity (denormalized).
    """
    from training.engine import _autocast_context, _denormalize_predictions, runtime_device

    device = runtime_device(config)
    batch = {
        k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
        for k, v in base_batch.items()
    }

    if atom_mask_indices is not None:
        for idx in atom_mask_indices:
            batch["drug_x"][0, idx, :] = 0.0

    if residue_mask_indices is not None:
        for idx in residue_mask_indices:
            batch["protein_embeddings"][0, idx, :] = 0.0

    preds = []
    for model in models:
        model.eval()
        with _autocast_context(config):
            output, _, _ = model(
                batch["drug_x"],
                batch["drug_adj"],
                batch["drug_mask"],
                batch["protein_embeddings"],
                batch["protein_mask"],
                drug_edge_features=batch["drug_edge_features"],
                compute_diff_loss=False,
            )
        preds.append(_denormalize_predictions(
            output.detach().float().cpu(), normalizer, config.normalize_targets
        ).item())
    return float(np.mean(preds))


def run_perturbation_test(
    models: Sequence[Any],
    base_batch: Dict[str, Any],
    config: ExperimentConfig,
    normalizer: Any,
    residue_scores: np.ndarray,
    atom_scores: np.ndarray,
    topk: int,
) -> Dict[str, float]:
    """
    Compute perturbation-based signal quality.

    Returns score drops for top-k vs random masking of residues and atoms.
    A larger gap (top-k drop >> random drop) indicates the model's high-attention
    positions are functionally important.
    """
    rng = np.random.default_rng(seed=0)

    # Baseline
    baseline = _masked_predict(models, base_batch, config, normalizer)

    # Top-k residue mask
    top_residue_idx = np.argsort(residue_scores)[::-1][:topk].tolist()
    rnd_residue_idx = rng.choice(len(residue_scores), size=topk, replace=False).tolist()

    score_topk_residue = _masked_predict(
        models, base_batch, config, normalizer, residue_mask_indices=top_residue_idx
    )
    score_rnd_residue = _masked_predict(
        models, base_batch, config, normalizer, residue_mask_indices=rnd_residue_idx
    )

    # Top-k atom mask
    top_atom_idx = np.argsort(atom_scores)[::-1][:topk].tolist()
    rnd_atom_idx = rng.choice(len(atom_scores), size=topk, replace=False).tolist()

    score_topk_atom = _masked_predict(
        models, base_batch, config, normalizer, atom_mask_indices=top_atom_idx
    )
    score_rnd_atom = _masked_predict(
        models, base_batch, config, normalizer, atom_mask_indices=rnd_atom_idx
    )

    return {
        "baseline_score": baseline,
        "score_drop_topk_residue": baseline - score_topk_residue,
        "score_drop_random_residue": baseline - score_rnd_residue,
        "score_drop_topk_atom": baseline - score_topk_atom,
        "score_drop_random_atom": baseline - score_rnd_atom,
        "residue_mask_signal": (baseline - score_topk_residue) - (baseline - score_rnd_residue),
        "atom_mask_signal": (baseline - score_topk_atom) - (baseline - score_rnd_atom),
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

_DPI_PNG = 600
_DPI_PDF = 300


def _save_fig(fig: Any, path_stem: Path) -> None:
    if not HAS_PLOT:
        return
    fig.savefig(str(path_stem) + ".png", dpi=_DPI_PNG, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".pdf", dpi=_DPI_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path_stem}.png/.pdf", flush=True)


def plot_complex_figures(
    pdb_id: str,
    atom_to_residue: np.ndarray,
    residue_to_atom: np.ndarray,
    residue_labels: List[str],
    atom_labels: List[str],
    residue_contacts: np.ndarray,
    figs_dir: Path,
) -> None:
    if not HAS_PLOT:
        return

    # --- Heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(atom_labels) * 0.35 + 2)))
    tick_n = min(14, len(residue_labels))
    tick_pos = np.linspace(0, len(residue_labels) - 1, tick_n, dtype=int)

    sns.heatmap(atom_to_residue, ax=axes[0], cmap="mako",
                cbar_kws={"label": "Mean attention"})
    axes[0].set_title(f"{pdb_id}: ligand → protein attention", fontsize=13)
    axes[0].set_xlabel("Residues")
    axes[0].set_ylabel("Ligand atoms")
    axes[0].set_xticks(tick_pos + 0.5)
    axes[0].set_xticklabels([residue_labels[i] for i in tick_pos], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(np.arange(len(atom_labels)) + 0.5)
    axes[0].set_yticklabels(atom_labels, rotation=0, fontsize=8)

    sns.heatmap(residue_to_atom, ax=axes[1], cmap="crest",
                cbar_kws={"label": "Mean attention"})
    axes[1].set_title(f"{pdb_id}: protein → ligand attention", fontsize=13)
    axes[1].set_xlabel("Ligand atoms")
    axes[1].set_ylabel("Residues")
    axes[1].set_xticks(np.arange(len(atom_labels)) + 0.5)
    axes[1].set_xticklabels(atom_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(tick_pos + 0.5)
    axes[1].set_yticklabels([residue_labels[i] for i in tick_pos], rotation=0, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, figs_dir / f"ibam_{pdb_id}_heatmap")

    # --- Residue barplot with contact overlay ---
    residue_scores = atom_to_residue.mean(axis=0)
    top_n = min(20, len(residue_labels))
    top_idx = np.argsort(residue_scores)[::-1][:top_n][::-1]
    colors = ["#ef4444" if residue_contacts[i] else "#3b82f6" for i in top_idx]

    fig2, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35 + 1.5)))
    ax.barh([residue_labels[i] for i in top_idx], residue_scores[top_idx], color=colors)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#ef4444", label="Contact residue"),
                        Patch(color="#3b82f6", label="Non-contact")], fontsize=9)
    ax.set_xlabel("Mean residue attention")
    ax.set_title(f"{pdb_id}: top {top_n} residues by attention vs. contacts", fontsize=13)
    fig2.tight_layout()
    _save_fig(fig2, figs_dir / f"ibam_{pdb_id}_barplot")


def plot_summary_boxplot(results: List[Dict[str, Any]], figs_dir: Path) -> None:
    if not HAS_PLOT or not results:
        return

    metrics = ["residue_contact_auroc", "atom_contact_auroc",
               "residue_topk_overlap", "atom_topk_overlap"]
    labels = ["Residue AUROC", "Atom AUROC", "Residue top-k", "Atom top-k"]
    data = [[r[m] for r in results if m in r] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("#93c5fd")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_ylabel("Score")
    ax.set_title("iBAM interpretability benchmark — enrichment across complexes", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, figs_dir / "ibam_summary_boxplot")


# ---------------------------------------------------------------------------
# Per-complex runner
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = get_config_profile(args.profile)
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.device is not None:
        config.device = args.device
    config.build_caches_on_start = False
    return config


def run_complex(
    entry: Dict[str, Any],
    config: ExperimentConfig,
    pdb_cache_dir: Path,
    results_dir: Path,
    figs_dir: Path,
    topk: int,
    force: bool,
) -> Optional[Dict[str, Any]]:
    pdb_id: str = entry["pdb_id"]
    ligand_resname: str = entry["ligand_resname"]
    protein_name: str = entry.get("protein_name", pdb_id)
    print(f"\n[complex] {pdb_id} — {protein_name} + {ligand_resname}", flush=True)

    # 1. Download PDB
    try:
        pdb_text = fetch_pdb(pdb_id, pdb_cache_dir, force=force)
    except Exception as exc:
        print(f"  [skip] PDB download failed: {exc}", flush=True)
        return None

    # 2. Parse ligand and protein
    try:
        protein_atoms, ligand_groups = parse_pdb_records(pdb_text, ligand_resname=ligand_resname)
        if not ligand_groups:
            print(f"  [skip] no HETATM records found for resname={ligand_resname}", flush=True)
            return None
        ligand_key, ligand_atoms, chain_id = choose_best_ligand_occurrence(
            protein_atoms, ligand_groups
        )
        residues = extract_chain_residues(protein_atoms, chain_id)
    except Exception as exc:
        print(f"  [skip] PDB parsing failed: {exc}", flush=True)
        return None

    if not residues:
        print(f"  [skip] no protein residues found for chain {chain_id}", flush=True)
        return None

    sequence = "".join(r["aa"] for r in residues)
    print(f"  sequence length: {len(sequence)}, ligand atoms: {len(ligand_atoms)}", flush=True)

    # 3. Fetch ligand SMILES
    ligand_smiles = smiles_from_pdb_ligand(pdb_id, ligand_resname, pdb_text)
    if ligand_smiles is None:
        print(f"  [skip] could not obtain SMILES for {ligand_resname}", flush=True)
        return None

    # 4. Build isolated caches and run inference
    cache_tag = f"ibam_{pdb_id}_{ligand_resname}".lower()
    try:
        isolated_config, graph_cache, protein_cache = build_isolated_caches(
            config,
            [ligand_smiles],
            [sequence],
            str(results_dir / "cache" / cache_tag),
            force_rebuild=force,
            cache_prefix=cache_tag,
        )
    except Exception as exc:
        print(f"  [skip] cache build failed: {exc}", flush=True)
        return None

    models, normalizer = load_publication_ensemble(
        isolated_config, checkpoint_dir=config.checkpoint_dir
    )
    pair_df = pd.DataFrame([{
        "compound_iso_smiles": ligand_smiles,
        "target_sequence": sequence,
    }])
    loader = make_unlabeled_prediction_loader(
        pair_df, graph_cache, protein_cache, isolated_config
    )
    payload = predict_unlabeled(
        models, loader, isolated_config, normalizer, collect_attention=True
    )

    # 5. Extract attention
    try:
        atom_to_residue, residue_to_atom = _attention_arrays_from_batches(payload["attention"])
    except Exception as exc:
        print(f"  [skip] attention extraction failed: {exc}", flush=True)
        return None

    # Trim to actual sequence length (attention map may be padded)
    n_atoms = len(ligand_atoms)
    n_residues = len(residues)
    atom_to_residue = atom_to_residue[:n_atoms, :n_residues]
    residue_to_atom = residue_to_atom[:n_residues, :n_atoms]

    # 6. Contact masks
    residue_contacts = residue_contact_mask(
        residues, ligand_atoms, cutoff=entry.get("contact_cutoff_angstrom", 4.5)
    )
    atom_contacts = atom_contact_mask(
        ligand_atoms, residues, cutoff=entry.get("contact_cutoff_angstrom", 4.5)
    )
    n_contact_residues = int(residue_contacts.sum())
    n_contact_atoms = int(atom_contacts.sum())
    print(f"  contact residues: {n_contact_residues}, contact atoms: {n_contact_atoms}", flush=True)

    if n_contact_residues == 0 or n_contact_atoms == 0:
        print(f"  [warn] no contacts found at cutoff {entry.get('contact_cutoff_angstrom', 4.5)} Å", flush=True)

    # 7. Enrichment metrics
    metrics = compute_structure_alignment_metrics(
        atom_to_residue, residue_to_atom, residue_contacts, atom_contacts
    )
    metrics["predicted_affinity"] = float(payload["predictions"][0])
    print(f"  residue_contact_auroc={metrics['residue_contact_auroc']:.3f}  "
          f"atom_contact_auroc={metrics['atom_contact_auroc']:.3f}", flush=True)

    # 8. Perturbation test
    residue_scores = atom_to_residue.mean(axis=0)
    atom_scores = residue_to_atom.mean(axis=0)
    try:
        base_batch = next(iter(loader))
        perturb = run_perturbation_test(
            models, base_batch, isolated_config, normalizer,
            residue_scores, atom_scores,
            topk=min(topk, n_residues, n_atoms),
        )
        metrics.update(perturb)
        print(f"  residue_mask_signal={perturb['residue_mask_signal']:.4f}  "
              f"atom_mask_signal={perturb['atom_mask_signal']:.4f}", flush=True)
    except Exception as exc:
        print(f"  [warn] perturbation test failed: {exc}", flush=True)

    # 9. Figures
    residue_labels = [
        f"{r['aa']}{r['position']}" for r in residues
    ]
    atom_labels = [
        f"{a.get('element', 'X')}{i+1}" for i, a in enumerate(ligand_atoms)
    ]
    figs_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_complex_figures(
            pdb_id, atom_to_residue, residue_to_atom,
            residue_labels, atom_labels, residue_contacts, figs_dir,
        )
    except Exception as exc:
        print(f"  [warn] figure generation failed: {exc}", flush=True)

    result = {
        "pdb_id": pdb_id,
        "ligand": ligand_resname,
        "protein": protein_name,
        "n_residues": n_residues,
        "n_atoms": n_atoms,
        "n_contact_residues": n_contact_residues,
        "n_contact_atoms": n_contact_atoms,
    }
    result.update(metrics)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark-json", default="data/interpretability_benchmark.json",
                        help="Path to benchmark complex definitions.")
    parser.add_argument("--profile", default="max_rmse_cluster_diffusion",
                        help="Model profile to load for evaluation.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory.")
    parser.add_argument("--device", default=None, help="Device override.")
    parser.add_argument("--topk", type=int, default=10,
                        help="k for top-k perturbation masking.")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for CSV and figures.")
    parser.add_argument("--pdb-cache-dir", default="data/pdb_cache",
                        help="Local PDB file cache directory.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs exist; re-download PDB files.")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    figs_dir = results_dir / "figs"
    pdb_cache_dir = Path(args.pdb_cache_dir)
    pdb_cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)

    benchmark = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    print(f"[bench] {len(benchmark)} complexes", flush=True)

    all_results: List[Dict[str, Any]] = []
    for entry in benchmark:
        result = run_complex(
            entry, config, pdb_cache_dir, results_dir, figs_dir,
            topk=args.topk, force=args.force,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("[bench] no results — check PDB downloads and model checkpoint", flush=True)
        sys.exit(1)

    # Write CSV
    csv_path = results_dir / "interpretability_benchmark.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[out] {csv_path}", flush=True)

    # Summary boxplot
    try:
        plot_summary_boxplot(all_results, figs_dir)
    except Exception as exc:
        print(f"[warn] summary boxplot failed: {exc}", flush=True)

    # Print summary table
    display_cols = [
        "pdb_id", "protein", "ligand",
        "residue_contact_auroc", "atom_contact_auroc",
        "residue_topk_overlap", "atom_topk_overlap",
        "residue_mask_signal", "atom_mask_signal",
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    print("\n=== Benchmark Summary ===")
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
