"""Revalidate PDB atom-contact labels in the model's exact graph order.

Original result files remain unchanged. The archived structural benchmark
compared PDB-order contact labels with canonical-SMILES-order attention scores.
This script enumerates every element/bond-preserving mapping, retains all
distinct resulting label vectors, and reports metric ranges if labels differ
under molecular symmetry. It never selects a mapping by attention performance.

Only complexes with both atom-label classes need a new forward pass. Saturated
cases have undefined AUROC and unit top-k overlap for every ranking. Cached
graphs/protein embeddings are loaded directly, without rebuilding any cache.

Run: python analysis/revalidate_atom_contacts.py
Outputs: results/atom_contacts_revalidated.{csv,json,md,npz} and
         results/atom_contacts_revalidated_mapping.csv
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from scipy import stats
from sklearn.metrics import roc_auc_score

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from case_studies_results_generation import (  # noqa: E402
    _attention_arrays_from_batches,
    atom_contact_mask,
    choose_best_ligand_occurrence,
    extract_chain_residues,
    parse_pdb_records,
    residue_contact_mask,
)
from config_profiles import get_config_profile  # noqa: E402
from data.cache_builders import GraphCache, ProteinEmbeddingCache  # noqa: E402
from models.rmse_model import smiles_to_graph  # noqa: E402
from training.inference import load_ensemble, make_unlabeled_prediction_loader, predict_unlabeled  # noqa: E402
from utils.structure_alignment import ligand_mol_from_pdb, mapped_contact_vectors  # noqa: E402

PROTEIN_NAMES = {"1KE6": "CDK2", "2HYY": "ABL1", "4RJ3": "CDK2",
                 "4WKQ": "EGFR", "6YOJ": "FAK1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atom_metrics(labels, scores):
    k = int(labels.sum())
    # Preserve the released benchmark's deterministic top-k ranking convention.
    ranked = np.argsort(scores)[::-1]
    overlap = float(labels[ranked[:k]].sum()) / k
    auroc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None
    return {"auroc": auroc, "topk_overlap": overlap}


def t_interval(values):
    x = np.asarray(values, dtype=float)
    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return [float(x.mean() - half), float(x.mean() + half)]


def main():
    cfg = get_config_profile("max_rmse_cluster_diffusion")
    cfg.device, cfg.use_amp, cfg.num_workers = "cpu", False, 0
    cfg.ensemble_size, cfg.max_pairs_per_batch = 1, 1
    cfg.build_caches_on_start = False
    checkpoint = ROOT / "checkpoints" / f"{cfg.profile_name}_member_0.safetensors"
    originals = [RESULTS / "interpretability_benchmark.csv",
                 RESULTS / "interpretability_residue_level.csv",
                 RESULTS / "overlap_chance_reference.csv",
                 RESULTS / "overlap_chance_summary.json", checkpoint,
                 checkpoint.with_suffix(".json")]
    baseline = pd.read_csv(originals[0])
    released_residues = pd.read_csv(originals[1])
    for r in baseline.itertuples():
        tag = f"ibam_{r.pdb_id}_{r.ligand}".lower()
        originals.extend((RESULTS / "cache" / tag).glob("*"))
        originals.append(RESULTS / "downloads" / "interpretability_pdb" / f"{r.pdb_id}.pdb")
        originals.append(RESULTS / f"ibam_matrix_{r.pdb_id}.npz")
    before = {str(p.relative_to(ROOT)): sha256(p) for p in originals}
    rows, mapping_rows, details, arrays = [], [], {}, {}
    models = normalizer = None

    for r in baseline.itertuples():
        pdb = r.pdb_id
        tag = f"ibam_{pdb}_{r.ligand}".lower()
        cache_dir = RESULTS / "cache" / tag
        graph_cache = GraphCache(cache_dir / f"{tag}_graphs.pt",
                                 cache_dir / f"{tag}_graphs_manifest.json")
        protein_cache = ProteinEmbeddingCache(cache_dir / f"{tag}_proteins.pt",
                                               cache_dir / f"{tag}_proteins_manifest.json")
        smiles = next(iter(graph_cache.smiles_to_key))
        graph = Chem.MolFromSmiles(smiles)
        if graph is None:
            raise ValueError(f"Invalid cached SMILES for {pdb}")
        # Validate that graph order is exactly the one serialized for inference.
        fresh = smiles_to_graph(smiles, max_atoms=None)
        cached = graph_cache.get(smiles)
        cache_checks = {name: bool(torch.equal(old, new)) for name, old, new in zip(
            ("node_features", "adjacency", "atom_mask", "edge_features"),
            (cached["node_features"], cached["adjacency"], cached["atom_mask"],
             cached["edge_features"]), fresh)}
        if not all(cache_checks.values()):
            raise ValueError(f"Regenerated graph differs from exact saved graph: {pdb} {cache_checks}")

        pdb_text = (RESULTS / "downloads" / "interpretability_pdb" / f"{pdb}.pdb").read_text()
        atoms, ligands = parse_pdb_records(pdb_text, ligand_resname=r.ligand)
        ligand_key, ligand_atoms, chain = choose_best_ligand_occurrence(atoms, ligands)
        residues = extract_chain_residues(atoms, chain)
        if any(a["element"] in {"H", "D"} for a in ligand_atoms):
            raise ValueError("Unexpected ligand hydrogen in heavy-atom contact evaluation")
        if any(a["element"] in {"H", "D"} for res in residues for a in res["atoms"]):
            raise ValueError("Unexpected protein hydrogen in heavy-atom contact evaluation")
        sequence = "".join(res["aa"] for res in residues)
        if sequence not in protein_cache.sequence_to_key:
            raise ValueError("Coordinate-derived sequence differs from saved protein cache")
        contacts = atom_contact_mask(ligand_atoms, residues, cutoff=4.5)
        residue_contacts = residue_contact_mask(residues, ligand_atoms, cutoff=4.5)
        released = released_residues[released_residues.pdb_id == pdb].sort_values("residue_index")
        residue_match = np.array_equal(residue_contacts, released.contact.to_numpy(int))
        npz = np.load(RESULTS / f"ibam_matrix_{pdb}.npz")
        matrix_match = np.array_equal(residue_contacts, npz["contact"])
        if (len(residues), int(residue_contacts.sum()), len(ligand_atoms), int(contacts.sum())) != (
                r.n_residues, r.n_contact_residues, r.n_atoms, r.n_contact_atoms):
            raise ValueError(f"Coordinate counts disagree with released benchmark for {pdb}")
        if not residue_match or not matrix_match:
            raise ValueError(f"Residue contact labels differ for {pdb}")

        bound = ligand_mol_from_pdb(pdb_text, ligand_atoms)
        matches, vectors = mapped_contact_vectors(graph, bound, contacts)
        for i, atom in enumerate(ligand_atoms):
            mapping_rows.append({"pdb_id": pdb, "pdb_atom_index": i,
                "pdb_atom_name": atom["name"], "element": atom["element"],
                "contact": int(contacts[i]),
                "possible_graph_indices": ";".join(str(j) for j in sorted({m[i] for m in matches}))})

        new_metrics, old_order_rerun = [], None
        predicted_affinity = rerun_residue_auroc = None
        if np.unique(contacts).size == 2:
            if models is None:
                models, normalizer = load_ensemble([checkpoint], cfg)
            pairs = pd.DataFrame([{"compound_iso_smiles": smiles, "target_sequence": sequence}])
            loader = make_unlabeled_prediction_loader(pairs, graph_cache, protein_cache, cfg)
            payload = predict_unlabeled(models, loader, cfg, normalizer, collect_attention=True)
            a2r, r2a = _attention_arrays_from_batches(payload["attention"])
            a2r, r2a = a2r[:r.n_atoms, :r.n_residues], r2a[:r.n_residues, :r.n_atoms]
            scores = r2a.mean(axis=0)
            arrays[pdb + "_atom_scores"] = scores
            arrays[pdb + "_atom_to_residue"] = a2r
            arrays[pdb + "_residue_to_atom"] = r2a
            old_order_rerun = atom_metrics(contacts, scores)
            new_metrics = [atom_metrics(v, scores) for v in vectors]
            predicted_affinity = float(payload["predictions"][0])
            rerun_residue_auroc = float(roc_auc_score(residue_contacts, a2r.mean(axis=0)))
        else:
            new_metrics = [{"auroc": None, "topk_overlap": 1.0} for _ in vectors]
        arrays[pdb + "_pdb_contact_labels"] = contacts
        arrays[pdb + "_graph_contact_label_vectors"] = np.asarray(vectors)
        aurocs = [m["auroc"] for m in new_metrics if m["auroc"] is not None]
        overlaps = [m["topk_overlap"] for m in new_metrics]
        rows.append({"pdb_id": pdb, "protein": PROTEIN_NAMES[pdb], "ligand": r.ligand,
            "n_residues": len(residues), "n_contact_residues": int(residue_contacts.sum()),
            "n_atoms": len(contacts), "n_contact_atoms": int(contacts.sum()),
            "atom_base_rate": float(contacts.mean()), "n_valid_atom_mappings": len(matches),
            "n_distinct_graph_label_vectors": len(vectors),
            "archived_atom_auroc": r.atom_contact_auroc,
            "old_order_rerun_auroc": old_order_rerun["auroc"] if old_order_rerun else None,
            "corrected_atom_auroc_min": min(aurocs) if aurocs else None,
            "corrected_atom_auroc_max": max(aurocs) if aurocs else None,
            "archived_atom_topk_overlap": r.atom_topk_overlap,
            "corrected_atom_topk_overlap_min": min(overlaps),
            "corrected_atom_topk_overlap_max": max(overlaps),
            "residue_labels_identical": residue_match and matrix_match})
        details[pdb] = {"smiles": smiles, "protein_chain": chain, "ligand_key": ligand_key,
            "cache_tensor_equality": cache_checks,
            "all_pdb_to_graph_mappings": [list(m) for m in matches],
            "distinct_graph_contact_vectors": [v.tolist() for v in vectors],
            "old_order_rerun_metrics": old_order_rerun,
            "corrected_metrics_by_distinct_label_vector": new_metrics,
            "rerun_predicted_affinity": predicted_affinity,
            "archived_predicted_affinity": r.predicted_affinity,
            "rerun_residue_auroc": rerun_residue_auroc,
            "archived_residue_auroc": r.residue_contact_auroc}
        print(pdb, "mappings", len(matches), "label vectors", len(vectors),
              "AUROC", aurocs or "undefined", "overlap", overlaps, flush=True)

    unchanged = all(sha256(ROOT / p) == value for p, value in before.items())
    if not unchanged:
        raise RuntimeError("An input changed during revalidation; outputs not written")
    choices = [[m["topk_overlap"] for m in details[r.pdb_id]["corrected_metrics_by_distinct_label_vector"]]
               for r in baseline.itertuples()]
    base_rates = baseline.n_contact_atoms.to_numpy(float) / baseline.n_atoms.to_numpy(float)
    panel_results = []
    for choice in itertools.product(*choices):
        excess = np.asarray(choice) - base_rates
        panel_results.append({"atom_overlap_mean": float(np.mean(choice)),
            "atom_base_rate_mean": float(base_rates.mean()),
            "atom_excess_mean": float(excess.mean()), "atom_excess_ci95": t_interval(excess)})
    summary = {"method": "All element/bond-preserving PDB-to-cached-graph mappings; no attention-based mapping selection",
        "mapping_chirality": "Chirality constraints were not used, conservatively retaining all connectivity-preserving automorphisms; all resulting contact vectors are reported.",
        "interval_method": "Two-sided Student t interval of complex-level excesses, df=4, matching the archived overlap analysis",
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "rdkit_version": rdBase.rdkitVersion, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": before[str(checkpoint.relative_to(ROOT))],
        "inference_complexes": [p for p, v in details.items() if v["rerun_predicted_affinity"] is not None],
        "original_inputs_unchanged": unchanged, "input_sha256": before,
        "n_complexes": len(rows), "n_residues": int(baseline.n_residues.sum()),
        "n_contact_residues": int(baseline.n_contact_residues.sum()),
        "n_single_class_atom_cases": int((baseline.n_atoms == baseline.n_contact_atoms).sum()),
        "all_residue_labels_identical": all(r["residue_labels_identical"] for r in rows),
        "panel_results_over_all_distinct_label_assignments": panel_results,
        "complexes": details}
    out = RESULTS / "atom_contacts_revalidated"
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False, na_rep="NA")
    pd.DataFrame(mapping_rows).to_csv(RESULTS / "atom_contacts_revalidated_mapping.csv", index=False)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    text = ["# Atom-contact revalidation", "", summary["method"] + ".", "",
        "Original files and cached tensors were unchanged (SHA-256 verified).",
        "Only 1KE6 required inference; the other four atom-label vectors are single-class.",
        "All 1,385 residue labels, including 100 contacts, match the released labels.", "",
        "| PDB | Mappings | Label vectors | Archived atom AUROC | Corrected AUROC range | Corrected top-k overlap range |",
        "|---|---:|---:|---:|---|---|"]
    for row in rows:
        a = "undefined" if row["corrected_atom_auroc_min"] is None else (
            f"{row['corrected_atom_auroc_min']:.6f} to {row['corrected_atom_auroc_max']:.6f}")
        text.append(f"| {row['pdb_id']} | {row['n_valid_atom_mappings']} | {row['n_distinct_graph_label_vectors']} | "
                    f"{row['archived_atom_auroc']:.6f} | {a} | "
                    f"{row['corrected_atom_topk_overlap_min']:.6f} to {row['corrected_atom_topk_overlap_max']:.6f} |")
    text += ["", "Archived zero AUROCs in saturated cases are numerical fallbacks, not measurements.",
             "The JSON records every atom mapping, symmetry-related label vector, and conditional panel result.",
             "The NPZ retains the new forward-pass attention arrays, enabling metric recomputation without inference.",
             "", "## Panel overlap and chance reference", "", "```json", json.dumps(panel_results, indent=2), "```", ""]
    out.with_suffix(".md").write_text("\n".join(text))
    print("Panel:", panel_results, flush=True)
    print("Wrote atom_contacts_revalidated outputs; original inputs unchanged.", flush=True)


if __name__ == "__main__":
    main()
