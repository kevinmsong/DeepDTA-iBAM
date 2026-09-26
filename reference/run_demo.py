"""One-pair CPU reference using the production model, graph builder, and loader.

Default mode requires trained weights and an existing protein embedding cache.
The explicit random mode demonstrates tensor flow only and reports no affinity
estimate. Neither mode downloads assets, trains weights, or runs ESM-C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILE = "max_rmse_cluster_diffusion"
AUDITED_CHECKPOINT_SHA256 = "369be983160a49950c44ba9fcfe2334a8fd0a3c5ab084a27aff3f38aad203213"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("trained", "random"), default="trained",
                        help="trained requires assets; random is an explicitly untrained tensor demonstration")
    result.add_argument("--smiles", default=GEFITINIB, help="one ligand SMILES (default: gefitinib)")
    result.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / f"{PROFILE}_member_0.safetensors")
    result.add_argument("--protein-cache", type=Path, default=ROOT / "data/cache/proteins.pt")
    result.add_argument("--protein-manifest", type=Path, default=ROOT / "data/cache/proteins_manifest.json")
    result.add_argument("--protein-key", help="key in the protein manifest; otherwise use its shortest cached target")
    result.add_argument("--output-dir", type=Path,
                        help="optionally write summary.json and interaction_map.npz here")
    result.add_argument("--threads", type=int, default=2, help="CPU threads (default: 2)")
    result.add_argument("--seed", type=int, default=1337, help="random demonstration seed")
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def mean_interaction_map(attention, atom_mask, residue_mask):
    """Average production atom-to-residue weights over heads and blocks."""
    import torch

    keys = sorted(k for k in attention if k.startswith("layer_") and k.endswith("_atom_to_residue"))
    if not keys:
        raise ValueError("The model did not return atom-to-residue attention")
    layers = [attention[key][0].mean(dim=0) for key in keys]
    matrix = torch.stack(layers).mean(dim=0)
    return matrix[atom_mask[0].bool()][:, residue_mask[0].bool()], keys


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from rdkit import Chem

    from config_profiles import get_config_profile
    from data.cache_builders import ProteinEmbeddingCache
    from models.rmse_model import smiles_to_graph
    from training.engine import build_model
    from training.inference import load_ensemble

    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if Chem.MolFromSmiles(args.smiles) is None:
        raise ValueError("--smiles is not a valid molecular structure")
    if args.mode == "random" and args.protein_key:
        raise ValueError("--protein-key requires trained mode; random mode uses synthetic embeddings")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = get_config_profile(PROFILE)
    config.device = "cpu"
    config.use_amp = False
    config.ensemble_size = 1
    config.build_caches_on_start = False
    config.num_workers = 0

    summary = {"mode": args.mode, "profile": PROFILE, "device": "cpu",
               "torch_version": torch.__version__, "cpu_threads": args.threads,
               "smiles": args.smiles, "seed": args.seed}
    normalizer = None
    if args.mode == "trained":
        metadata_path = args.checkpoint.with_suffix(".json")
        paths = (args.checkpoint, metadata_path, args.protein_cache, args.protein_manifest)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Trained inference requires existing assets; missing: " + ", ".join(missing)
                                    + ". See reference/README.md. Use --mode random only for an untrained tensor demonstration.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("profile_name") != PROFILE:
            raise ValueError(f"This entrypoint supports only the {PROFILE} checkpoint architecture")
        mean, std = float(metadata["target_mean"]), float(metadata["target_std"])
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("Checkpoint target normalization metadata is invalid")
        cache = ProteinEmbeddingCache(args.protein_cache, args.protein_manifest)
        if cache.manifest.get("model_name") != config.protein_embedding_model:
            raise ValueError("The protein cache must contain ESM-C 600M embeddings")
        if cache.manifest.get("embedding_dim") != config.protein_embedding_dim:
            raise ValueError("Protein cache width does not match the checkpoint architecture")
        items = cache.manifest["items"]
        if not items:
            raise ValueError("Protein cache manifest is empty")
        key = args.protein_key or min(items, key=lambda k: (len(items[k]["sequence"]), k))
        if key not in items:
            raise ValueError(f"Protein key {key!r} is absent from the manifest")
        sequence = items[key]["sequence"]
        record = cache.get(sequence)
        protein = record["embeddings"].float().unsqueeze(0)
        protein_mask = record["mask"].bool().unsqueeze(0)
        if protein.shape != (1, len(sequence), config.protein_embedding_dim) or not protein_mask.any():
            raise ValueError("Cached protein tensors do not match their sequence")
        models, normalizer = load_ensemble([args.checkpoint], config)
        model = models[0]
        weight_hash = sha256(args.checkpoint)
        summary.update({
            "status": "checkpoint inference; not experimental binding evidence",
            "checkpoint_sha256": weight_hash,
            "matches_audited_manuscript_checkpoint": weight_hash == AUDITED_CHECKPOINT_SHA256,
            "metadata_epoch": metadata.get("epoch"),
            "protein_key": key,
            "target_selection": "explicit key" if args.protein_key else "shortest cached target; not necessarily EGFR",
            "protein_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "protein_cache_sha256": sha256(args.protein_cache),
            "protein_manifest_sha256": sha256(args.protein_manifest),
            "metadata_sha256": sha256(metadata_path),
            "checkpoint_history_note": "The audited checkpoint has epoch-47 metadata; its separate training summary ends at epoch 44. Selection history remains unresolved.",
        })
    else:
        model = build_model(config)
        protein = torch.randn(1, 16, config.protein_embedding_dim)
        protein_mask = torch.ones(1, 16, dtype=torch.bool)
        summary.update({"status": "UNTRAINED tensor demonstration; output is not an affinity estimate",
                        "target_selection": "16 synthetic embedding rows; no protein sequence or ESM-C inference",
                        "matches_audited_manuscript_checkpoint": False})

    x, adjacency, atoms, edges = smiles_to_graph(args.smiles, max_atoms=None)
    x, adjacency, atoms, edges = [value.unsqueeze(0) for value in (x, adjacency, atoms, edges)]
    if not atoms.any():
        raise ValueError("The ligand graph contains no atoms")
    model.eval()
    with torch.inference_mode():
        output, attention, _ = model(x, adjacency, atoms, protein, protein_mask,
                                    drug_edge_features=edges, compute_diff_loss=False)
        matrix, keys = mean_interaction_map(attention, atoms, protein_mask)
        profile = matrix.mean(dim=0)
    if not torch.isfinite(output).all() or not torch.isfinite(matrix).all():
        raise ValueError("Nonfinite model output")
    if normalizer is None:
        summary["untrained_raw_output_not_affinity"] = float(output.item())
    else:
        summary["predicted_affinity_kiba_units"] = float((output.float() * normalizer.std + normalizer.mean).item())
    summary.update({"parameter_count": sum(p.numel() for p in model.parameters()),
                    "interaction_map_shape": list(matrix.shape),
                    "residue_profile_shape": list(profile.shape),
                    "attention_blocks_averaged": len(keys),
                    "attention_heads_per_block": int(attention[keys[0]].shape[1]),
                    "maximum_attention_row_sum_error": float((matrix.sum(dim=1) - 1).abs().max()),
                    "interpretation": "Attention weights are model couplings, not validated physical contacts or causal explanations.",
                    "workflow_references": {
                        "ranking": "analysis/score_kinase_panel.py",
                        "generation": "run_generation_validation.py",
                        "saved_case_studies": "case_studies_results_generation.py",
                    }})
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir / "interaction_map.npz",
                            atom_to_residue=matrix.numpy(), residue_profile=profile.numpy())
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        cli.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
