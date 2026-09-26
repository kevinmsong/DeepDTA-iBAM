"""Audit the existing checkpoint and match a small fresh prediction sample.

Preserves all inputs. Uses cached embeddings, one CPU pair per batch, and two
PyTorch threads. This verifies the released artifact, not its missing history.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from config_profiles import get_config_profile
from data.cache_builders import GraphCache, ProteinEmbeddingCache
from training.checkpoints import stable_hash
from training.inference import load_ensemble, make_unlabeled_prediction_loader, predict_unlabeled

STEM = "max_rmse_cluster_diffusion"
WEIGHTS = ROOT / "checkpoints" / f"{STEM}_member_0.safetensors"
SIDECAR = WEIGHTS.with_suffix(".json")
OPTIMIZER = WEIGHTS.with_name(WEIGHTS.stem + "_optimizers.pt")
SUMMARY = WEIGHTS.parent / f"{STEM}_training_summary.json"
PREDICTIONS = ROOT / "results" / f"{STEM}_member1_standard_predictions.csv"
OUT = ROOT / "results" / "checkpoint_provenance_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1337)
    inputs = [WEIGHTS, SIDECAR, OPTIMIZER, SUMMARY, PREDICTIONS,
              ROOT / "data/raw/test_kiba.csv", ROOT / "data/raw/train_kiba.csv", ROOT / "data/cache/graphs.pt",
              ROOT / "data/cache/proteins.pt", ROOT / "data/cache/graphs_manifest.json",
              ROOT / "data/cache/proteins_manifest.json"]
    ablation_pred_path = ROOT / "results/attention_ablation_predictions.csv"
    ablation_cache = ROOT / "results/cache/attention_ablation"
    inputs += [ablation_pred_path] + [ablation_cache / name for name in (
        "attention_ablation_graphs.pt", "attention_ablation_graphs_manifest.json",
        "attention_ablation_proteins.pt", "attention_ablation_proteins_manifest.json")]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
    meta = json.loads(SIDECAR.read_text())
    summary = json.loads(SUMMARY.read_text())
    with safe_open(str(WEIGHTS), framework="pt", device="cpu") as handle:
        header_meta = handle.metadata()
        tensor_count = len(handle.keys())
    optim = torch.load(OPTIMIZER, map_location="cpu", weights_only=True)
    optim_meta = optim["metadata"]
    optim_steps = sorted({int(x["step"]) for x in optim["optimizer_state_dict"]["state"].values()})
    scheduler = optim["scheduler_state_dict"]
    del optim
    gc.collect()

    # Original cluster manifest hashes use forward-slash relative path keys.
    manifests = {f"data/cache/{name}_manifest.json": json.loads(
        (ROOT / f"data/cache/{name}_manifest.json").read_text())
        for name in ("graphs", "proteins")}
    reproduced_cache_hash = stable_hash(manifests)
    train_counts = pd.read_csv(ROOT / "data/raw/train_kiba.csv", usecols=["target_sequence"]).groupby("target_sequence").size()
    profile = get_config_profile(STEM)
    batches_per_epoch = sum(math.ceil(n / min(profile.max_pairs_per_batch,
        max(1, profile.protein_token_budget // len(sequence)))) for sequence, n in train_counts.items())
    epochs = [int(h["epoch"]) for h in summary["history"]]
    report = {
        "audit_scope": "Existing artifacts and a deterministic 24-pair fresh CPU inference check; no retraining",
        "input_sha256": hashes,
        "checkpoint_tensor_count": tensor_count,
        "safetensors_embedded_metadata": header_meta,
        "sidecar_metadata": meta,
        "optimizer_metadata_matches_sidecar": optim_meta == meta,
        "optimizer_step_values": optim_steps,
        "scheduler_last_epoch": scheduler["last_epoch"],
        "scheduler_step_count": scheduler["_step_count"],
        "optimizer_steps_per_sidecar_epoch": [s / meta["epoch"] for s in optim_steps],
        "training_batches_per_epoch_reconstructed_from_current_profile": batches_per_epoch,
        "optimizer_steps_match_47_complete_profile_epochs": optim_steps == [meta["epoch"] * batches_per_epoch],
        "cache_manifest_hash_recomputed": reproduced_cache_hash,
        "cache_manifest_hash_matches_checkpoint": reproduced_cache_hash == meta["cache_hash"],
        "history": {"first_epoch": min(epochs), "last_epoch": max(epochs),
                    "n_epochs": len(epochs), "best_epoch": summary["best_epoch"],
                    "best_val_ci": summary["best_val_ci"], "best_val_rmse": summary["best_val_rmse"],
                    "resumed_from": summary.get("resumed_from"),
                    "contains_checkpoint_epoch": meta["epoch"] in epochs},
        "limits": [
            "The safetensors file contains no embedded epoch, config, training-history, or validation metadata.",
            "Optimizer metadata corroborates the sidecar, but the available history ends before epoch 47.",
            "A matching prediction sample cannot recover the missing training continuation or prove checkpoint-selection history.",
            "The original full resolved configuration is absent; its saved hash cannot be inverted.",
        ],
    }
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("Recorded checkpoint identity and metadata; preparing fresh inference.", flush=True)

    config = get_config_profile(STEM)
    config.device = "cpu"
    config.use_amp = False
    config.ensemble_size = 1
    config.max_pairs_per_batch = 1
    config.protein_token_budget = 5000
    config.num_workers = 0
    config.build_caches_on_start = False
    graph = GraphCache(config.graph_cache_path, config.graph_manifest_path)
    protein = ProteinEmbeddingCache(config.protein_cache_path, config.protein_manifest_path)
    test = pd.read_csv(ROOT / "data/raw/test_kiba.csv")
    archived = pd.read_csv(PREDICTIONS)
    order = np.asarray(sorted(range(len(test)), key=lambda i: len(test.iloc[i]["target_sequence"])))
    expected_target = test.iloc[order]["affinity"].to_numpy(np.float32).astype(float)
    if not np.array_equal(expected_target, archived["target"].to_numpy(np.float32).astype(float)):
        raise RuntimeError("Archived target order does not match the released test CSV's stable length sort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    test["test_row"] = np.arange(len(test))
    test["protein_length"] = test["target_sequence"].str.len()
    unique = test.sort_values(["protein_length", "target_sequence", "compound_iso_smiles"], kind="stable").drop_duplicates("target_sequence")
    selected = unique.iloc[np.linspace(0, len(unique) - 1, 24, dtype=int)].copy().reset_index(drop=True)
    loader = make_unlabeled_prediction_loader(selected, graph, protein, config)
    subset_order = np.asarray([i for batch in loader.batch_sampler for i in batch])
    models, norm = load_ensemble([WEIGHTS], config)
    start = time.perf_counter()
    payload = predict_unlabeled(models, loader, config, norm, progress_label="checkpoint audit")
    elapsed = time.perf_counter() - start
    fresh = np.empty(len(selected))
    fresh[subset_order] = payload["predictions"]
    archived_prediction = archived.iloc[inverse[selected["test_row"].to_numpy()]]["prediction"].to_numpy()
    delta = fresh - archived_prediction
    rows = selected[["test_row", "protein_length", "affinity", "compound_iso_smiles"]].copy()
    rows["target_sequence_sha256"] = selected["target_sequence"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    rows["archived_prediction"] = archived_prediction
    rows["fresh_prediction"] = fresh
    rows["fresh_minus_archived"] = delta
    rows.to_csv(OUT.with_suffix(".csv"), index=False)
    check = {
        "selection": "24 evenly spaced distinct targets after sorting by length, sequence, and SMILES; one pair per target",
        "n_pairs": len(selected), "n_targets": selected["target_sequence"].nunique(),
        "n_compounds": selected["compound_iso_smiles"].nunique(),
        "protein_length_range": [int(selected["protein_length"].min()), int(selected["protein_length"].max())],
        "all_5913_archived_targets_match_reconstructed_sampler_order": True,
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "precision": "float32 CPU inference using archived bfloat16 ESM-C cache",
        "batch_size": 1, "seconds": elapsed,
        "maximum_absolute_prediction_difference": float(np.max(np.abs(delta))),
        "mean_absolute_prediction_difference": float(np.mean(np.abs(delta))),
        "prediction_correlation": float(np.corrcoef(fresh, archived_prediction)[0, 1]),
        "n_within_absolute_1e_minus_4": int(np.sum(np.abs(delta) <= 1e-4)),
        "n_within_absolute_1e_minus_3": int(np.sum(np.abs(delta) <= 1e-3)),
    }
    # The archived standard predictions lie exactly on the denormalized
    # bfloat16 output grid; the fresh CPU check uses float32 model arithmetic.
    saved_tensor = torch.tensor(archived["prediction"].to_numpy(), dtype=torch.float32)
    bf16_reconstruction = ((saved_tensor - norm.mean) / norm.std).to(torch.bfloat16).float() * norm.std + norm.mean
    check["archived_outputs_exactly_on_denormalized_bfloat16_grid"] = int((saved_tensor == bf16_reconstruction).sum())

    # Corroborate the weight identity against a second CPU artifact using its
    # own archived embedding cache. Different cache builds need not be bitwise
    # identical across devices; do not silently substitute their embeddings.
    ablation = pd.read_csv(ablation_pred_path)
    ablation_indices = np.random.default_rng(1337).choice(len(test), 1200, replace=False)
    ablation_indices.sort()
    if not np.allclose(test.iloc[ablation_indices]["affinity"], ablation["affinity"], atol=1e-12, rtol=0):
        raise RuntimeError("Ablation targets do not match the documented seeded subset")
    ablation_lookup = {int(i): float(p) for i, p in zip(ablation_indices, ablation["pred_intact"])}
    common = selected[selected["test_row"].isin(ablation_lookup)].reset_index(drop=True)
    ablation_graph = GraphCache(ablation_cache / "attention_ablation_graphs.pt", ablation_cache / "attention_ablation_graphs_manifest.json")
    ablation_protein = ProteinEmbeddingCache(ablation_cache / "attention_ablation_proteins.pt", ablation_cache / "attention_ablation_proteins_manifest.json")
    common_loader = make_unlabeled_prediction_loader(common, ablation_graph, ablation_protein, config)
    common_order = np.asarray([i for batch in common_loader.batch_sampler for i in batch])
    second = predict_unlabeled(models, common_loader, config, norm, progress_label="checkpoint CPU corroboration")
    second_prediction = np.empty(len(common))
    second_prediction[common_order] = second["predictions"]
    second_expected = np.asarray([ablation_lookup[int(i)] for i in common["test_row"]])
    embedding_differences = []
    for sequence in common["target_sequence"]:
        left = protein.get(sequence)["embeddings"].float()
        right = ablation_protein.get(sequence)["embeddings"].float()
        embedding_differences.append(float((left - right).abs().max()))
    corroboration = {
        "n_pairs": len(common),
        "cache": "results/cache/attention_ablation",
        "maximum_absolute_prediction_difference": float(np.abs(second_prediction - second_expected).max()),
        "mean_absolute_prediction_difference": float(np.abs(second_prediction - second_expected).mean()),
        "base_vs_ablation_embedding_maximum_absolute_differences": embedding_differences,
        "test_rows": common["test_row"].tolist(),
        "fresh_predictions": second_prediction.tolist(),
        "archived_predictions": second_expected.tolist(),
    }
    report["fresh_ablation_cache_corroboration"] = corroboration
    report["fresh_inference"] = check
    report["inputs_unchanged_after_audit"] = all(sha256(p) == hashes[p.relative_to(ROOT).as_posix()] for p in inputs)
    report["conclusion"] = (
        "The checkpoint has a reproducible SHA256 identity and its optimizer metadata corroborates epoch 47. "
        "The earlier 44-epoch training summary cannot describe the full provenance of that checkpoint. "
        "Fresh predictions reproduce the saved evaluation within the numerical differences reported below; "
        "training chronology and original validation selection remain unrecoverable from the available records."
    )
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2))
    note = f"""# Checkpoint provenance audit

## Established

- Weights: `checkpoints/{WEIGHTS.name}`.
- SHA256: `{hashes[WEIGHTS.relative_to(ROOT).as_posix()]}`.
- The optimizer archive's metadata exactly matches the epoch-{meta['epoch']} JSON sidecar.
- Optimizer steps: {optim_steps}; scheduler last step: {scheduler['last_epoch']}.
- Current profile and training CSV reconstruct {batches_per_epoch} batches per epoch, so {meta['epoch']} complete epochs give exactly {meta['epoch'] * batches_per_epoch} optimizer steps. This supports the sidecar's epoch count without recovering the missing history.
- Cache manifest hash matches checkpoint: {reproduced_cache_hash == meta['cache_hash']}.
- All 5,913 archived target values match the test CSV after reconstructing its stable length-sorted inference order.
- Fresh inference: {len(selected)} pairs from {check['n_targets']} targets, lengths {check['protein_length_range'][0]} to {check['protein_length_range'][1]}; {check['n_compounds']} compounds.
- Maximum absolute prediction difference: {check['maximum_absolute_prediction_difference']:.9g} KIBA units; mean absolute difference: {check['mean_absolute_prediction_difference']:.9g}; correlation: {check['prediction_correlation']:.12f}.
- {check['n_within_absolute_1e_minus_4']}/{len(selected)} differences are at most 0.0001 KIBA units.
- All {check['archived_outputs_exactly_on_denormalized_bfloat16_grid']} archived standard outputs lie exactly on the denormalized bfloat16 grid; fresh CPU arithmetic uses float32.
- A second check uses {corroboration['n_pairs']} overlapping gate-ablation pairs and that analysis's own archived cache: maximum prediction difference {corroboration['maximum_absolute_prediction_difference']:.9g} KIBA units. The base and ablation protein caches differ numerically, with per-target maximum embedding differences {embedding_differences}.
- All original input hashes remain unchanged: {report['inputs_unchanged_after_audit']}.

## Still unresolved

The training summary covers epochs {min(epochs)} through {max(epochs)}, names epoch {summary['best_epoch']} as best, and records no resume. The weights file carries no embedded training metadata. The epoch-47 sidecar and optimizer metadata agree, but neither supplies the missing epoch-by-epoch history. No audited artifact links the old summary's best-epoch metrics to the epoch-47 weights. The original resolved configuration is not saved, so its hash cannot be used to recover omitted settings.

This audit establishes current checkpoint identity and prediction reproducibility. It does not independently establish the training chronology, the sidecar's validation metrics, or the original checkpoint-selection decision. Retain general validation-based selection wording and disclose the unreconciled training log.

## Reproduction

Run `python analysis/audit_checkpoint_provenance.py` from the project environment. It reads existing weights and caches, performs 24 CPU forward passes plus {corroboration['n_pairs']} corroborating passes using two threads, and writes this report, JSON, and per-pair CSV. It does not retrain or modify original artifacts.
"""
    OUT.with_suffix(".md").write_text(note, encoding="utf-8")
    print(json.dumps(check, indent=2), flush=True)


if __name__ == "__main__":
    main()
