"""Canonical inference helpers for cached DeepDTA-iBAM predictors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from config_profiles import ExperimentConfig
from data.cache_builders import GraphCache, ProteinEmbeddingCache
from data.datasets import (
    DeepDTABatchCollator,
    KIBAPairDataset,
    TargetNormalizer,
    TokenBudgetBatchSampler,
    UnlabeledPairDataset,
)
from models.rmse_model import DeepDTAGenIBAM
from training.engine import (
    _autocast_context,
    _denormalize_predictions,
    _move_batch_to_device,
    build_model,
    evaluate_ensemble,
    runtime_device,
)


def load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ensemble(
    checkpoint_paths: Sequence[Path],
    config: ExperimentConfig,
) -> Tuple[List[DeepDTAGenIBAM], TargetNormalizer]:
    models: List[DeepDTAGenIBAM] = []
    metadata = load_checkpoint_metadata(checkpoint_paths[0].with_suffix(".json"))
    normalizer = TargetNormalizer(mean=float(metadata["target_mean"]), std=float(metadata["target_std"]))

    for checkpoint_path in checkpoint_paths:
        model = build_model(config).to(config.resolved_device)
        state_dict = load_file(str(checkpoint_path), device=str(config.resolved_device))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)

    return models, normalizer


def make_prediction_loader(
    dataframe,
    graph_cache: GraphCache,
    protein_cache: ProteinEmbeddingCache,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
) -> DataLoader:
    dataset = KIBAPairDataset(dataframe.reset_index(drop=True), graph_cache, protein_cache)
    sampler = TokenBudgetBatchSampler(
        dataset.protein_lengths,
        max_pairs_per_batch=config.max_pairs_per_batch,
        protein_token_budget=config.protein_token_budget,
        shuffle=False,
        seed=config.seed,
    )
    collator = DeepDTABatchCollator(normalizer, normalize_targets=config.normalize_targets)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.resolved_device.type == "cuda",
        collate_fn=collator,
    )


class UnlabeledBatchCollator:
    """Collate arbitrary case-study pairs without targets."""

    def __call__(self, items):
        batch = {}
        batch.update(DeepDTABatchCollator._pad_graphs(items))
        batch.update(DeepDTABatchCollator._pad_proteins(items))
        batch["compound_iso_smiles"] = [item["compound_iso_smiles"] for item in items]
        batch["target_sequence"] = [item["target_sequence"] for item in items]
        batch["protein_lengths"] = torch.tensor([item["protein_length"] for item in items], dtype=torch.long)
        batch["atom_counts"] = torch.tensor([item["atom_count"] for item in items], dtype=torch.long)
        return batch


def make_unlabeled_prediction_loader(
    dataframe,
    graph_cache: GraphCache,
    protein_cache: ProteinEmbeddingCache,
    config: ExperimentConfig,
) -> DataLoader:
    dataset = UnlabeledPairDataset(dataframe.reset_index(drop=True), graph_cache, protein_cache)
    sampler = TokenBudgetBatchSampler(
        dataset.protein_lengths,
        max_pairs_per_batch=config.max_pairs_per_batch,
        protein_token_budget=config.protein_token_budget,
        shuffle=False,
        seed=config.seed,
    )
    collator = UnlabeledBatchCollator()
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.resolved_device.type == "cuda",
        collate_fn=collator,
    )


def predict_affinity_batch(
    models: Sequence[DeepDTAGenIBAM],
    loader: DataLoader,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
) -> Dict[str, Any]:
    return evaluate_ensemble(models, loader, config, normalizer)


@torch.no_grad()
def predict_unlabeled(
    models: Sequence[DeepDTAGenIBAM],
    loader: DataLoader,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
    *,
    collect_attention: bool = False,
    progress_label: Optional[str] = None,
) -> Dict[str, Any]:
    predictions = []
    attention_batches = []
    for model in models:
        model.eval()

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    progress_interval = 1
    if isinstance(total_batches, int) and total_batches > 10:
        progress_interval = max(1, total_batches // 10)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, runtime_device(config))
        member_predictions = []
        member_attention = []
        for model in models:
            with _autocast_context(config):
                output, attention_maps, _ = model(
                    batch["drug_x"],
                    batch["drug_adj"],
                    batch["drug_mask"],
                    batch["protein_embeddings"],
                    batch["protein_mask"],
                    drug_edge_features=batch["drug_edge_features"],
                    compute_diff_loss=False,
                )
            member_predictions.append(
                _denormalize_predictions(output.detach().float().cpu(), normalizer, config.normalize_targets)
            )
            if collect_attention:
                cpu_attention = {
                    key: value.detach().float().cpu()
                    for key, value in attention_maps.items()
                    if isinstance(value, torch.Tensor)
                }
                member_attention.append(cpu_attention)
        predictions.append(torch.stack(member_predictions, dim=0).mean(dim=0))
        if collect_attention:
            attention_batches.append(member_attention)
        if progress_label is not None and (
            batch_idx == 1
            or total_batches is None
            or batch_idx == total_batches
            or batch_idx % progress_interval == 0
        ):
            if isinstance(total_batches, int):
                print(
                    f"[progress] {progress_label}: batch {batch_idx}/{total_batches}",
                    flush=True,
                )
            else:
                print(f"[progress] {progress_label}: batch {batch_idx}", flush=True)

    prediction_tensor = torch.cat(predictions, dim=0).view(-1)
    result = {"predictions": prediction_tensor.tolist()}
    if collect_attention:
        result["attention"] = attention_batches
    return result


__all__ = [
    "load_checkpoint_metadata",
    "load_ensemble",
    "make_prediction_loader",
    "make_unlabeled_prediction_loader",
    "predict_affinity_batch",
    "predict_unlabeled",
]
