"""Training engine for the modular DeepDTA-iBAM stack."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_profiles import ExperimentConfig
from data.cache_builders import GraphCache, ProteinEmbeddingCache, load_or_build_caches
from data.datasets import (
    DeepDTABatchCollator,
    KIBAPairDataset,
    TargetNormalizer,
    TargetGroupedTokenBudgetBatchSampler,
    TokenBudgetBatchSampler,
    maybe_subset_dataframe,
)
from models.rmse_model import DeepDTAGenIBAM
from training.checkpoints import CheckpointMetadata, cache_hash_from_manifests, load_metadata, save_model_checkpoint
from utils.metrics import concordance_index, mae, pearson_correlation, r_squared, rmse


class EMA:
    """Exponential moving average of trainable parameters."""

    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def compute_metrics(predictions: Sequence[float], targets: Sequence[float]) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return {
        "RMSE": float(rmse(targets, predictions)),
        "MAE": float(mae(targets, predictions)),
        "CI": float(concordance_index(targets, predictions)),
        "R2": float(r_squared(targets, predictions)),
        "Pearson": float(pearson_correlation(targets, predictions)),
    }


def ranking_weight_for_epoch(config: ExperimentConfig, epoch: int) -> float:
    if config.ranking_weight <= 0:
        return 0.0
    return config.ranking_weight if epoch > config.ranking_warmup_epochs else 0.0


def diffusion_weight_for_epoch(config: ExperimentConfig, epoch: int) -> float:
    if config.diffusion_max_weight <= 0:
        return 0.0
    if config.diffusion_ramp_end_epoch <= config.diffusion_warmup_epochs:
        return config.diffusion_max_weight if epoch > config.diffusion_warmup_epochs else 0.0
    if epoch <= config.diffusion_warmup_epochs:
        return 0.0
    if epoch >= config.diffusion_ramp_end_epoch:
        return config.diffusion_max_weight
    progress = (epoch - config.diffusion_warmup_epochs) / max(
        1,
        config.diffusion_ramp_end_epoch - config.diffusion_warmup_epochs,
    )
    return config.diffusion_max_weight * progress


def build_ranking_pair_indices(
    targets: torch.Tensor,
    group_keys: Sequence[str],
    min_delta: float,
    max_pairs: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = targets.view(-1)
    grouped_positions: Dict[str, List[int]] = {}
    for idx, key in enumerate(group_keys):
        grouped_positions.setdefault(key, []).append(idx)

    left_parts: List[torch.Tensor] = []
    right_parts: List[torch.Tensor] = []
    delta_parts: List[torch.Tensor] = []
    for positions in grouped_positions.values():
        if len(positions) < 2:
            continue
        index_tensor = torch.tensor(positions, device=targets.device, dtype=torch.long)
        group_targets = targets.index_select(0, index_tensor)
        left_local, right_local = torch.triu_indices(
            len(positions),
            len(positions),
            offset=1,
            device=targets.device,
        )
        deltas = group_targets[left_local] - group_targets[right_local]
        keep = deltas.abs() >= min_delta
        if not keep.any():
            continue
        left_parts.append(index_tensor.index_select(0, left_local[keep]))
        right_parts.append(index_tensor.index_select(0, right_local[keep]))
        delta_parts.append(deltas[keep])

    if not left_parts:
        empty = torch.empty(0, dtype=torch.long, device=targets.device)
        empty_signs = torch.empty(0, dtype=targets.dtype, device=targets.device)
        return empty, empty, empty_signs

    left = torch.cat(left_parts, dim=0)
    right = torch.cat(right_parts, dim=0)
    deltas = torch.cat(delta_parts, dim=0)
    if left.numel() > max_pairs:
        topk = torch.topk(deltas.abs(), k=max_pairs, largest=True).indices
        left = left.index_select(0, topk)
        right = right.index_select(0, topk)
        deltas = deltas.index_select(0, topk)
    return left, right, deltas.sign()


def pairwise_logistic_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_keys: Sequence[str],
    min_delta: float,
    max_pairs: int,
) -> Tuple[torch.Tensor, int]:
    left, right, signs = build_ranking_pair_indices(targets, group_keys, min_delta, max_pairs)
    if left.numel() == 0:
        return predictions.new_zeros(()), 0
    score_diff = predictions.view(-1).index_select(0, left) - predictions.view(-1).index_select(0, right)
    loss = F.softplus(-signs * score_diff).mean()
    return loss, int(left.numel())


def build_model(config: ExperimentConfig) -> DeepDTAGenIBAM:
    return DeepDTAGenIBAM(
        node_features=config.node_features,
        edge_features=config.edge_features,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_layers=config.gat_layers,
        gat_heads=config.gat_heads,
        protein_embedding_dim=config.protein_embedding_dim,
        protein_adapter_dim=config.protein_adapter_dim,
        protein_adapter_heads=config.protein_adapter_heads,
        protein_adapter_layers=config.protein_adapter_layers,
        protein_adapter_ff_mult=config.protein_adapter_ff_mult,
        fusion_mode=config.fusion_mode,
        fusion_dim=config.fusion_dim,
        fusion_layers=config.fusion_layers,
        fusion_heads=config.fusion_heads,
        fc_hidden_dim=config.fc_hidden_dim,
        dropout=config.dropout,
        diff_hidden_dim=config.diff_hidden_dim,
        diff_T=config.diff_T,
        diff_inference_steps=config.diff_inference_steps,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def runtime_device(config: ExperimentConfig) -> torch.device:
    if config.resolved_device.type == "cuda" and torch.cuda.device_count() > 1:
        return torch.device("cuda:0")
    return config.resolved_device


def available_cuda_gpu_count(config: ExperimentConfig) -> int:
    if config.resolved_device.type != "cuda":
        return 0
    return torch.cuda.device_count()


def maybe_parallelize_model(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    device = runtime_device(config)
    model = model.to(device)
    if device.type == "cuda":
        device_count = available_cuda_gpu_count(config)
        if device_count > 1:
            return nn.DataParallel(model, device_ids=list(range(device_count)))
    return model


def make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, config: ExperimentConfig) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = config.min_lr / config.learning_rate
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loss(config: ExperimentConfig) -> nn.Module:
    if config.loss_name.lower() == "huber":
        return nn.SmoothL1Loss(beta=config.huber_delta)
    return nn.MSELoss()


def build_dataloaders(
    config: ExperimentConfig,
    graph_cache: GraphCache,
    protein_cache: ProteinEmbeddingCache,
) -> Tuple[DataLoader, DataLoader, DataLoader, TargetNormalizer]:
    train_df = maybe_subset_dataframe(pd.read_csv(config.train_file), config.train_subset_fraction, config.seed)
    val_df = pd.read_csv(config.val_file).reset_index(drop=True)
    test_df = pd.read_csv(config.test_file).reset_index(drop=True)

    normalizer = TargetNormalizer.from_series(train_df["affinity"])
    collator = DeepDTABatchCollator(normalizer, normalize_targets=config.normalize_targets)

    train_dataset = KIBAPairDataset(train_df, graph_cache, protein_cache)
    val_dataset = KIBAPairDataset(val_df, graph_cache, protein_cache)
    test_dataset = KIBAPairDataset(test_df, graph_cache, protein_cache)

    if config.target_grouped_batches:
        train_sampler = TargetGroupedTokenBudgetBatchSampler(
            train_dataset.protein_lengths,
            train_dataset.target_sequences,
            max_pairs_per_batch=config.max_pairs_per_batch,
            protein_token_budget=config.protein_token_budget,
            shuffle=True,
            seed=config.seed,
        )
    else:
        train_sampler = TokenBudgetBatchSampler(
            train_dataset.protein_lengths,
            max_pairs_per_batch=config.max_pairs_per_batch,
            protein_token_budget=config.protein_token_budget,
            shuffle=True,
            seed=config.seed,
        )
    eval_sampler_kwargs = {
        "max_pairs_per_batch": config.max_pairs_per_batch,
        "protein_token_budget": config.protein_token_budget,
        "shuffle": False,
        "seed": config.seed,
    }
    val_sampler = TokenBudgetBatchSampler(val_dataset.protein_lengths, **eval_sampler_kwargs)
    test_sampler = TokenBudgetBatchSampler(test_dataset.protein_lengths, **eval_sampler_kwargs)

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.resolved_device.type == "cuda",
        "collate_fn": collator,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, **loader_kwargs)
    return train_loader, val_loader, test_loader, normalizer


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    tensor_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_batch[key] = value.to(device, non_blocking=True)
        else:
            tensor_batch[key] = value
    return tensor_batch


def _autocast_context(config: ExperimentConfig):
    if config.use_amp and runtime_device(config).type == "cuda":
        dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _denormalize_tensor(
    predictions: torch.Tensor,
    normalizer: TargetNormalizer,
    normalize_targets: bool,
) -> torch.Tensor:
    if normalize_targets:
        return predictions * normalizer.std + normalizer.mean
    return predictions


def _denormalize_predictions(
    predictions: torch.Tensor,
    normalizer: TargetNormalizer,
    normalize_targets: bool,
) -> torch.Tensor:
    return _denormalize_tensor(predictions, normalizer, normalize_targets)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _sync_runtime_device(config: ExperimentConfig) -> None:
    device = runtime_device(config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _log_progress(message: str) -> None:
    tqdm.write(message)


def _format_metric_summary(metrics: Dict[str, float], keys: Sequence[str]) -> str:
    return ", ".join(f"{key}={float(metrics[key]):.4f}" for key in keys if key in metrics)


def _init_attention_diagnostics() -> Dict[str, float]:
    return {
        "entropy_sum": 0.0,
        "max_weight_sum": 0.0,
        "collapsed_heads": 0.0,
        "head_count": 0.0,
    }


def _update_attention_diagnostics(
    aggregates: Dict[str, float],
    attention_maps: Optional[Dict[str, torch.Tensor]],
) -> None:
    if not attention_maps:
        return
    atom_query_mask = attention_maps.get("atom_query_mask")
    residue_query_mask = attention_maps.get("residue_query_mask")
    for name, weights in attention_maps.items():
        if not isinstance(weights, torch.Tensor) or weights.ndim != 4:
            continue
        if "atom_to_residue" in name:
            query_mask = atom_query_mask
        elif "residue_to_atom" in name:
            query_mask = residue_query_mask
        else:
            continue

        probs = weights.detach().float().clamp_min(1e-9)
        entropy = -(probs * probs.log()).sum(dim=-1)
        max_weight = probs.max(dim=-1).values
        normalizer = math.log(max(2, probs.size(-1)))
        entropy = entropy / normalizer
        if isinstance(query_mask, torch.Tensor):
            valid = query_mask.detach().to(device=probs.device, dtype=torch.bool).unsqueeze(1).expand_as(entropy)
        else:
            valid = torch.ones_like(entropy, dtype=torch.bool)
        valid_counts = valid.sum(dim=-1).clamp_min(1)
        per_head_entropy = (entropy * valid.float()).sum(dim=-1) / valid_counts
        per_head_max = (max_weight * valid.float()).sum(dim=-1) / valid_counts
        collapsed = (per_head_entropy >= 0.95) | (per_head_max >= 0.95)
        aggregates["entropy_sum"] += float(per_head_entropy.sum().item())
        aggregates["max_weight_sum"] += float(per_head_max.sum().item())
        aggregates["collapsed_heads"] += float(collapsed.sum().item())
        aggregates["head_count"] += float(collapsed.numel())


def _finalize_attention_diagnostics(aggregates: Dict[str, float]) -> Dict[str, float]:
    head_count = max(1.0, aggregates["head_count"])
    collapse_fraction = aggregates["collapsed_heads"] / head_count
    return {
        "AttentionEntropy": aggregates["entropy_sum"] / head_count,
        "AttentionMaxWeight": aggregates["max_weight_sum"] / head_count,
        "AttentionCollapseFrac": collapse_fraction,
        "AttentionCollapseWarn": float(collapse_fraction > 0.5),
    }


def _optimizer_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_optimizers.pt")


def _load_optimizer_sidecar(checkpoint_path: Path) -> Dict[str, Any]:
    sidecar_path = _optimizer_sidecar_path(checkpoint_path)
    if not sidecar_path.exists():
        return {}
    return torch.load(sidecar_path, map_location="cpu", weights_only=False)


def _find_latest_complete_ensemble(config: ExperimentConfig) -> Dict[str, Any]:
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    pattern = f"{config.profile_name}_member_*.safetensors"
    members_by_epoch: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for checkpoint_path in checkpoint_dir.glob(pattern):
        metadata_path = checkpoint_path.with_suffix(".json")
        if not metadata_path.exists():
            continue
        metadata = load_metadata(metadata_path)
        if metadata.get("profile_name") != config.profile_name:
            continue
        member_index = int(metadata["member_index"])
        epoch = int(metadata["epoch"])
        members_by_epoch.setdefault(epoch, {})[member_index] = {
            "checkpoint_path": checkpoint_path,
            "metadata": metadata,
        }

    required_members = set(range(config.ensemble_size))
    candidates: List[Tuple[int, float, Dict[int, Dict[str, Any]]]] = []
    for epoch, members in members_by_epoch.items():
        if set(members.keys()) != required_members:
            continue
        latest_mtime = max(
            member["checkpoint_path"].stat().st_mtime for member in members.values()
        )
        candidates.append((epoch, latest_mtime, members))

    if not candidates:
        found_members = sorted(
            {
                member_index
                for members in members_by_epoch.values()
                for member_index in members
            }
        )
        raise FileNotFoundError(
            "Unable to find a complete checkpoint ensemble for "
            f"profile '{config.profile_name}' with ensemble size {config.ensemble_size}. "
            f"Found member indices: {found_members or 'none'}."
        )

    epoch, _, members = max(candidates, key=lambda item: (item[0], item[1]))
    ordered_members = []
    for member_index in range(config.ensemble_size):
        member = members[member_index]
        checkpoint_path = member["checkpoint_path"]
        optimizer_bundle = _load_optimizer_sidecar(checkpoint_path)
        ordered_members.append(
            {
                "checkpoint_path": checkpoint_path,
                "metadata": member["metadata"],
                "optimizer_state": optimizer_bundle.get("optimizer_state_dict"),
                "scheduler_state": optimizer_bundle.get("scheduler_state_dict"),
            }
        )

    return {
        "epoch": epoch,
        "members": ordered_members,
    }


def _is_better_validation_snapshot(
    candidate: Dict[str, float],
    best: Optional[Dict[str, float]],
    config: ExperimentConfig,
) -> bool:
    if best is None:
        return True

    mode = config.selection_metric.lower()
    if mode == "rmse":
        return float(candidate["RMSE"]) < float(best["RMSE"]) - 1e-8
    if mode == "ci":
        return float(candidate["CI"]) > float(best["CI"]) + 1e-8
    if float(candidate["CI"]) > float(best["CI"]) + 1e-8:
        return True
    if abs(float(candidate["CI"]) - float(best["CI"])) <= 1e-8:
        return float(candidate["RMSE"]) < float(best["RMSE"]) - 1e-8
    return False


def _saved_checkpoint_paths(config: ExperimentConfig) -> List[Path]:
    return [
        Path(config.checkpoint_dir) / f"{config.profile_name}_member_{member_idx}.safetensors"
        for member_idx in range(config.ensemble_size)
    ]


def _load_saved_best_ensemble(
    config: ExperimentConfig,
    models: Optional[Sequence[nn.Module]] = None,
) -> List[nn.Module]:
    restored_models = list(models) if models is not None else []
    if models is not None and len(restored_models) != config.ensemble_size:
        raise ValueError("Expected one in-memory model per ensemble member when restoring best checkpoints")

    for member_idx, checkpoint_path in enumerate(_saved_checkpoint_paths(config)):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best-checkpoint member not found: {checkpoint_path}")
        model = restored_models[member_idx] if models is not None else maybe_parallelize_model(build_model(config), config)
        state_dict = {
            key: value.clone()
            for key, value in load_file(str(checkpoint_path), device=str(runtime_device(config))).items()
        }
        unwrap_model(model).load_state_dict(state_dict, strict=True)
        del state_dict
        model.eval()
        if models is None:
            restored_models.append(model)
    return restored_models


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: nn.Module,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler],
    ema: Optional[EMA],
    member_idx: Optional[int] = None,
    ensemble_size: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    diffusion_weight = diffusion_weight_for_epoch(config, epoch)
    active_ranking_weight = ranking_weight_for_epoch(config, epoch)
    regression_weight = 1.0 - active_ranking_weight if active_ranking_weight > 0 else 1.0
    total_loss = 0.0
    total_regression_loss = 0.0
    total_ranking_loss = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_examples = 0
    total_ranking_pairs = 0
    predictions: List[float] = []
    targets: List[float] = []

    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch - 1)
    progress_desc = f"train:{epoch}"
    if member_idx is not None:
        member_label = member_idx + 1
        if ensemble_size is not None and ensemble_size > 1:
            progress_desc = f"train:{epoch} member:{member_label}/{ensemble_size}"
        else:
            progress_desc = f"train:{epoch} member:{member_label}"
    expected_batches = len(loader)
    expected_examples = len(loader.dataset) if hasattr(loader, "dataset") else None
    wait_message = f"[epoch {epoch}] waiting for first batch from {progress_desc} ({expected_batches} batches"
    if expected_examples is not None:
        wait_message += f", {expected_examples} samples"
    wait_message += ")"
    _log_progress(wait_message)
    progress = tqdm(loader, desc=progress_desc, leave=False)
    epoch_start_time = time.perf_counter()
    first_batch_wait_seconds: Optional[float] = None
    for step_idx, batch in enumerate(progress, start=1):
        if step_idx == 1:
            first_batch_wait_seconds = time.perf_counter() - epoch_start_time
            sample_count = len(batch["compound_iso_smiles"]) if "compound_iso_smiles" in batch else "unknown"
            max_protein_length = (
                int(batch["protein_lengths"].max().item())
                if "protein_lengths" in batch and isinstance(batch["protein_lengths"], torch.Tensor)
                else "unknown"
            )
            max_atom_count = (
                int(batch["atom_counts"].max().item())
                if "atom_counts" in batch and isinstance(batch["atom_counts"], torch.Tensor)
                else "unknown"
            )
            _log_progress(
                f"[epoch {epoch}] first batch ready for {progress_desc} after {first_batch_wait_seconds:.2f}s "
                f"(batch_size={sample_count}, max_protein_len={max_protein_length}, max_atoms={max_atom_count})"
            )
            _log_progress(f"[epoch {epoch}] {progress_desc} first batch: moving tensors to {runtime_device(config)}")
        step_start_time = time.perf_counter()
        batch = _move_batch_to_device(batch, runtime_device(config))
        if step_idx == 1:
            _sync_runtime_device(config)
            transfer_seconds = time.perf_counter() - step_start_time
            _log_progress(
                f"[epoch {epoch}] {progress_desc} first batch transfer complete in {transfer_seconds:.2f}s; starting forward"
            )
        optimizer.zero_grad(set_to_none=True)

        forward_start_time = time.perf_counter()
        with _autocast_context(config):
            output, _, diff_loss = model(
                batch["drug_x"],
                batch["drug_adj"],
                batch["drug_mask"],
                batch["protein_embeddings"],
                batch["protein_mask"],
                drug_edge_features=batch["drug_edge_features"],
                compute_diff_loss=True,
            )
            affinity_loss = loss_fn(output.float(), batch["affinity_target"].float())
            denormalized_output = _denormalize_tensor(output.float(), normalizer, config.normalize_targets)
            ranking_loss, ranking_pairs = pairwise_logistic_ranking_loss(
                denormalized_output,
                batch["affinity_raw"].float(),
                batch["target_sequence"],
                min_delta=config.ranking_delta,
                max_pairs=config.ranking_pairs_per_batch,
            )
            total_batch_loss = regression_weight * affinity_loss + active_ranking_weight * ranking_loss
            total_batch_loss = total_batch_loss + diffusion_weight * diff_loss
        if step_idx == 1:
            _sync_runtime_device(config)
            forward_seconds = time.perf_counter() - forward_start_time
            _log_progress(
                f"[epoch {epoch}] {progress_desc} first batch forward complete in {forward_seconds:.2f}s; starting backward"
            )

        backward_start_time = time.perf_counter()
        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        if step_idx == 1:
            _sync_runtime_device(config)
            backward_seconds = time.perf_counter() - backward_start_time
            total_step_seconds = time.perf_counter() - step_start_time
            _log_progress(
                f"[epoch {epoch}] {progress_desc} first batch backward+step complete in {backward_seconds:.2f}s "
                f"(total_step_time={total_step_seconds:.2f}s)"
            )

        scheduler.step()
        if ema is not None:
            ema.update()

        preds = _denormalize_predictions(output.detach().float().cpu(), normalizer, config.normalize_targets)
        trgs = batch["affinity_raw"].detach().float().cpu()
        batch_squared_error = torch.square(preds.view(-1) - trgs.view(-1)).sum().item()
        batch_absolute_error = torch.abs(preds.view(-1) - trgs.view(-1)).sum().item()
        total_squared_error += float(batch_squared_error)
        total_absolute_error += float(batch_absolute_error)
        total_examples += int(trgs.numel())
        predictions.extend(preds.view(-1).tolist())
        targets.extend(trgs.view(-1).tolist())
        total_loss += float(total_batch_loss.item())
        total_regression_loss += float(affinity_loss.item())
        total_ranking_loss += float(ranking_loss.item())
        total_ranking_pairs += ranking_pairs
        running_avg_loss = total_loss / step_idx
        running_rmse = math.sqrt(total_squared_error / max(1, total_examples))
        running_mae = total_absolute_error / max(1, total_examples)
        progress.set_postfix(
            seen=total_examples,
            rmse=f"{running_rmse:.4f}",
            mae=f"{running_mae:.4f}",
            batch_loss=f"{total_batch_loss.item():.4f}",
            avg_loss=f"{running_avg_loss:.4f}",
            rank=f"{ranking_loss.item():.4f}",
            rank_w=f"{active_ranking_weight:.2f}",
            diff=f"{diffusion_weight:.4f}",
            lr=f"{_current_lr(optimizer):.2e}",
        )

    metrics = compute_metrics(predictions, targets)
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["regression_loss"] = total_regression_loss / max(1, len(loader))
    metrics["ranking_loss"] = total_ranking_loss / max(1, len(loader))
    metrics["ranking_pairs"] = float(total_ranking_pairs)
    metrics["ranking_weight"] = active_ranking_weight
    metrics["diffusion_weight"] = diffusion_weight
    metrics["first_batch_wait_seconds"] = (
        float(first_batch_wait_seconds) if first_batch_wait_seconds is not None else float("nan")
    )
    return metrics


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
) -> Tuple[List[float], List[float]]:
    model.eval()
    predictions: List[float] = []
    targets: List[float] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, runtime_device(config))
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
        preds = _denormalize_predictions(output.detach().float().cpu(), normalizer, config.normalize_targets)
        predictions.extend(preds.view(-1).tolist())
        targets.extend(batch["affinity_raw"].detach().float().cpu().view(-1).tolist())
    return predictions, targets


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, float]:
    predictions, targets = predict_loader(model, loader, config, normalizer)
    metrics = compute_metrics(predictions, targets)
    if loss_fn is not None:
        metrics["loss"] = float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))
    return metrics


@torch.no_grad()
def evaluate_ensemble(
    models: Sequence[nn.Module],
    loader: DataLoader,
    config: ExperimentConfig,
    normalizer: TargetNormalizer,
) -> Dict[str, Any]:
    for model in models:
        model.eval()

    all_predictions: List[float] = []
    all_targets: List[float] = []
    attention_diagnostics = _init_attention_diagnostics()
    for batch in loader:
        batch = _move_batch_to_device(batch, runtime_device(config))
        member_predictions = []
        for member_idx, model in enumerate(models):
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
            if member_idx == 0:
                _update_attention_diagnostics(attention_diagnostics, attention_maps)
            member_predictions.append(_denormalize_predictions(output.detach().float().cpu(), normalizer, config.normalize_targets))
        ensemble_output = torch.stack(member_predictions, dim=0).mean(dim=0)
        all_predictions.extend(ensemble_output.view(-1).tolist())
        all_targets.extend(batch["affinity_raw"].detach().float().cpu().view(-1).tolist())

    metrics = compute_metrics(all_predictions, all_targets)
    metrics.update(_finalize_attention_diagnostics(attention_diagnostics))
    metrics["predictions"] = all_predictions
    metrics["targets"] = all_targets
    return metrics


def train_ensemble(config: ExperimentConfig, resume: bool = False) -> Dict[str, Any]:
    graph_cache, protein_cache = load_or_build_caches(config)
    train_loader, val_loader, test_loader, normalizer = build_dataloaders(config, graph_cache, protein_cache)
    cache_hash = cache_hash_from_manifests([config.graph_manifest_path, config.protein_manifest_path])
    loss_fn = make_loss(config)

    resume_bundle = _find_latest_complete_ensemble(config) if resume else None
    start_epoch = 1
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, float]] = None
    if resume_bundle is not None:
        start_epoch = int(resume_bundle["epoch"]) + 1
        best_epoch = int(resume_bundle["epoch"])
        metadata_metrics = resume_bundle["members"][0]["metadata"].get("metrics", {})
        best_val_metrics = {
            "RMSE": float(metadata_metrics.get("RMSE", float("inf"))),
            "CI": float(metadata_metrics.get("CI", float("-inf"))),
        }

    total_steps = max(1, len(train_loader) * (config.num_epochs + start_epoch - 1))
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and runtime_device(config).type == "cuda" else None

    models: List[nn.Module] = []
    optimizers: List[torch.optim.Optimizer] = []
    schedulers: List[torch.optim.lr_scheduler.LambdaLR] = []
    emas: List[EMA] = []
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    _log_progress(
        "[train] "
        f"profile={config.profile_name}, device={runtime_device(config)}, cuda_gpus={available_cuda_gpu_count(config)}, "
        f"ensemble_size={config.ensemble_size}, train_samples={len(train_loader.dataset)}, "
        f"val_samples={len(val_loader.dataset)}, test_samples={len(test_loader.dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}, "
        f"epochs_this_run={config.num_epochs}, starting_epoch={start_epoch}, "
        f"selection_metric={config.selection_metric}"
    )
    if resume_bundle is not None:
        _log_progress(
            "[resume] "
            f"loaded epoch={resume_bundle['epoch']} with checkpoints="
            + ", ".join(member["checkpoint_path"].name for member in resume_bundle["members"])
        )

    for member_idx in range(config.ensemble_size):
        seed = config.ensemble_seed(member_idx)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = maybe_parallelize_model(build_model(config), config)
        base_model = unwrap_model(model)
        if resume_bundle is not None:
            checkpoint_path = resume_bundle["members"][member_idx]["checkpoint_path"]
            state_dict = {
                key: value.clone()
                for key, value in load_file(str(checkpoint_path), device=str(runtime_device(config))).items()
            }
            base_model.load_state_dict(state_dict, strict=True)
            del state_dict
        optimizer = make_optimizer(base_model, config)
        if resume_bundle is not None:
            optimizer_state = resume_bundle["members"][member_idx]["optimizer_state"]
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
        scheduler = make_scheduler(optimizer, total_steps, config)
        if resume_bundle is not None:
            scheduler_state = resume_bundle["members"][member_idx]["scheduler_state"]
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
        ema = EMA(base_model, config.ema_decay)
        if resume_bundle is not None:
            checkpoint_path = resume_bundle["members"][member_idx]["checkpoint_path"]
            has_optimizer_state = resume_bundle["members"][member_idx]["optimizer_state"] is not None
            has_scheduler_state = resume_bundle["members"][member_idx]["scheduler_state"] is not None
            _log_progress(
                "[resume] "
                f"member={member_idx + 1}/{config.ensemble_size}, checkpoint={checkpoint_path.name}, "
                f"optimizer_state={'yes' if has_optimizer_state else 'no'}, "
                f"scheduler_state={'yes' if has_scheduler_state else 'no'}"
            )
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        emas.append(ema)

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        _log_progress(f"[epoch {epoch}] starting")
        epoch_train_metrics = []
        for member_idx, model in enumerate(models):
            member_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizers[member_idx],
                scheduler=schedulers[member_idx],
                loss_fn=loss_fn,
                config=config,
                normalizer=normalizer,
                epoch=epoch,
                scaler=scaler,
                ema=emas[member_idx],
                member_idx=member_idx,
                ensemble_size=len(models),
            )
            epoch_train_metrics.append(member_metrics)
            _log_progress(
                f"[epoch {epoch}] member={member_idx + 1}/{len(models)} "
                + _format_metric_summary(
                    member_metrics,
                    ("RMSE", "MAE", "loss", "ranking_loss", "ranking_weight", "diffusion_weight"),
                )
                + f", lr={_current_lr(optimizers[member_idx]):.2e}"
            )

        for ema in emas:
            ema.apply_shadow()
        val_metrics = evaluate_ensemble(models, val_loader, config, normalizer)
        for ema in emas:
            ema.restore()

        mean_train_rmse = float(np.mean([metrics["RMSE"] for metrics in epoch_train_metrics]))
        history.append(
            {
                "epoch": epoch,
                "train_rmse_mean": mean_train_rmse,
                "val_rmse": float(val_metrics["RMSE"]),
                "val_mae": float(val_metrics["MAE"]),
                "val_ci": float(val_metrics["CI"]),
                "val_attention_entropy": float(val_metrics["AttentionEntropy"]),
                "val_attention_max_weight": float(val_metrics["AttentionMaxWeight"]),
                "val_attention_collapse_frac": float(val_metrics["AttentionCollapseFrac"]),
            }
        )

        if _is_better_validation_snapshot(val_metrics, best_val_metrics, config):
            best_val_metrics = {
                "RMSE": float(val_metrics["RMSE"]),
                "CI": float(val_metrics["CI"]),
            }
            best_epoch = epoch
            patience_counter = 0
            for member_idx, model in enumerate(models):
                emas[member_idx].apply_shadow()
                metadata = CheckpointMetadata(
                    profile_name=config.profile_name,
                    member_index=member_idx,
                    member_seed=config.ensemble_seed(member_idx),
                    epoch=epoch,
                    metrics={k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
                    config_hash=config.config_hash(),
                    cache_hash=cache_hash,
                    target_mean=normalizer.mean,
                    target_std=normalizer.std,
                )
                checkpoint_name = Path(config.checkpoint_dir) / f"{config.profile_name}_member_{member_idx}.safetensors"
                save_model_checkpoint(
                    unwrap_model(model).state_dict(),
                    metadata,
                    checkpoint_name,
                    optimizer_state=optimizers[member_idx].state_dict(),
                    scheduler_state=schedulers[member_idx].state_dict(),
                )
                emas[member_idx].restore()
            _log_progress(
                f"[epoch {epoch}] validation "
                + _format_metric_summary(
                    val_metrics,
                    ("RMSE", "MAE", "CI", "R2", "Pearson", "AttentionEntropy", "AttentionCollapseFrac"),
                )
                + f", train_rmse_mean={mean_train_rmse:.4f}, best_epoch={best_epoch}, "
                f"checkpoint_dir={config.checkpoint_dir}, status=new_best"
            )
        else:
            patience_counter += 1
            best_val_rmse = float(best_val_metrics["RMSE"]) if best_val_metrics is not None else float("inf")
            best_val_ci = float(best_val_metrics["CI"]) if best_val_metrics is not None else float("-inf")
            _log_progress(
                f"[epoch {epoch}] validation "
                + _format_metric_summary(
                    val_metrics,
                    ("RMSE", "MAE", "CI", "R2", "Pearson", "AttentionEntropy", "AttentionCollapseFrac"),
                )
                + f", train_rmse_mean={mean_train_rmse:.4f}, best_epoch={best_epoch}, "
                f"best_val_ci={best_val_ci:.4f}, best_val_rmse={best_val_rmse:.4f}, "
                f"patience={patience_counter}/{config.patience}"
            )

        if patience_counter >= config.patience:
            best_val_rmse = float(best_val_metrics["RMSE"]) if best_val_metrics is not None else float("inf")
            best_val_ci = float(best_val_metrics["CI"]) if best_val_metrics is not None else float("-inf")
            _log_progress(
                f"[early-stop] epoch={epoch}, best_epoch={best_epoch}, "
                f"best_val_ci={best_val_ci:.4f}, best_val_rmse={best_val_rmse:.4f}"
            )
            break

    best_models = _load_saved_best_ensemble(config, models=models)
    test_metrics = evaluate_ensemble(best_models, test_loader, config, normalizer)
    best_val_rmse = float(best_val_metrics["RMSE"]) if best_val_metrics is not None else float("inf")
    best_val_ci = float(best_val_metrics["CI"]) if best_val_metrics is not None else float("-inf")
    _log_progress(
        "[test] "
        + _format_metric_summary(
            test_metrics,
            ("RMSE", "MAE", "CI", "R2", "Pearson", "AttentionEntropy", "AttentionCollapseFrac"),
        )
        + f", best_epoch={best_epoch}, best_val_ci={best_val_ci:.4f}, best_val_rmse={best_val_rmse:.4f}"
    )

    return {
        "history": history,
        "best_val_rmse": best_val_rmse,
        "best_val_ci": best_val_ci,
        "best_epoch": best_epoch,
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in {"predictions", "targets"}},
        "target_mean": normalizer.mean,
        "target_std": normalizer.std,
        "cache_hash": cache_hash,
        "device": str(runtime_device(config)),
        "cuda_gpu_count": available_cuda_gpu_count(config),
        "selection_metric": config.selection_metric,
        "resumed_from": (
            {
                "epoch": int(resume_bundle["epoch"]),
                "checkpoint_paths": [
                    str(member["checkpoint_path"])
                    for member in resume_bundle["members"]
                ],
            }
            if resume_bundle is not None
            else None
        ),
    }


__all__ = [
    "EMA",
    "build_ranking_pair_indices",
    "build_dataloaders",
    "build_model",
    "compute_metrics",
    "diffusion_weight_for_epoch",
    "evaluate_ensemble",
    "evaluate_model",
    "available_cuda_gpu_count",
    "pairwise_logistic_ranking_loss",
    "predict_loader",
    "ranking_weight_for_epoch",
    "runtime_device",
    "train_ensemble",
    "train_one_epoch",
    "unwrap_model",
]
