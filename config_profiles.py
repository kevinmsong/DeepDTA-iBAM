"""Reusable experiment profiles for the max-RMSE DeepDTA-iBAM stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class ExperimentConfig:
    """Single-source configuration for training, caching, and inference."""

    profile_name: str

    # Data
    train_file: str = "data/raw/train_kiba.csv"
    val_file: str = "data/raw/val_kiba.csv"
    test_file: str = "data/raw/test_kiba.csv"
    train_subset_fraction: float = 1.0

    # Cache/output paths
    cache_root: str = "data/cache"
    graph_cache_name: str = "graphs.pt"
    graph_manifest_name: str = "graphs_manifest.json"
    protein_cache_name: str = "proteins.pt"
    protein_manifest_name: str = "proteins_manifest.json"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Drug encoder
    node_features: int = 78
    edge_features: int = 12
    gat_hidden_dim: int = 512
    gat_layers: int = 6
    gat_heads: int = 8

    # Protein cache / adapter
    protein_embedding_model: str = "esmc_600m"
    protein_embedding_dim: int = 1152
    protein_window_size: int = 1022
    protein_window_overlap: int = 128
    protein_cache_dtype: str = "bfloat16"
    protein_adapter_dim: int = 512
    protein_adapter_heads: int = 8
    protein_adapter_layers: int = 2
    protein_adapter_ff_mult: int = 4

    # Fusion / head
    fusion_mode: str = "bidirectional"
    fusion_dim: int = 512
    fusion_layers: int = 3
    fusion_heads: int = 8
    fc_hidden_dim: int = 1024
    dropout: float = 0.1

    # Diffusion
    diff_hidden_dim: int = 256
    diff_T: int = 1000
    diff_inference_steps: int = 50
    diffusion_max_weight: float = 0.02
    diffusion_warmup_epochs: int = 2
    diffusion_ramp_end_epoch: int = 6

    # Targets / loss
    normalize_targets: bool = True
    loss_name: str = "mse"
    huber_delta: float = 3.0
    selection_metric: str = "ci_then_rmse"
    ranking_weight: float = 0.4
    ranking_delta: float = 0.3
    ranking_pairs_per_batch: int = 2048
    ranking_warmup_epochs: int = 3
    target_grouped_batches: bool = True

    # Training
    num_epochs: int = 40
    patience: int = 8
    max_pairs_per_batch: int = 96
    protein_token_budget: int = 48_000
    num_workers: int = 4
    learning_rate: float = 2e-4
    min_lr: float = 1e-6
    warmup_ratio: float = 0.10
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    ensemble_size: int = 5
    seed: int = 1337
    ensemble_seed_stride: int = 1000
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    device: str = field(default_factory=_default_device)

    # Execution modes
    cache_batch_size: int = 1
    build_caches_on_start: bool = True
    save_best_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def graph_cache_path(self) -> Path:
        return Path(self.cache_root) / self.graph_cache_name

    @property
    def graph_manifest_path(self) -> Path:
        return Path(self.cache_root) / self.graph_manifest_name

    @property
    def protein_cache_path(self) -> Path:
        return Path(self.cache_root) / self.protein_cache_name

    @property
    def protein_manifest_path(self) -> Path:
        return Path(self.cache_root) / self.protein_manifest_name

    @property
    def resolved_device(self) -> torch.device:
        return torch.device(self.device)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ensemble_seed(self, member_idx: int) -> int:
        return self.seed + member_idx * self.ensemble_seed_stride


def _baseline_repro() -> ExperimentConfig:
    return ExperimentConfig(
        profile_name="baseline_repro",
        protein_embedding_model="esmc_300m",
        protein_embedding_dim=960,
        protein_window_size=1024,
        protein_window_overlap=0,
        protein_adapter_dim=960,
        protein_adapter_heads=8,
        protein_adapter_layers=1,
        protein_adapter_ff_mult=2,
        fusion_dim=512,
        fusion_layers=2,
        fusion_heads=8,
        gat_hidden_dim=512,
        gat_layers=6,
        gat_heads=8,
        fc_hidden_dim=1024,
        diff_hidden_dim=256,
        diff_T=1000,
        diff_inference_steps=50,
        diffusion_max_weight=0.05,
        diffusion_warmup_epochs=0,
        diffusion_ramp_end_epoch=0,
        normalize_targets=False,
        loss_name="huber",
        huber_delta=3.0,
        selection_metric="rmse",
        ranking_weight=0.0,
        target_grouped_batches=False,
        num_epochs=10,
        patience=5,
        max_pairs_per_batch=80,
        protein_token_budget=80_000,
        learning_rate=1e-4,
        min_lr=1e-6,
        warmup_ratio=0.05,
        weight_decay=0.0,
        ensemble_size=3,
        dropout=0.0,
    )


def _max_rmse_cluster() -> ExperimentConfig:
    return ExperimentConfig(
        profile_name="max_rmse_cluster",
        loss_name="huber",
        huber_delta=1.0,
        protein_adapter_layers=4,
        diffusion_max_weight=0.0,
        diffusion_warmup_epochs=0,
        diffusion_ramp_end_epoch=0,
        learning_rate=1e-4,
        num_epochs=40,
        patience=8,
        ensemble_size=5,
        build_caches_on_start=False,
    )


def _diffusion_egfr_seed() -> ExperimentConfig:
    config = _max_rmse_cluster()
    config.profile_name = "diffusion_egfr_seed"
    config.diffusion_max_weight = 0.02
    config.diffusion_warmup_epochs = 2
    config.diffusion_ramp_end_epoch = 6
    return config


def _max_rmse_cluster_diffusion() -> ExperimentConfig:
    config = _max_rmse_cluster()
    config.profile_name = "max_rmse_cluster_diffusion"
    config.diffusion_max_weight = 0.02
    config.diffusion_warmup_epochs = 2
    config.diffusion_ramp_end_epoch = 6
    return config


def _max_rmse_cluster_no_fusion() -> ExperimentConfig:
    config = _max_rmse_cluster()
    config.profile_name = "max_rmse_cluster_no_fusion"
    config.fusion_mode = "none"
    return config


# ---------------------------------------------------------------------------
# Ablation profiles
# Each runs as a single-member ensemble (ensemble_size=1); the ablation
# runner sweeps 3 seeds to obtain mean ± SD.  All share the same base
# hyperparameters as _max_rmse_cluster (Huber δ=1.0, 40 epochs, AdamW).
# ---------------------------------------------------------------------------

def _abl_base() -> ExperimentConfig:
    """Base regressor: no cross-attention, no ranking loss, no diffusion."""
    config = _max_rmse_cluster()
    config.profile_name = "abl_base"
    config.fusion_mode = "none"
    config.ranking_weight = 0.0
    config.diffusion_max_weight = 0.0
    config.diffusion_warmup_epochs = 0
    config.diffusion_ramp_end_epoch = 0
    config.ensemble_size = 1
    return config


def _abl_no_fusion() -> ExperimentConfig:
    """No cross-attention, but ranking and diffusion auxiliary losses active."""
    config = _max_rmse_cluster()
    config.profile_name = "abl_no_fusion"
    config.fusion_mode = "none"
    config.ranking_weight = 0.4
    config.ranking_warmup_epochs = 3
    config.diffusion_max_weight = 0.02
    config.diffusion_warmup_epochs = 2
    config.diffusion_ramp_end_epoch = 6
    config.ensemble_size = 1
    return config


def _abl_no_ranking() -> ExperimentConfig:
    """Full bidirectional cross-attention + diffusion, ranking loss removed."""
    config = _max_rmse_cluster()
    config.profile_name = "abl_no_ranking"
    config.fusion_mode = "bidirectional"
    config.ranking_weight = 0.0
    config.diffusion_max_weight = 0.02
    config.diffusion_warmup_epochs = 2
    config.diffusion_ramp_end_epoch = 6
    config.ensemble_size = 1
    return config


def _abl_no_diffusion() -> ExperimentConfig:
    """Full bidirectional cross-attention + ranking, diffusion auxiliary removed."""
    config = _max_rmse_cluster()
    config.profile_name = "abl_no_diffusion"
    config.fusion_mode = "bidirectional"
    config.ranking_weight = 0.4
    config.ranking_warmup_epochs = 3
    config.diffusion_max_weight = 0.0
    config.diffusion_warmup_epochs = 0
    config.diffusion_ramp_end_epoch = 0
    config.ensemble_size = 1
    return config


def _abl_full() -> ExperimentConfig:
    """Full model: bidirectional cross-attention + ranking loss + diffusion."""
    config = _max_rmse_cluster()
    config.profile_name = "abl_full"
    config.fusion_mode = "bidirectional"
    config.ranking_weight = 0.4
    config.ranking_warmup_epochs = 3
    config.diffusion_max_weight = 0.02
    config.diffusion_warmup_epochs = 2
    config.diffusion_ramp_end_epoch = 6
    config.ensemble_size = 1
    return config


def _inference() -> ExperimentConfig:
    return ExperimentConfig(
        profile_name="inference",
        num_epochs=1,
        patience=1,
        build_caches_on_start=False,
        ensemble_size=1,
        use_amp=torch.cuda.is_available(),
    )


_PROFILES = {
    "baseline_repro": _baseline_repro,
    "max_rmse_cluster": _max_rmse_cluster,
    "diffusion_egfr_seed": _diffusion_egfr_seed,
    "max_rmse_cluster_diffusion": _max_rmse_cluster_diffusion,
    "max_rmse_cluster_no_fusion": _max_rmse_cluster_no_fusion,
    "inference": _inference,
    # Ablation profiles (single-member; sweep seeds externally)
    "abl_base": _abl_base,
    "abl_no_fusion": _abl_no_fusion,
    "abl_no_ranking": _abl_no_ranking,
    "abl_no_diffusion": _abl_no_diffusion,
    "abl_full": _abl_full,
}


def get_config_profile(name: str) -> ExperimentConfig:
    """Return a named experiment profile."""

    if name not in _PROFILES:
        raise KeyError(f"Unknown config profile: {name}")
    return _PROFILES[name]()


__all__ = ["ExperimentConfig", "get_config_profile"]
