"""Checkpoint metadata helpers for the modular DeepDTA-iBAM stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from safetensors.torch import save_file


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class CheckpointMetadata:
    """Standardized metadata stored next to every model checkpoint."""

    profile_name: str
    member_index: int
    member_seed: int
    epoch: int
    metrics: Dict[str, float]
    config_hash: str
    cache_hash: str
    target_mean: float
    target_std: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_model_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
    metadata: CheckpointMetadata,
    checkpoint_path: Path,
    optimizer_state: Optional[Mapping[str, Any]] = None,
    scheduler_state: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save weights and standardized metadata."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.contiguous() for key, value in state_dict.items()}
    save_file(state, str(checkpoint_path))

    metadata_path = checkpoint_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)

    if optimizer_state is not None or scheduler_state is not None:
        torch.save(
            {
                "optimizer_state_dict": optimizer_state,
                "scheduler_state_dict": scheduler_state,
                "metadata": metadata.to_dict(),
            },
            checkpoint_path.with_name(f"{checkpoint_path.stem}_optimizers.pt"),
        )


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_hash_from_manifests(paths: Iterable[Path]) -> str:
    payload: Dict[str, Any] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload[str(path)] = json.load(handle)
    return stable_hash(payload)


__all__ = [
    "CheckpointMetadata",
    "cache_hash_from_manifests",
    "load_metadata",
    "save_model_checkpoint",
    "stable_hash",
]
