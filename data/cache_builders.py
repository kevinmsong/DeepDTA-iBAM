"""Cache builders for graph features and full-length protein embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from config_profiles import ExperimentConfig


def _stable_id(prefix: str, raw_value: str) -> str:
    digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class GraphCache:
    """In-memory graph cache backed by a serialized tensor dictionary."""

    def __init__(self, cache_path: Path, manifest_path: Path):
        self.cache_path = cache_path
        self.manifest_path = manifest_path
        self.records: Dict[str, Dict[str, torch.Tensor]] = torch.load(cache_path, map_location="cpu", weights_only=True)
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.smiles_to_key = {entry["smiles"]: key for key, entry in self.manifest["items"].items()}

    def get(self, smiles: str) -> Dict[str, torch.Tensor]:
        return self.records[self.smiles_to_key[smiles]]


class ProteinEmbeddingCache:
    """In-memory protein embedding cache backed by a serialized tensor dictionary."""

    def __init__(self, cache_path: Path, manifest_path: Path):
        self.cache_path = cache_path
        self.manifest_path = manifest_path
        self.records: Dict[str, Dict[str, torch.Tensor]] = torch.load(cache_path, map_location="cpu", weights_only=True)
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.sequence_to_key = {entry["sequence"]: key for key, entry in self.manifest["items"].items()}

    def get(self, sequence: str) -> Dict[str, torch.Tensor]:
        return self.records[self.sequence_to_key[sequence]]


class GraphCacheBuilder:
    """Build a serialized graph cache for all unique SMILES in the dataset."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build(self, smiles_list: Iterable[str]) -> Dict[str, Any]:
        from models.rmse_model import smiles_to_graph

        unique_smiles = sorted(set(smiles_list))
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        manifest: Dict[str, Any] = {"node_features": self.config.node_features, "edge_features": self.config.edge_features, "items": {}}

        for smiles in unique_smiles:
            key = _stable_id("graph", smiles)
            node_features, adj, mask, edge_features = smiles_to_graph(smiles, max_atoms=None)
            cache[key] = {
                "node_features": node_features.cpu(),
                "adjacency": adj.cpu(),
                "atom_mask": mask.cpu(),
                "edge_features": edge_features.cpu(),
            }
            manifest["items"][key] = {
                "smiles": smiles,
                "num_atoms": int(mask.sum().item()),
            }

        self.config.graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, self.config.graph_cache_path)
        with self.config.graph_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest


class ProteinEmbeddingCacheBuilder:
    """Build a serialized protein embedding cache from raw sequences."""

    def __init__(self, config: ExperimentConfig, embedder: Optional[Any] = None):
        self.config = config
        self.embedder = embedder

    def _make_embedder(self) -> Any:
        if self.embedder is not None:
            return self.embedder
        from models.rmse_model import ProteinESM

        return ProteinESM(
            model_name=self.config.protein_embedding_model,
            embedding_dim=self.config.protein_embedding_dim,
            window_size=self.config.protein_window_size,
            overlap=self.config.protein_window_overlap,
            cache_dtype=self.config.protein_cache_dtype,
            device=self.config.device,
        )

    @staticmethod
    def _chunk_sequence(sequence: str, window_size: int, overlap: int) -> List[Tuple[int, int, str]]:
        if len(sequence) <= window_size:
            return [(0, len(sequence), sequence)]

        step = max(1, window_size - overlap)
        chunks: List[Tuple[int, int, str]] = []
        start = 0
        while start < len(sequence):
            end = min(start + window_size, len(sequence))
            chunk = sequence[start:end]
            chunks.append((start, end, chunk))
            if end >= len(sequence):
                break
            start += step
        return chunks

    @staticmethod
    def _stitch_chunks(
        chunk_embeddings: Sequence[Tuple[int, int, torch.Tensor]],
        sequence_length: int,
        embedding_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stitched = torch.zeros(sequence_length, embedding_dim, dtype=torch.float32)
        counts = torch.zeros(sequence_length, 1, dtype=torch.float32)
        for start, end, emb in chunk_embeddings:
            expected_len = end - start
            emb = emb[:expected_len].float()
            stitched[start : start + emb.size(0)] += emb
            counts[start : start + emb.size(0)] += 1.0
        counts = counts.clamp_min(1.0)
        stitched = stitched / counts
        mask = torch.ones(sequence_length, dtype=torch.bool)
        return stitched, mask

    def build(self, sequences: Iterable[str]) -> Dict[str, Any]:
        embedder = self._make_embedder()
        unique_sequences = sorted(set(sequences))
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        manifest: Dict[str, Any] = {
            "model_name": self.config.protein_embedding_model,
            "embedding_dim": self.config.protein_embedding_dim,
            "window_size": self.config.protein_window_size,
            "overlap": self.config.protein_window_overlap,
            "cache_dtype": self.config.protein_cache_dtype,
            "items": {},
        }

        for sequence in unique_sequences:
            key = _stable_id("protein", sequence)
            chunks = self._chunk_sequence(sequence, self.config.protein_window_size, self.config.protein_window_overlap)
            embedded_chunks: List[Tuple[int, int, torch.Tensor]] = []
            for batch_start in range(0, len(chunks), self.config.cache_batch_size):
                batch = chunks[batch_start : batch_start + self.config.cache_batch_size]
                batch_embeddings = embedder.embed_chunks([chunk for _, _, chunk in batch])
                for (start, end, _), chunk_embedding in zip(batch, batch_embeddings):
                    embedded_chunks.append((start, end, chunk_embedding.cpu()))

            stitched, mask = self._stitch_chunks(embedded_chunks, len(sequence), embedder.embedding_dim)
            desired_dtype = torch.bfloat16 if self.config.protein_cache_dtype == "bfloat16" else torch.float16
            cache[key] = {
                "embeddings": stitched.to(dtype=desired_dtype),
                "mask": mask.cpu(),
            }
            manifest["items"][key] = {
                "sequence": sequence,
                "length": len(sequence),
            }

        self.config.protein_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, self.config.protein_cache_path)
        with self.config.protein_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest


def _manifest_contains_values(manifest_path: Path, field_name: str, values: Sequence[str]) -> bool:
    if not manifest_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_values = {entry[field_name] for entry in manifest.get("items", {}).values() if field_name in entry}
    return set(values).issubset(manifest_values)


def load_or_build_caches(config: ExperimentConfig) -> Tuple[GraphCache, ProteinEmbeddingCache]:
    """Build caches on demand, then load them."""

    train_df = pd.read_csv(config.train_file)
    val_df = pd.read_csv(config.val_file)
    test_df = pd.read_csv(config.test_file)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    unique_smiles = all_df["compound_iso_smiles"].drop_duplicates().tolist()
    unique_sequences = all_df["target_sequence"].drop_duplicates().tolist()

    graph_cache_ready = (
        not config.build_caches_on_start
        and config.graph_cache_path.exists()
        and config.graph_manifest_path.exists()
        and _manifest_contains_values(config.graph_manifest_path, "smiles", unique_smiles)
    )
    protein_cache_ready = (
        not config.build_caches_on_start
        and config.protein_cache_path.exists()
        and config.protein_manifest_path.exists()
        and _manifest_contains_values(config.protein_manifest_path, "sequence", unique_sequences)
    )

    if not graph_cache_ready:
        GraphCacheBuilder(config).build(all_df["compound_iso_smiles"].tolist())
    if not protein_cache_ready:
        ProteinEmbeddingCacheBuilder(config).build(all_df["target_sequence"].tolist())

    return (
        GraphCache(config.graph_cache_path, config.graph_manifest_path),
        ProteinEmbeddingCache(config.protein_cache_path, config.protein_manifest_path),
    )


def build_isolated_caches(
    config: ExperimentConfig,
    smiles: Sequence[str],
    sequences: Sequence[str],
    cache_root: str,
    *,
    force_rebuild: bool = False,
    cache_prefix: str = "case_study",
) -> Tuple[ExperimentConfig, GraphCache, ProteinEmbeddingCache]:
    """Build or reuse caches for arbitrary case-study molecules and proteins.

    The returned config is a shallow copy with cache paths redirected under
    ``cache_root`` so publication workflows do not reuse or overwrite the
    training cache.
    """

    isolated = ExperimentConfig(**config.to_dict())
    isolated.cache_root = cache_root
    isolated.graph_cache_name = f"{cache_prefix}_graphs.pt"
    isolated.graph_manifest_name = f"{cache_prefix}_graphs_manifest.json"
    isolated.protein_cache_name = f"{cache_prefix}_proteins.pt"
    isolated.protein_manifest_name = f"{cache_prefix}_proteins_manifest.json"
    isolated.build_caches_on_start = force_rebuild

    graph_cache_ready = (
        not force_rebuild
        and isolated.graph_cache_path.exists()
        and isolated.graph_manifest_path.exists()
        and _manifest_contains_values(isolated.graph_manifest_path, "smiles", smiles)
    )
    protein_cache_ready = (
        not force_rebuild
        and isolated.protein_cache_path.exists()
        and isolated.protein_manifest_path.exists()
        and _manifest_contains_values(isolated.protein_manifest_path, "sequence", sequences)
    )

    if not graph_cache_ready:
        GraphCacheBuilder(isolated).build(smiles)
    if not protein_cache_ready:
        ProteinEmbeddingCacheBuilder(isolated).build(sequences)

    return (
        isolated,
        GraphCache(isolated.graph_cache_path, isolated.graph_manifest_path),
        ProteinEmbeddingCache(isolated.protein_cache_path, isolated.protein_manifest_path),
    )


__all__ = [
    "GraphCache",
    "GraphCacheBuilder",
    "ProteinEmbeddingCache",
    "ProteinEmbeddingCacheBuilder",
    "build_isolated_caches",
    "load_or_build_caches",
]
