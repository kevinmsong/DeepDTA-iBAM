"""Dataset, collation, and batching utilities for cached DeepDTA-iBAM training."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from data.cache_builders import GraphCache, ProteinEmbeddingCache


@dataclass
class TargetNormalizer:
    """Z-score normalizer for affinity values."""

    mean: float
    std: float

    @classmethod
    def from_series(cls, series: pd.Series) -> "TargetNormalizer":
        std = float(series.std())
        if std <= 0:
            std = 1.0
        return cls(mean=float(series.mean()), std=std)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


class KIBAPairDataset(Dataset):
    """Dataset that resolves raw CSV rows against precomputed graph/protein caches."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        graph_cache: GraphCache,
        protein_cache: ProteinEmbeddingCache,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.graph_cache = graph_cache
        self.protein_cache = protein_cache

        self.target_sequences = self.dataframe["target_sequence"].tolist()
        self.protein_lengths = [int(protein_cache.get(seq)["mask"].sum().item()) for seq in self.dataframe["target_sequence"]]
        self.atom_counts = [int(graph_cache.get(smiles)["atom_mask"].sum().item()) for smiles in self.dataframe["compound_iso_smiles"]]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        graph = self.graph_cache.get(row["compound_iso_smiles"])
        protein = self.protein_cache.get(row["target_sequence"])
        return {
            "compound_iso_smiles": row["compound_iso_smiles"],
            "target_sequence": row["target_sequence"],
            "graph": graph,
            "protein": protein,
            "affinity": float(row["affinity"]),
            "protein_length": self.protein_lengths[index],
            "atom_count": self.atom_counts[index],
        }


class UnlabeledPairDataset(Dataset):
    """Dataset for arbitrary ligand-protein pairs without affinity labels."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        graph_cache: GraphCache,
        protein_cache: ProteinEmbeddingCache,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.graph_cache = graph_cache
        self.protein_cache = protein_cache

        self.target_sequences = self.dataframe["target_sequence"].tolist()
        self.protein_lengths = [int(protein_cache.get(seq)["mask"].sum().item()) for seq in self.target_sequences]
        self.atom_counts = [
            int(graph_cache.get(smiles)["atom_mask"].sum().item())
            for smiles in self.dataframe["compound_iso_smiles"]
        ]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        graph = self.graph_cache.get(row["compound_iso_smiles"])
        protein = self.protein_cache.get(row["target_sequence"])
        return {
            "compound_iso_smiles": row["compound_iso_smiles"],
            "target_sequence": row["target_sequence"],
            "graph": graph,
            "protein": protein,
            "protein_length": self.protein_lengths[index],
            "atom_count": self.atom_counts[index],
        }


class TokenBudgetBatchSampler(Sampler[List[int]]):
    """Length-aware batch sampler bounded by pair count and protein token budget."""

    def __init__(
        self,
        lengths: Sequence[int],
        max_pairs_per_batch: int,
        protein_token_budget: int,
        shuffle: bool = True,
        seed: int = 1337,
        bucket_size: int = 256,
    ):
        self.lengths = list(lengths)
        self.max_pairs_per_batch = max_pairs_per_batch
        self.protein_token_budget = protein_token_budget
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self._epoch = 0

    def __len__(self) -> int:
        batch_count = 0
        batch_size = 0
        max_length = 0
        for idx in self._ordered_indices():
            seq_len = self.lengths[idx]
            proposed_size = batch_size + 1
            proposed_max_length = max(max_length, seq_len)
            proposed_tokens = proposed_size * proposed_max_length
            if batch_size and (
                proposed_size > self.max_pairs_per_batch or proposed_tokens > self.protein_token_budget
            ):
                batch_count += 1
                batch_size = 0
                max_length = 0

            batch_size += 1
            max_length = max(max_length, seq_len)

        if batch_size:
            batch_count += 1
        return batch_count

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _ordered_indices(self) -> List[int]:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: self.lengths[idx])

        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)
            return [idx for bucket in buckets for idx in bucket]
        return indices

    def __iter__(self) -> Iterator[List[int]]:
        ordered = self._ordered_indices()
        batch: List[int] = []
        max_length = 0
        for idx in ordered:
            seq_len = self.lengths[idx]
            proposed_size = len(batch) + 1
            proposed_max_length = max(max_length, seq_len)
            proposed_tokens = proposed_size * proposed_max_length
            if batch and (
                proposed_size > self.max_pairs_per_batch or proposed_tokens > self.protein_token_budget
            ):
                yield batch
                batch = []
                max_length = 0
            batch.append(idx)
            max_length = max(max_length, seq_len)

        if batch:
            yield batch


class TargetGroupedTokenBudgetBatchSampler(Sampler[List[int]]):
    """Batch one target at a time while respecting pair-count and token budgets."""

    def __init__(
        self,
        lengths: Sequence[int],
        group_keys: Sequence[str],
        max_pairs_per_batch: int,
        protein_token_budget: int,
        shuffle: bool = True,
        seed: int = 1337,
    ):
        if len(lengths) != len(group_keys):
            raise ValueError("lengths and group_keys must be the same length")
        self.lengths = list(lengths)
        self.group_keys = list(group_keys)
        self.max_pairs_per_batch = max_pairs_per_batch
        self.protein_token_budget = protein_token_budget
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _batched_group_indices(self) -> List[List[int]]:
        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx, key in enumerate(self.group_keys):
            grouped[key].append(idx)

        rng = random.Random(self.seed + self._epoch)
        groups = list(grouped.items())
        if self.shuffle:
            rng.shuffle(groups)

        batches: List[List[int]] = []
        for _, indices in groups:
            if self.shuffle:
                rng.shuffle(indices)
            protein_length = max(self.lengths[idx] for idx in indices)
            max_batch_by_budget = max(1, self.protein_token_budget // max(1, protein_length))
            batch_size = max(1, min(self.max_pairs_per_batch, max_batch_by_budget))
            for start in range(0, len(indices), batch_size):
                batches.append(indices[start : start + batch_size])

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __len__(self) -> int:
        return len(self._batched_group_indices())

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batched_group_indices()


class DeepDTABatchCollator:
    """Collate cached graph/protein records into padded batched tensors."""

    def __init__(self, normalizer: TargetNormalizer, normalize_targets: bool = True):
        self.normalizer = normalizer
        self.normalize_targets = normalize_targets

    @staticmethod
    def _pad_graphs(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_atoms = max(int(item["graph"]["atom_mask"].sum().item()) for item in items)
        node_dim = items[0]["graph"]["node_features"].size(-1)
        edge_dim = items[0]["graph"]["edge_features"].size(-1)

        batch_size = len(items)
        node_features = torch.zeros(batch_size, max_atoms, node_dim, dtype=torch.float32)
        adjacency = torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.float32)
        atom_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
        edge_features = torch.zeros(batch_size, max_atoms, max_atoms, edge_dim, dtype=torch.float32)

        for batch_idx, item in enumerate(items):
            graph = item["graph"]
            num_atoms = int(graph["atom_mask"].sum().item())
            node_features[batch_idx, :num_atoms] = graph["node_features"][:num_atoms].float()
            adjacency[batch_idx, :num_atoms, :num_atoms] = graph["adjacency"][:num_atoms, :num_atoms].float()
            atom_mask[batch_idx, :num_atoms] = graph["atom_mask"][:num_atoms]
            edge_features[batch_idx, :num_atoms, :num_atoms] = graph["edge_features"][:num_atoms, :num_atoms].float()

        return {
            "drug_x": node_features,
            "drug_adj": adjacency,
            "drug_mask": atom_mask,
            "drug_edge_features": edge_features,
        }

    @staticmethod
    def _pad_proteins(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_residues = max(int(item["protein"]["mask"].sum().item()) for item in items)
        embed_dim = items[0]["protein"]["embeddings"].size(-1)

        batch_size = len(items)
        embeddings = torch.zeros(batch_size, max_residues, embed_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_residues, dtype=torch.bool)

        for batch_idx, item in enumerate(items):
            protein = item["protein"]
            num_residues = int(protein["mask"].sum().item())
            embeddings[batch_idx, :num_residues] = protein["embeddings"][:num_residues].float()
            mask[batch_idx, :num_residues] = protein["mask"][:num_residues]

        return {"protein_embeddings": embeddings, "protein_mask": mask}

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        batch.update(self._pad_graphs(items))
        batch.update(self._pad_proteins(items))

        raw_affinity = torch.tensor([item["affinity"] for item in items], dtype=torch.float32).view(-1, 1)
        if self.normalize_targets:
            target = self.normalizer.normalize(raw_affinity)
        else:
            target = raw_affinity.clone()

        batch["affinity_raw"] = raw_affinity
        batch["affinity_target"] = target
        batch["compound_iso_smiles"] = [item["compound_iso_smiles"] for item in items]
        batch["target_sequence"] = [item["target_sequence"] for item in items]
        batch["protein_lengths"] = torch.tensor([item["protein_length"] for item in items], dtype=torch.long)
        batch["atom_counts"] = torch.tensor([item["atom_count"] for item in items], dtype=torch.long)
        return batch


def maybe_subset_dataframe(dataframe: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return dataframe.reset_index(drop=True)
    return dataframe.sample(frac=fraction, random_state=seed).reset_index(drop=True)


__all__ = [
    "DeepDTABatchCollator",
    "KIBAPairDataset",
    "TargetNormalizer",
    "TargetGroupedTokenBudgetBatchSampler",
    "TokenBudgetBatchSampler",
    "UnlabeledPairDataset",
    "maybe_subset_dataframe",
]
