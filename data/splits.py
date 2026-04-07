"""Dataset splitting helpers for hardened DeepDTA-iBAM training protocols."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


REQUIRED_COLUMNS = ("compound_iso_smiles", "target_sequence", "affinity")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitArtifacts:
    """Materialized train/val/test split paths plus a reproducibility summary."""

    mode: str
    train_file: str
    val_file: str
    test_file: str
    manifest_path: str | None
    summary: Dict[str, Any]


def _ensure_required_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required dataset columns: {missing}")


def _normalize_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, float]:
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    return {name: value / total for name, value in ratios.items()}


def _load_split_dataframe(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path).reset_index(drop=True)
    _ensure_required_columns(dataframe)
    return dataframe


def _load_full_dataframe(train_file: str, val_file: str, test_file: str) -> pd.DataFrame:
    return pd.concat(
        [
            _load_split_dataframe(train_file),
            _load_split_dataframe(val_file),
            _load_split_dataframe(test_file),
        ],
        ignore_index=True,
    )


def _scaffold_key_from_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return f"INVALID::{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    if scaffold:
        return scaffold
    # Acyclic compounds would otherwise all collapse into the empty scaffold.
    return f"ACYCLIC::{smiles}"


def _group_sizes(dataframe: pd.DataFrame, group_column: str) -> Dict[str, int]:
    grouped = dataframe.groupby(group_column, sort=False).size()
    return {str(key): int(value) for key, value in grouped.items()}


def _fill_ratio(current: int, target: float) -> float:
    if target <= 0:
        return float("inf")
    return current / target


def _assign_group_splits(
    group_sizes: Mapping[str, int],
    ratios: Mapping[str, float],
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, float]]:
    total_rows = sum(group_sizes.values())
    target_counts = {split: total_rows * ratios[split] for split in SPLIT_NAMES}

    rng = random.Random(seed)
    groups = list(group_sizes.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: item[1], reverse=True)

    assignments: Dict[str, str] = {}
    current_counts = {split: 0 for split in SPLIT_NAMES}
    for group_key, group_count in groups:
        best_split = min(
            SPLIT_NAMES,
            key=lambda split: (
                _fill_ratio(current_counts[split], target_counts[split]),
                current_counts[split],
                SPLIT_NAMES.index(split),
            ),
        )
        assignments[group_key] = best_split
        current_counts[best_split] += group_count

    return assignments, current_counts, target_counts


def _pairwise_overlap_summary(values_by_split: Mapping[str, set[str]]) -> Dict[str, int]:
    train_values = values_by_split["train"]
    val_values = values_by_split["val"]
    test_values = values_by_split["test"]
    return {
        "train_val": len(train_values & val_values),
        "train_test": len(train_values & test_values),
        "val_test": len(val_values & test_values),
    }


def _build_summary(
    split_frames: Mapping[str, pd.DataFrame],
    mode: str,
    ratios: Mapping[str, float],
    seed: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "mode": mode,
        "seed": int(seed),
        "ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "rows": {name: int(len(frame)) for name, frame in split_frames.items()},
        "unique_compounds": {
            name: int(frame["compound_iso_smiles"].nunique()) for name, frame in split_frames.items()
        },
        "unique_targets": {
            name: int(frame["target_sequence"].nunique()) for name, frame in split_frames.items()
        },
        "compound_overlap": _pairwise_overlap_summary(
            {name: set(frame["compound_iso_smiles"]) for name, frame in split_frames.items()}
        ),
        "target_overlap": _pairwise_overlap_summary(
            {name: set(frame["target_sequence"]) for name, frame in split_frames.items()}
        ),
    }

    if mode == "scaffold":
        scaffold_sets = {
            name: { _scaffold_key_from_smiles(smiles) for smiles in frame["compound_iso_smiles"].unique() }
            for name, frame in split_frames.items()
        }
        summary["scaffold_overlap"] = _pairwise_overlap_summary(scaffold_sets)

    return summary


def _materialize_scaffold_split(
    dataframe: pd.DataFrame,
    ratios: Mapping[str, float],
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    working = dataframe.copy()
    group_column = "_scaffold_key"
    working[group_column] = working["compound_iso_smiles"].map(_scaffold_key_from_smiles)

    group_sizes = _group_sizes(working, group_column)
    assignments, realized_counts, target_counts = _assign_group_splits(group_sizes, ratios, seed)
    working["_split"] = working[group_column].map(assignments)

    split_frames = {
        name: working.loc[working["_split"] == name].drop(columns=["_split"], errors="ignore").reset_index(drop=True)
        for name in SPLIT_NAMES
    }
    for name in SPLIT_NAMES:
        split_frames[name] = split_frames[name].drop(columns=[group_column], errors="ignore")

    summary = _build_summary(split_frames, mode="scaffold", ratios=ratios, seed=seed)
    summary["group_column"] = "murcko_scaffold"
    summary["group_counts"] = {key: int(value) for key, value in realized_counts.items()}
    summary["group_targets"] = {key: float(value) for key, value in target_counts.items()}
    summary["num_groups"] = int(len(group_sizes))
    return split_frames, summary


def prepare_split_artifacts(
    train_file: str,
    val_file: str,
    test_file: str,
    mode: str = "scaffold",
    output_root: str = "data/generated_splits",
    seed: int = 1337,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> SplitArtifacts:
    """Return split files for the requested strategy, materializing new CSVs when needed."""

    if mode == "standard":
        summary = _build_summary(
            {
                "train": _load_split_dataframe(train_file),
                "val": _load_split_dataframe(val_file),
                "test": _load_split_dataframe(test_file),
            },
            mode="standard",
            ratios=_normalize_split_ratios(train_ratio, val_ratio, test_ratio),
            seed=seed,
        )
        return SplitArtifacts(
            mode=mode,
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            manifest_path=None,
            summary=summary,
        )

    if mode != "scaffold":
        raise ValueError(f"Unsupported split mode: {mode}. Only 'scaffold' and 'standard' are supported.")

    ratios = _normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    full_dataframe = _load_full_dataframe(train_file, val_file, test_file)
    split_frames, summary = _materialize_scaffold_split(full_dataframe, ratios=ratios, seed=seed)

    ratio_tag = "_".join(f"{name}{ratios[name]:.3f}".replace(".", "p") for name in SPLIT_NAMES)
    split_dir = Path(output_root) / f"{mode}_seed{seed}_{ratio_tag}"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    manifest_path = split_dir / "manifest.json"

    split_frames["train"].to_csv(train_path, index=False)
    split_frames["val"].to_csv(val_path, index=False)
    split_frames["test"].to_csv(test_path, index=False)

    manifest = {
        "mode": mode,
        "seed": int(seed),
        "source_files": {
            "train": train_file,
            "val": val_file,
            "test": test_file,
        },
        "output_files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "summary": summary,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return SplitArtifacts(
        mode=mode,
        train_file=str(train_path),
        val_file=str(val_path),
        test_file=str(test_path),
        manifest_path=str(manifest_path),
        summary=summary,
    )


__all__ = ["SplitArtifacts", "prepare_split_artifacts"]
