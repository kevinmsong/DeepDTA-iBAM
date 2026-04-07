"""Precompute graph and protein caches for DeepDTA-iBAM."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config_profiles import get_config_profile
from data.cache_builders import GraphCacheBuilder, ProteinEmbeddingCacheBuilder
from data.splits import prepare_split_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph and protein caches for DeepDTA-iBAM.")
    parser.add_argument(
        "--profile",
        default="max_rmse_cluster",
        choices=["baseline_repro", "max_rmse_cluster", "inference"],
        help="Named experiment profile.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional cache-builder device override.")
    parser.add_argument(
        "--split-mode",
        choices=["scaffold"],
        default="scaffold",
        help="Dataset split strategy. Only scaffold-hardened splits are supported.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional seed override for generated hardened splits. Defaults to the config seed.",
    )
    parser.add_argument(
        "--split-output-root",
        type=str,
        default="data/generated_splits",
        help="Directory where generated hardened split CSVs and manifests are written.",
    )
    parser.add_argument("--split-train-ratio", type=float, default=0.9, help="Train ratio for generated hardened splits.")
    parser.add_argument("--split-val-ratio", type=float, default=0.05, help="Validation ratio for generated hardened splits.")
    parser.add_argument("--split-test-ratio", type=float, default=0.05, help="Test ratio for generated hardened splits.")
    args = parser.parse_args()

    config = get_config_profile(args.profile)
    if args.device is not None:
        config.device = args.device

    split_artifacts = prepare_split_artifacts(
        train_file=config.train_file,
        val_file=config.val_file,
        test_file=config.test_file,
        mode=args.split_mode,
        output_root=args.split_output_root,
        seed=config.seed if args.split_seed is None else args.split_seed,
        train_ratio=args.split_train_ratio,
        val_ratio=args.split_val_ratio,
        test_ratio=args.split_test_ratio,
    )
    config.train_file = split_artifacts.train_file
    config.val_file = split_artifacts.val_file
    config.test_file = split_artifacts.test_file

    train_df = pd.read_csv(config.train_file)
    val_df = pd.read_csv(config.val_file)
    test_df = pd.read_csv(config.test_file)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    graph_manifest = GraphCacheBuilder(config).build(all_df["compound_iso_smiles"].tolist())
    protein_manifest = ProteinEmbeddingCacheBuilder(config).build(all_df["target_sequence"].tolist())
    print(
        json.dumps(
            {
                "data_split": split_artifacts.summary,
                "graph_manifest": graph_manifest,
                "protein_manifest": protein_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
