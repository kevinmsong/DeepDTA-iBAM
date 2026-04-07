"""Evaluate saved DeepDTA-iBAM checkpoints on scaffold or standard KIBA splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from config_profiles import get_config_profile
from data.cache_builders import load_or_build_caches
from data.splits import prepare_split_artifacts
from training.engine import evaluate_ensemble
from training.inference import load_ensemble, make_prediction_loader


PROFILE_CHOICES = [
    "baseline_repro",
    "max_rmse_cluster",
    "max_rmse_cluster_diffusion",
    "max_rmse_cluster_no_fusion",
    "diffusion_egfr_seed",
    "inference",
    # Ablation profiles
    "abl_base",
    "abl_no_fusion",
    "abl_no_ranking",
    "abl_no_diffusion",
    "abl_full",
]

_HPC_REFERENCE: Dict[str, float] = {
    "RMSE": 0.7242,
    "MAE": 0.4701,
    "CI": 0.7708,
    "R2": 0.3982,
    "Pearson": 0.6753,
}
_MATCH_TOLERANCE = 0.001

_DISPLAY_KEYS: Tuple[str, ...] = (
    "RMSE",
    "MAE",
    "CI",
    "R2",
    "Pearson",
    "AttentionEntropy",
    "AttentionMaxWeight",
    "AttentionCollapseFrac",
    "AttentionCollapseWarn",
)
_RAW_KEYS = {"predictions", "targets"}


def _print_metrics_table(split_name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== {split_name.capitalize()} Split Metrics ===")
    for key in _DISPLAY_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            print(f"  {key:<26}: {value:.6f}")
        else:
            print(f"  {key:<26}: {value}")


def _print_hpc_check(metrics: Dict[str, Any]) -> None:
    print("\n=== HPC Reference Check (5-member scaffold val split) ===")
    all_pass = True
    for metric, reference in _HPC_REFERENCE.items():
        computed = metrics.get(metric)
        if computed is None:
            print(f"  {metric:<8}  computed=N/A       reference={reference:.4f}  MISSING")
            all_pass = False
            continue
        delta = computed - reference
        status = "PASS" if abs(delta) <= _MATCH_TOLERANCE else "WARN"
        if status == "WARN":
            all_pass = False
        print(f"  {metric:<8}  computed={computed:.4f}  reference={reference:.4f}  delta={delta:+.4f}  {status}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME METRICS DIFFER'}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved DeepDTA-iBAM checkpoints on scaffold or standard KIBA splits."
    )
    parser.add_argument(
        "--profile",
        default="max_rmse_cluster_diffusion",
        choices=PROFILE_CHOICES,
        help="Named experiment profile.",
    )
    parser.add_argument(
        "--profile-name-override",
        type=str,
        default=None,
        help="Override the checkpoint filename prefix without changing the architecture defaults.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional checkpoint directory override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--member-count",
        type=int,
        default=None,
        help="Evaluate only the first N checkpoint members.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["bidirectional", "none"],
        default=None,
        help="Optional fusion mode override for the loaded architecture.",
    )
    parser.add_argument(
        "--rebuild-caches",
        action="store_true",
        help="Force cache rebuild even if cache files already exist.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "both"],
        default="both",
        help="Which split(s) to evaluate.",
    )
    parser.add_argument(
        "--data-split",
        choices=["scaffold", "standard"],
        default="scaffold",
        help="Evaluate on the scaffold-hardened split or the original standard split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for scaffold split generation. Ignored for standard split evaluation.",
    )
    parser.add_argument(
        "--split-output-root",
        type=str,
        default="data/generated_splits",
        help="Directory for generated split CSVs.",
    )
    parser.add_argument(
        "--split-train-ratio",
        type=float,
        default=0.9,
        help="Train ratio.",
    )
    parser.add_argument(
        "--split-val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio.",
    )
    parser.add_argument(
        "--split-test-ratio",
        type=float,
        default=0.05,
        help="Test ratio.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional metrics JSON path override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for evaluation (default: 0 for simpler monitored runs).",
    )
    parser.add_argument(
        "--max-pairs-per-batch",
        type=int,
        default=None,
        help="Optional evaluation batch cap to avoid CUDA OOM on long proteins.",
    )
    parser.add_argument(
        "--protein-token-budget",
        type=int,
        default=None,
        help="Optional protein token budget override for evaluation.",
    )
    return parser


def _saveable_metrics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        split_name: {key: value for key, value in metrics.items() if key not in _RAW_KEYS}
        for split_name, metrics in results.items()
    }


def main() -> None:
    args = _build_arg_parser().parse_args()

    config = get_config_profile(args.profile)
    if args.profile_name_override is not None:
        config.profile_name = args.profile_name_override
    if args.device is not None:
        config.device = args.device
    if args.fusion_mode is not None:
        config.fusion_mode = args.fusion_mode
    if args.member_count is not None:
        config.ensemble_size = args.member_count
    if args.rebuild_caches:
        config.build_caches_on_start = True
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    config.num_workers = args.num_workers
    if args.max_pairs_per_batch is not None:
        config.max_pairs_per_batch = args.max_pairs_per_batch
    if args.protein_token_budget is not None:
        config.protein_token_budget = args.protein_token_budget

    print(
        f"[eval] profile={config.profile_name} device={config.resolved_device} "
        f"members={config.ensemble_size} data_split={args.data_split}"
    )

    split_seed = config.seed if args.split_seed is None else args.split_seed
    split_artifacts = prepare_split_artifacts(
        train_file=config.train_file,
        val_file=config.val_file,
        test_file=config.test_file,
        mode=args.data_split,
        output_root=args.split_output_root,
        seed=split_seed,
        train_ratio=args.split_train_ratio,
        val_ratio=args.split_val_ratio,
        test_ratio=args.split_test_ratio,
    )
    config.train_file = split_artifacts.train_file
    config.val_file = split_artifacts.val_file
    config.test_file = split_artifacts.test_file
    print(
        f"[eval] Split rows: train={split_artifacts.summary['rows']['train']} "
        f"val={split_artifacts.summary['rows']['val']} test={split_artifacts.summary['rows']['test']}"
    )

    print("[eval] Loading caches" + (" (rebuilding forced)" if args.rebuild_caches else ""))
    graph_cache, protein_cache = load_or_build_caches(config)

    checkpoint_paths: List[Path] = [
        Path(config.checkpoint_dir) / f"{config.profile_name}_member_{member_idx}.safetensors"
        for member_idx in range(config.ensemble_size)
    ]
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint file(s):\n" + "\n".join(f"  {path}" for path in missing))

    print(f"[eval] Loading {len(checkpoint_paths)} checkpoint(s) from {config.checkpoint_dir}/")
    models, normalizer = load_ensemble(checkpoint_paths, config)
    print(f"[eval] Normalizer: mean={normalizer.mean:.4f} std={normalizer.std:.4f}")

    splits_to_run: List[Tuple[str, str]] = []
    if args.eval_split in ("val", "both"):
        splits_to_run.append(("val", config.val_file))
    if args.eval_split in ("test", "both"):
        splits_to_run.append(("test", config.test_file))

    results: Dict[str, Dict[str, Any]] = {}
    for split_name, csv_path in splits_to_run:
        dataframe = pd.read_csv(csv_path).reset_index(drop=True)
        print(f"\n[eval] Evaluating {split_name} split ({len(dataframe)} samples)...")
        loader = make_prediction_loader(dataframe, graph_cache, protein_cache, config, normalizer)
        metrics = evaluate_ensemble(models, loader, config, normalizer)
        results[split_name] = metrics
        _print_metrics_table(split_name, metrics)
        should_compare_hpc = (
            args.data_split == "scaffold"
            and split_name == "val"
            and config.profile_name == "max_rmse_cluster"
            and config.ensemble_size == 5
        )
        if should_compare_hpc:
            _print_hpc_check(metrics)

    payload: Dict[str, Any] = _saveable_metrics(results)
    payload["profile"] = config.profile_name
    payload["checkpoint_dir"] = config.checkpoint_dir
    payload["ensemble_size"] = config.ensemble_size
    payload["data_split"] = split_artifacts.summary

    output_path = (
        Path(args.output_json)
        if args.output_json is not None
        else Path(config.checkpoint_dir) / f"{config.profile_name}_{args.data_split}_eval_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n[eval] Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
