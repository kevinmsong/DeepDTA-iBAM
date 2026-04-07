"""Train the modular DeepDTA-iBAM stack from reusable config profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config_profiles import ExperimentConfig, get_config_profile
from data.splits import prepare_split_artifacts
from training.engine import train_ensemble


PROFILE_CHOICES = [
    "baseline_repro",
    "max_rmse_cluster",
    "diffusion_egfr_seed",
    "max_rmse_cluster_diffusion",
    "max_rmse_cluster_no_fusion",
    "inference",
    # Ablation profiles (single-member; sweep seeds via run_ablations.py)
    "abl_base",
    "abl_no_fusion",
    "abl_no_ranking",
    "abl_no_diffusion",
    "abl_full",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DeepDTA-iBAM with reusable config profiles.")
    parser.add_argument(
        "--profile",
        default="max_rmse_cluster",
        choices=PROFILE_CHOICES,
        help="Named experiment profile.",
    )
    parser.add_argument(
        "--profile-name-override",
        type=str,
        default=None,
        help="Override the runtime profile name used for checkpoint and summary file names.",
    )
    parser.add_argument(
        "--train-subset-fraction",
        type=float,
        default=None,
        help="Optional subset fraction override for fast smoke tests.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Optional ensemble size override.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional max epoch override.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional early-stopping patience override.",
    )
    parser.add_argument(
        "--build-caches-on-start",
        action="store_true",
        help="Force cache rebuild at training start even if cache files already exist.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest complete checkpoint ensemble for the selected profile.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Override the cache directory for graph and protein caches.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override the checkpoint directory.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override the log directory.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["bidirectional", "none"],
        default=None,
        help="Optional fusion mode override.",
    )
    parser.add_argument(
        "--enable-diffusion",
        action="store_true",
        help="Enable diffusion training on top of the selected base profile.",
    )
    parser.add_argument(
        "--disable-diffusion",
        action="store_true",
        help="Disable diffusion training even if the selected profile enables it.",
    )
    parser.add_argument(
        "--diffusion-max-weight",
        type=float,
        default=None,
        help="Optional diffusion loss weight override.",
    )
    parser.add_argument(
        "--diffusion-warmup-epochs",
        type=int,
        default=None,
        help="Optional diffusion warmup epoch override.",
    )
    parser.add_argument(
        "--diffusion-ramp-end-epoch",
        type=int,
        default=None,
        help="Optional diffusion ramp end epoch override.",
    )
    parser.add_argument(
        "--diff-hidden-dim",
        type=int,
        default=None,
        help="Optional diffusion hidden dimension override.",
    )
    parser.add_argument(
        "--diff-T",
        type=int,
        default=None,
        help="Optional total diffusion timestep override.",
    )
    parser.add_argument(
        "--diff-inference-steps",
        type=int,
        default=None,
        help="Optional diffusion sampling step override.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["scaffold", "standard"],
        default="scaffold",
        help="Dataset split strategy.",
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
    parser.add_argument(
        "--split-train-ratio",
        type=float,
        default=0.9,
        help="Train ratio for generated hardened splits.",
    )
    parser.add_argument(
        "--split-val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio for generated hardened splits.",
    )
    parser.add_argument(
        "--split-test-ratio",
        type=float,
        default=0.05,
        help="Test ratio for generated hardened splits.",
    )
    return parser


def apply_runtime_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.profile_name_override is not None:
        config.profile_name = args.profile_name_override
    if args.train_subset_fraction is not None:
        config.train_subset_fraction = args.train_subset_fraction
    if args.device is not None:
        config.device = args.device
    if args.ensemble_size is not None:
        config.ensemble_size = args.ensemble_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.patience is not None:
        config.patience = args.patience
    if args.build_caches_on_start:
        config.build_caches_on_start = True
    if args.cache_root is not None:
        config.cache_root = args.cache_root
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.fusion_mode is not None:
        config.fusion_mode = args.fusion_mode
    return config


def apply_diffusion_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    diffusion_enabled = config.diffusion_max_weight > 0.0
    if args.enable_diffusion:
        diffusion_enabled = True
        if args.profile_name_override is None and args.profile == "max_rmse_cluster":
            config.profile_name = "max_rmse_cluster_diffusion"
    if args.disable_diffusion:
        diffusion_enabled = False

    if diffusion_enabled and config.diffusion_max_weight <= 0.0:
        config.diffusion_max_weight = 0.02
        if config.diffusion_warmup_epochs <= 0:
            config.diffusion_warmup_epochs = 2
        if config.diffusion_ramp_end_epoch <= config.diffusion_warmup_epochs:
            config.diffusion_ramp_end_epoch = 6

    if not diffusion_enabled:
        config.diffusion_max_weight = 0.0
        config.diffusion_warmup_epochs = 0
        config.diffusion_ramp_end_epoch = 0

    if args.diffusion_max_weight is not None:
        config.diffusion_max_weight = args.diffusion_max_weight
    if args.diffusion_warmup_epochs is not None:
        config.diffusion_warmup_epochs = args.diffusion_warmup_epochs
    if args.diffusion_ramp_end_epoch is not None:
        config.diffusion_ramp_end_epoch = args.diffusion_ramp_end_epoch
    if args.diff_hidden_dim is not None:
        config.diff_hidden_dim = args.diff_hidden_dim
    if args.diff_T is not None:
        config.diff_T = args.diff_T
    if args.diff_inference_steps is not None:
        config.diff_inference_steps = args.diff_inference_steps
    return config


def prepare_training_config(args: argparse.Namespace) -> ExperimentConfig:
    config = get_config_profile(args.profile)
    config = apply_runtime_overrides(config, args)
    config = apply_diffusion_overrides(config, args)
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = prepare_training_config(args)

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

    resume_requested = bool(args.resume)
    checkpoint_dir = Path(config.checkpoint_dir)
    if resume_requested and not checkpoint_dir.exists():
        print(
            f"[train_rmse] requested resume but checkpoint_dir does not exist yet: {checkpoint_dir}. "
            "Starting a fresh run instead.",
            flush=True,
        )
        resume_requested = False

    print(
        json.dumps(
            {
                "profile": config.profile_name,
                "diffusion_enabled": config.diffusion_max_weight > 0.0,
                "diffusion_max_weight": config.diffusion_max_weight,
                "diffusion_warmup_epochs": config.diffusion_warmup_epochs,
                "diffusion_ramp_end_epoch": config.diffusion_ramp_end_epoch,
                "checkpoint_dir": config.checkpoint_dir,
                "cache_root": config.cache_root,
                "resume": resume_requested,
            },
            indent=2,
        ),
        flush=True,
    )

    summary = train_ensemble(config, resume=resume_requested)
    summary["data_split"] = split_artifacts.summary
    if split_artifacts.manifest_path is not None:
        summary["data_split"]["manifest_path"] = split_artifacts.manifest_path

    output_path = Path(config.checkpoint_dir) / f"{config.profile_name}_training_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
