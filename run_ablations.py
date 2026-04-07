"""
HPC-ready ablation runner for DeepDTA-iBAM.

Ablation matrix: 5 configs × 2 splits × 3 seeds = 30 task slots.

Usage
-----
# Direct (one combination):
  python run_ablations.py --config abl_full --split scaffold --seed 1337

# SLURM array job:
  python run_ablations.py --task-id "$SLURM_ARRAY_TASK_ID"
  # OR: env var is read automatically when neither --task-id nor --config is set
  #     SLURM_ARRAY_TASK_ID=5 python run_ablations.py

# Run all 30 combinations locally (sequential):
  python run_ablations.py --all

# Smoke-test (1 epoch, fast):
  python run_ablations.py --config abl_full --split standard --seed 1337 --num-epochs 1

# Print the full matrix (no run):
  python run_ablations.py --dry-run

# Print a ready-to-submit SLURM script:
  python run_ablations.py --print-slurm > submit_ablations.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from config_profiles import ExperimentConfig, get_config_profile
from data.cache_builders import load_or_build_caches
from data.splits import prepare_split_artifacts
from training.engine import evaluate_ensemble, train_ensemble
from training.inference import load_ensemble, make_prediction_loader
from utils.metrics import spearman_correlation


# ---------------------------------------------------------------------------
# Ablation matrix definition
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: List[str] = [
    "abl_base",        # no cross-attention, no ranking, no diffusion
    "abl_no_fusion",   # no cross-attention, ranking + diffusion active
    "abl_no_ranking",  # full - ranking loss
    "abl_no_diffusion",# full - diffusion auxiliary loss
    "abl_full",        # all components enabled
]

SPLIT_TYPES: List[str] = ["standard", "scaffold"]

SEEDS: List[int] = [1337, 2674, 4011]

_MATRIX: List[Tuple[str, str, int]] = [
    (cfg, split, seed)
    for cfg in ABLATION_CONFIGS
    for split in SPLIT_TYPES
    for seed in SEEDS
]  # 30 entries, 0-indexed

_CSV_FIELDS = ["model_name", "split_type", "seed", "RMSE", "MAE", "CI", "Pearson", "Spearman"]

# ---------------------------------------------------------------------------
# SLURM template
# ---------------------------------------------------------------------------

_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=dta_ablation
#SBATCH --array=0-{max_id}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

# Adjust to your cluster environment:
# module load python/3.10
# conda activate deepdta

cd {workdir}
python run_ablations.py --task-id "$SLURM_ARRAY_TASK_ID"
"""


# ---------------------------------------------------------------------------
# Per-run logic
# ---------------------------------------------------------------------------

def run_one(
    config_name: str,
    split_type: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Train and evaluate one ablation combination. Returns metrics dict."""

    run_dir = Path(args.output_root) / config_name / split_type / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(
            f"[skip] {config_name}/{split_type}/seed_{seed} — "
            f"metrics.json exists (use --force to rerun)",
            flush=True,
        )
        with metrics_path.open() as fh:
            return json.load(fh)

    # Build config and apply per-run overrides
    config: ExperimentConfig = get_config_profile(config_name)
    config.seed = seed
    config.checkpoint_dir = str(checkpoint_dir)
    config.log_dir = str(log_dir)
    config.build_caches_on_start = False  # caches pre-built or built once per cluster node

    if getattr(args, "num_epochs", None) is not None:
        config.num_epochs = args.num_epochs
    if getattr(args, "patience", None) is not None:
        config.patience = args.patience
    if getattr(args, "device", None) is not None:
        config.device = args.device

    # Prepare data splits
    split_artifacts = prepare_split_artifacts(
        train_file=config.train_file,
        val_file=config.val_file,
        test_file=config.test_file,
        mode=split_type,
        output_root=args.split_output_root,
        seed=seed,
        train_ratio=args.split_train_ratio,
        val_ratio=args.split_val_ratio,
        test_ratio=args.split_test_ratio,
    )
    config.train_file = split_artifacts.train_file
    config.val_file = split_artifacts.val_file
    config.test_file = split_artifacts.test_file

    print(
        f"\n[run] config={config_name}  split={split_type}  seed={seed}  "
        f"epochs={config.num_epochs}  output={run_dir}",
        flush=True,
    )
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    resume = getattr(args, "resume", False)
    # Auto-detect partial run: checkpoint dir has files but metrics.json absent
    has_partial_checkpoint = any(checkpoint_dir.glob("*.safetensors"))
    if resume and has_partial_checkpoint:
        print(f"  [resume] found existing checkpoints in {checkpoint_dir}", flush=True)
    summary = train_ensemble(config, resume=resume and has_partial_checkpoint)

    # -----------------------------------------------------------------------
    # Re-evaluation on test split to obtain raw predictions for Spearman
    # (train_ensemble evaluates internally but strips raw predictions from
    # its return value)
    # -----------------------------------------------------------------------
    graph_cache, protein_cache = load_or_build_caches(config)

    checkpoint_paths = [
        checkpoint_dir / f"{config_name}_member_{i}.safetensors"
        for i in range(config.ensemble_size)
    ]
    missing = [p for p in checkpoint_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint(s) missing after training: "
            + ", ".join(str(p) for p in missing)
        )

    models, normalizer = load_ensemble(checkpoint_paths, config)
    test_df = pd.read_csv(config.test_file).reset_index(drop=True)
    test_loader = make_prediction_loader(
        test_df, graph_cache, protein_cache, config, normalizer
    )
    test_metrics = evaluate_ensemble(models, test_loader, config, normalizer)

    spearman = float(
        spearman_correlation(test_metrics["targets"], test_metrics["predictions"])
    )

    elapsed = time.time() - t0

    result: Dict[str, Any] = {
        "model_name": config_name,
        "split_type": split_type,
        "seed": seed,
        "RMSE": float(test_metrics["RMSE"]),
        "MAE": float(test_metrics["MAE"]),
        "CI": float(test_metrics["CI"]),
        "Pearson": float(test_metrics["Pearson"]),
        "Spearman": spearman,
        "R2": float(test_metrics.get("R2", float("nan"))),
        "elapsed_seconds": round(elapsed, 1),
        "run_dir": str(run_dir),
        "best_epoch": summary.get("best_epoch"),
        "best_val_rmse": summary.get("best_val_rmse"),
        "best_val_ci": summary.get("best_val_ci"),
        "split_summary": split_artifacts.summary,
    }

    # Save per-run JSON
    with metrics_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[saved] {metrics_path}", flush=True)

    # Append row to the shared raw_results.csv
    _append_csv_row(args.output_root, result)

    return result


# ---------------------------------------------------------------------------
# CSV append (concurrent-safe on Linux via fcntl, best-effort on Windows)
# ---------------------------------------------------------------------------

def _append_csv_row(output_root: str, result: Dict[str, Any]) -> None:
    csv_path = Path(output_root) / "raw_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: result[k] for k in _CSV_FIELDS}

    if platform.system() != "Windows":
        import fcntl
        lock_path = csv_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                _write_csv_row(csv_path, row)
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
    else:
        _write_csv_row(csv_path, row)

    print(f"[csv]  appended to {csv_path}", flush=True)


def _write_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Selection modes (mutually exclusive group)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--task-id",
        type=int,
        default=None,
        metavar="ID",
        help=f"SLURM array task ID (0–{len(_MATRIX) - 1}). "
             "If omitted and SLURM_ARRAY_TASK_ID is set it is used automatically.",
    )
    sel.add_argument(
        "--all",
        action="store_true",
        help=f"Run all {len(_MATRIX)} combinations sequentially.",
    )

    # Direct selection (used only when --task-id / --all not given)
    p.add_argument("--config", choices=ABLATION_CONFIGS, default=None,
                   help="Ablation config name.")
    p.add_argument("--split", choices=SPLIT_TYPES, default=None,
                   help="Split type.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed.")

    # Paths
    p.add_argument("--output-root", default="results/ablations",
                   help="Root directory for run outputs.")
    p.add_argument("--split-output-root", default="data/generated_splits",
                   help="Directory for generated split CSVs.")
    p.add_argument("--split-train-ratio", type=float, default=0.9)
    p.add_argument("--split-val-ratio", type=float, default=0.05)
    p.add_argument("--split-test-ratio", type=float, default=0.05)

    # Training overrides
    p.add_argument("--num-epochs", type=int, default=None,
                   help="Override max training epochs (e.g. 1 for smoke test).")
    p.add_argument("--patience", type=int, default=None,
                   help="Override early-stopping patience.")
    p.add_argument("--device", type=str, default=None,
                   help="Device override, e.g. 'cuda' or 'cpu'.")
    p.add_argument("--resume", action="store_true",
                   help="Resume partial runs from existing checkpoints. "
                        "Completed runs (metrics.json present) are always skipped "
                        "unless --force is also set.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if metrics.json already exists.")

    # Utility
    p.add_argument("--dry-run", action="store_true",
                   help="Print the ablation matrix and exit without running.")
    p.add_argument("--print-slurm", action="store_true",
                   help="Print a SLURM submission script to stdout and exit.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- utility modes ---
    if args.dry_run:
        print(f"Ablation matrix — {len(_MATRIX)} combinations:\n")
        print(f"{'ID':>3}  {'config':<20} {'split':<10} seed")
        print("-" * 48)
        for i, (cfg, spl, s) in enumerate(_MATRIX):
            print(f"{i:>3}  {cfg:<20} {spl:<10} {s}")
        return

    if args.print_slurm:
        print(_SLURM_TEMPLATE.format(max_id=len(_MATRIX) - 1, workdir=Path.cwd()))
        return

    # --- determine task list ---
    tasks: List[Tuple[str, str, int]]

    if args.all:
        tasks = _MATRIX[:]
    elif args.task_id is not None:
        if not (0 <= args.task_id < len(_MATRIX)):
            parser.error(f"--task-id must be 0–{len(_MATRIX) - 1}")
        tasks = [_MATRIX[args.task_id]]
    elif os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not (0 <= task_id < len(_MATRIX)):
            sys.exit(f"SLURM_ARRAY_TASK_ID={task_id} out of range 0–{len(_MATRIX) - 1}")
        tasks = [_MATRIX[task_id]]
    elif args.config is not None and args.split is not None and args.seed is not None:
        tasks = [(args.config, args.split, args.seed)]
    else:
        parser.error(
            "Specify one of: --task-id INT, --all, SLURM_ARRAY_TASK_ID env var, "
            "or --config + --split + --seed."
        )

    for config_name, split_type, seed in tasks:
        run_one(config_name, split_type, seed, args)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
