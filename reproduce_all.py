"""
Master reproduction script for the DeepDTA-iBAM reviewer-resistant evaluation package.

Runs all evaluation stages in order:
  1. Ablation training + evaluation  (run_ablations.py)
  2. Ablation table aggregation      (aggregate_ablations.py)
  3. Interpretability benchmark      (run_interpretability_benchmark.py)
  4. Generation validation           (run_generation_validation.py)
  5. Original publication pipeline   (case_studies_results_generation.py)

Usage
-----
  # Full sequential run (local, single GPU):
  python reproduce_all.py

  # Resume after a 12-hour HPC job cut — skips completed runs, resumes partial:
  python reproduce_all.py --resume

  # Skip stages you don't need:
  python reproduce_all.py --skip-ablations --skip-generation

  # Only aggregate already-run ablation results:
  python reproduce_all.py --skip-ablations --skip-interp --skip-generation --skip-publication

  # HPC mode: print SLURM commands instead of running:
  python reproduce_all.py --hpc

  # Override device and use 1 epoch for a smoke test:
  python reproduce_all.py --device cuda --num-epochs 1

  # Specify protein sequence for generation validation:
  python reproduce_all.py --protein-sequence MKCPQALWVK...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, check: bool = True) -> int:
    """Run a subprocess command, streaming stdout/stderr."""
    print("\n" + "=" * 70, flush=True)
    print("CMD: " + " ".join(cmd), flush=True)
    print("=" * 70, flush=True)
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"[error] command exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    return result.returncode


def _python(*args: str) -> list[str]:
    return [sys.executable] + list(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation training (use existing results).")
    parser.add_argument("--skip-interp", action="store_true",
                        help="Skip interpretability benchmark.")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation validation.")
    parser.add_argument("--skip-publication", action="store_true",
                        help="Skip the original publication pipeline "
                             "(case_studies_results_generation.py).")
    parser.add_argument("--hpc", action="store_true",
                        help="Print SLURM submission commands instead of running locally.")
    parser.add_argument("--device", default=None,
                        help="Device override passed to all sub-scripts.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Max training epochs override (e.g. 1 for smoke test).")
    parser.add_argument("--profile", default="max_rmse_cluster_diffusion",
                        help="Model profile for interpretability and generation stages.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Checkpoint directory override for eval stages.")
    parser.add_argument("--output-dir", default="results",
                        help="Root results directory.")
    parser.add_argument("--protein-sequence", default=None,
                        help="Protein sequence for generation validation "
                             "(required for diffusion generator).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints. Completed ablation runs "
                             "(metrics.json present) are skipped; partial runs pick up "
                             "from the latest saved checkpoint. Safe to re-submit after "
                             "a 12-hour HPC time limit.")
    parser.add_argument("--force", action="store_true",
                        help="Pass --force to all sub-scripts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Stage 1: Ablation training
    # -----------------------------------------------------------------------
    if not args.skip_ablations:
        if args.hpc:
            print("\n# Stage 1: Ablation training — SLURM submission")
            print("# Generate submission script:")
            print(f"  {sys.executable} run_ablations.py --print-slurm > submit_ablations.sh")
            print("# Submit:")
            print("  sbatch submit_ablations.sh")
            print("# Or run a single task:")
            print(f"  {sys.executable} run_ablations.py --task-id 0")
        else:
            ablation_cmd = _python("run_ablations.py", "--all")
            if args.device:
                ablation_cmd += ["--device", args.device]
            if args.num_epochs:
                ablation_cmd += ["--num-epochs", str(args.num_epochs)]
            if args.resume:
                ablation_cmd.append("--resume")
            if args.force:
                ablation_cmd.append("--force")
            _run(ablation_cmd)

    # -----------------------------------------------------------------------
    # Stage 2: Ablation aggregation (always runs unless only running HPC cmds)
    # -----------------------------------------------------------------------
    if not (args.skip_ablations and args.hpc):
        agg_cmd = _python("aggregate_ablations.py",
                          "--output-dir", str(output_dir))
        _run(agg_cmd, check=False)  # non-fatal; results may be partial

    # -----------------------------------------------------------------------
    # Stage 3: Interpretability benchmark
    # -----------------------------------------------------------------------
    if not args.skip_interp:
        if args.hpc:
            print("\n# Stage 3: Interpretability benchmark")
            interp_cmd = [sys.executable, "run_interpretability_benchmark.py",
                          "--profile", args.profile, "--output-dir", str(output_dir)]
            if args.checkpoint_dir:
                interp_cmd += ["--checkpoint-dir", args.checkpoint_dir]
            print("  " + " ".join(interp_cmd))
        else:
            interp_cmd = _python(
                "run_interpretability_benchmark.py",
                "--profile", args.profile,
                "--output-dir", str(output_dir),
            )
            if args.checkpoint_dir:
                interp_cmd += ["--checkpoint-dir", args.checkpoint_dir]
            if args.device:
                interp_cmd += ["--device", args.device]
            if args.force:
                interp_cmd.append("--force")
            _run(interp_cmd, check=False)  # non-fatal

    # -----------------------------------------------------------------------
    # Stage 4: Generation validation
    # -----------------------------------------------------------------------
    if not args.skip_generation:
        if args.hpc:
            print("\n# Stage 4: Generation validation")
            gen_cmd = [sys.executable, "run_generation_validation.py",
                       "--profile", args.profile, "--output-dir", str(output_dir)]
            if args.protein_sequence:
                gen_cmd += ["--protein-sequence", args.protein_sequence]
            if args.checkpoint_dir:
                gen_cmd += ["--checkpoint-dir", args.checkpoint_dir]
            print("  " + " ".join(gen_cmd))
        else:
            gen_cmd = _python(
                "run_generation_validation.py",
                "--profile", args.profile,
                "--output-dir", str(output_dir),
            )
            if args.protein_sequence:
                gen_cmd += ["--protein-sequence", args.protein_sequence]
            if args.checkpoint_dir:
                gen_cmd += ["--checkpoint-dir", args.checkpoint_dir]
            if args.device:
                gen_cmd += ["--device", args.device]
            if args.force:
                gen_cmd.append("--force")
            _run(gen_cmd, check=False)  # non-fatal

    # -----------------------------------------------------------------------
    # Stage 5: Original publication pipeline
    # -----------------------------------------------------------------------
    if not args.skip_publication:
        if args.hpc:
            print("\n# Stage 5: Publication pipeline")
            print(f"  {sys.executable} case_studies_results_generation.py")
        else:
            pub_cmd = _python("case_studies_results_generation.py")
            if args.device:
                pub_cmd += ["--device", args.device]
            _run(pub_cmd, check=False)  # non-fatal

    print("\n[reproduce_all] done", flush=True)
    print(f"Results written to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
