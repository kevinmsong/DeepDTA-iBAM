"""Efficiency, scalability and cost profile of the DeepDTA-iBAM inference path.

The architectural claim this study makes is an interface claim, not an accuracy
claim: the interaction map and the affinity prediction come out of the same
forward pass, so interpretability is available at no separate inference cost.
That is a quantitative statement and it should be measured rather than asserted.

Reported here:

  * parameter counts by module, including the share attributable to the
    cross-attention fusion that produces the interaction maps;
  * inference throughput and latency on held-out KIBA pairs across batch sizes,
    with and without attention collection, so the marginal cost of emitting the
    map is a measured number;
  * scaling of latency with protein length and with ligand size, which is where
    the cross-attention step grows;
  * peak memory by batch size, and the on-disk and one-time build cost of the
    cached protein embeddings.

All timings here are single-process CPU numbers on the machine that ran the
analysis; the checkpoint itself was trained on GPU.  CPU numbers are the
relevant ones for a screening deployment without accelerators, and they are
reported as such rather than as the model's best achievable speed.

Run from anywhere:  python analysis/analysis_efficiency.py
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

from config_profiles import get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import (
    make_unlabeled_prediction_loader,
    _move_batch_to_device,
    _autocast_context,
    runtime_device,
)
from case_studies_results_generation import load_publication_ensemble

TEST_CSV = ROOT / "data" / "raw" / "test_kiba.csv"
CACHE_DIR = RESULTS / "cache" / "efficiency"
OUT_JSON = RESULTS / "efficiency_profile.json"
OUT_CSV = RESULTS / "efficiency_latency.csv"

SEED = 1337
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
N_TIMED_PAIRS = 256
N_WARMUP = 2


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def count_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()))


def module_breakdown(model) -> dict:
    total = count_params(model)
    parts = {}
    for name in ("gat", "protein_adapter", "fusion", "affinity_head", "diffusion"):
        mod = getattr(model, name, None)
        parts[name] = count_params(mod) if mod is not None else 0
    parts["other"] = total - sum(parts.values())
    out = {"total_parameters": total, "by_module": parts,
           "by_module_fraction": {k: (v / total if total else 0.0) for k, v in parts.items()}}
    # The cross-attention fusion is what makes the interaction map available.
    out["interaction_map_parameter_share"] = parts["fusion"] / total if total else 0.0
    # The diffusion head is only used for analog proposal, not for scoring.
    out["scoring_only_parameters"] = total - parts["diffusion"]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    test = pd.read_csv(TEST_CSV)
    sel = rng.choice(len(test), size=min(N_TIMED_PAIRS, len(test)), replace=False)
    sel.sort()
    sample = test.iloc[sel].reset_index(drop=True)
    smiles = sample["compound_iso_smiles"].astype(str).tolist()
    seqs = sample["target_sequence"].astype(str).tolist()

    cfg = get_config_profile("max_rmse_cluster_diffusion")
    cfg.build_caches_on_start = False
    cfg.ensemble_size = 1

    log("building caches (timed)")
    t0 = time.time()
    iso, gc_, pc_ = build_isolated_caches(
        cfg, sorted(set(smiles)), sorted(set(seqs)), str(CACHE_DIR),
        force_rebuild=False, cache_prefix="efficiency",
    )
    cache_seconds = time.time() - t0
    iso.num_workers = 0

    models, norm = load_publication_ensemble(iso, checkpoint_dir=cfg.checkpoint_dir)
    model = models[0]
    model.eval()
    device = runtime_device(iso)

    out: dict = {
        "seed": SEED,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "torch_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "note": "CPU inference timings; the released checkpoint was trained on GPU",
        "parameters": module_breakdown(model),
        "caches": {
            "protein_embedding_cache_bytes": dir_size_bytes(ROOT / "data" / "cache"),
            "sample_cache_build_seconds": round(cache_seconds, 2),
            "n_unique_proteins_in_sample": len(set(seqs)),
            "n_unique_ligands_in_sample": len(set(smiles)),
        },
    }

    log(f"total parameters {out['parameters']['total_parameters']:,}")

    # ---------------------------------------------------- throughput sweep
    frame = pd.DataFrame({"compound_iso_smiles": smiles, "target_sequence": seqs})
    sweep = []
    per_batch_rows = []

    max_len = max(len(s) for s in seqs) + 8
    oom_at = None
    for bs in BATCH_SIZES:
        if oom_at is not None and bs >= oom_at:
            # Once a batch size exhausts memory, every larger one will too, and
            # repeatedly provoking a multi-gigabyte failed allocation leaves the
            # process in a fragile state.  Record and move on.
            out.setdefault("memory_limits", []).append({
                "batch_size": bs, "collect_attention": None,
                "error": f"skipped: batch size {oom_at} already exhausted memory",
            })
            continue
        for collect in (False, True):
            # Batching is by token budget, not by batch_size: the loader uses a
            # TokenBudgetBatchSampler.  Setting batch_size alone leaves the
            # actual batch unchanged, which silently collapses the sweep.
            iso.max_pairs_per_batch = bs
            iso.protein_token_budget = bs * max_len
            loader = make_unlabeled_prediction_loader(frame, gc_, pc_, iso)
            times = []
            n_done = 0
            oom = None
            try:
                with torch.no_grad():
                    for bi, batch in enumerate(loader, start=1):
                        batch = _move_batch_to_device(batch, device)
                        t0 = time.perf_counter()
                        with _autocast_context(iso):
                            output, attention_maps, _ = model(
                                batch["drug_x"], batch["drug_adj"], batch["drug_mask"],
                                batch["protein_embeddings"], batch["protein_mask"],
                                drug_edge_features=batch["drug_edge_features"],
                                compute_diff_loss=False,
                            )
                        if collect:
                            # Materialize the maps the way an interpretability run does.
                            _ = {k: v.detach().float().cpu()
                                 for k, v in attention_maps.items()
                                 if isinstance(v, torch.Tensor)}
                        dt = time.perf_counter() - t0
                        bsz = int(batch["drug_mask"].shape[0])
                        if bi > N_WARMUP:
                            times.append(dt)
                            n_done += bsz
                            per_batch_rows.append({
                                "batch_size": bs, "collect_attention": int(collect),
                                "batch_index": bi, "pairs": bsz, "seconds": dt,
                                "max_protein_tokens": int(batch["protein_mask"].sum(dim=1).max()),
                                "max_ligand_atoms": int(batch["drug_mask"].sum(dim=1).max()),
                            })
            except RuntimeError as exc:
                # A large batch containing a long target exhausts memory in the
                # protein adapter's self-attention, which grows with the square
                # of sequence length times the batch.  That limit is itself a
                # scalability result, so record it instead of aborting.
                if "not enough memory" not in str(exc) and "out of memory" not in str(exc):
                    raise
                oom = str(exc).strip().splitlines()[-1]
                oom_at = bs if oom_at is None else min(oom_at, bs)
                log(f"  pairs/batch {bs:3d} attention={int(collect)}: out of memory")
                out.setdefault("memory_limits", []).append({
                    "batch_size": bs, "collect_attention": bool(collect),
                    "error": oom,
                })
                gc.collect()
            if oom is not None:
                # Timings collected before the failure cover only the batches
                # that happened to hold short targets, so they would understate
                # the cost at this batch size.  Record the limit, not a number.
                per_batch_rows[:] = [r for r in per_batch_rows
                                     if not (r["batch_size"] == bs
                                             and r["collect_attention"] == int(collect))]
                continue
            if not times:
                continue
            arr = np.array(times)
            observed = [r["pairs"] for r in per_batch_rows
                        if r["batch_size"] == bs and r["collect_attention"] == int(collect)]
            sweep.append({
                "batch_size": bs,
                "observed_pairs_per_batch_median": float(np.median(observed)) if observed else float("nan"),
                "collect_attention": bool(collect),
                "pairs_timed": n_done,
                "batches_timed": len(arr),
                "seconds_per_batch_median": float(np.median(arr)),
                "seconds_per_batch_p95": float(np.percentile(arr, 95)),
                "pairs_per_second": float(n_done / arr.sum()),
                "ms_per_pair": float(1000.0 * arr.sum() / max(n_done, 1)),
            })
            log(f"  pairs/batch {bs:3d} (observed "
                f"{sweep[-1]['observed_pairs_per_batch_median']:.0f}) "
                f"attention={int(collect)}: "
                f"{sweep[-1]['pairs_per_second']:.2f} pairs/s, "
                f"{sweep[-1]['ms_per_pair']:.1f} ms/pair")

    out["throughput"] = sweep

    # Marginal cost of emitting the interaction map, matched on batch size.
    marginal = {}
    for bs in BATCH_SIZES:
        off = next((s for s in sweep if s["batch_size"] == bs and not s["collect_attention"]), None)
        on = next((s for s in sweep if s["batch_size"] == bs and s["collect_attention"]), None)
        if off and on and off["ms_per_pair"] > 0:
            marginal[str(bs)] = {
                "ms_per_pair_without_maps": off["ms_per_pair"],
                "ms_per_pair_with_maps": on["ms_per_pair"],
                "relative_overhead": on["ms_per_pair"] / off["ms_per_pair"] - 1.0,
            }
    out["interaction_map_marginal_cost"] = marginal

    df = pd.DataFrame(per_batch_rows)
    df.to_csv(OUT_CSV, index=False)
    # Persist what is measured so far; the scaling loop below is a separate
    # pass and a failure there should not discard the throughput sweep.
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # --------------------------------------------------- scaling with length
    # Measured one pair at a time so each timing belongs to a single protein and
    # a single ligand, which is what makes a regression on length meaningful.
    log("timing single pairs for the scaling fit")
    iso.max_pairs_per_batch = 1
    iso.protein_token_budget = max_len
    loader = make_unlabeled_prediction_loader(frame, gc_, pc_, iso)
    single_rows = []
    with torch.no_grad():
        for bi, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, device)
            t0 = time.perf_counter()
            with _autocast_context(iso):
                model(batch["drug_x"], batch["drug_adj"], batch["drug_mask"],
                      batch["protein_embeddings"], batch["protein_mask"],
                      drug_edge_features=batch["drug_edge_features"],
                      compute_diff_loss=False)
            dt = time.perf_counter() - t0
            if bi > N_WARMUP:
                single_rows.append({
                    "seconds": dt,
                    "protein_tokens": int(batch["protein_mask"].sum()),
                    "ligand_atoms": int(batch["drug_mask"].sum()),
                })
    sdf = pd.DataFrame(single_rows)
    sdf.to_csv(RESULTS / "efficiency_single_pair_latency.csv", index=False)

    if len(sdf) > 10:
        for col, label in (("protein_tokens", "protein_length"),
                           ("ligand_atoms", "ligand_atoms")):
            x = sdf[col].to_numpy(float)
            y = sdf["seconds"].to_numpy(float)
            if x.std() > 0:
                slope, intercept = np.polyfit(x, y, 1)
                out.setdefault("scaling", {})[label] = {
                    "slope_ms_per_unit": float(1000 * slope),
                    "intercept_ms": float(1000 * intercept),
                    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                    "n": int(len(x)),
                    "range": [float(x.min()), float(x.max())],
                }
        out["single_pair_latency_ms"] = {
            "median": float(1000 * sdf["seconds"].median()),
            "p95": float(1000 * sdf["seconds"].quantile(0.95)),
        }

    OUT_JSON.write_text(json.dumps(out, indent=2))

    p = out["parameters"]
    print("\n--- parameters by module ---")
    for k, v in p["by_module"].items():
        print(f"  {k:18s} {v:>12,}  ({p['by_module_fraction'][k]:6.2%})")
    print(f"  {'TOTAL':18s} {p['total_parameters']:>12,}")
    print(f"\ninteraction-map (fusion) parameter share: {p['interaction_map_parameter_share']:.2%}")
    print("\n--- marginal cost of emitting interaction maps ---")
    for bs, m in marginal.items():
        print(f"  batch {bs:>3s}: {m['ms_per_pair_without_maps']:7.1f} -> "
              f"{m['ms_per_pair_with_maps']:7.1f} ms/pair "
              f"({m['relative_overhead']:+.1%})")
    print(f"\nwrote {OUT_JSON.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
