"""Inference-time ablation of the bidirectional cross-attention pathway.

The archived component ablation retrains each variant from scratch and resolves
nothing: with three seeds per variant every contrast returns p at or above 0.42,
and the repository's own power calculation puts the requirement at roughly 30 to
50 seeds.  Retraining at that scale is out of reach here, and more seeds would
not change what a retraining ablation can answer anyway.

This asks a different and narrower question that the released checkpoint can
answer exactly: how much does the trained model's affinity prediction depend on
the cross-attention pathway it actually learned?

Each CrossAttentionBlock mixes atom and residue tokens through two learned
scalar gates:

    atom_tokens    += atom_cross_gate    * attention_update
    residue_tokens += residue_cross_gate * attention_update

Setting those gates to zero removes the cross-modal information flow while
leaving every other weight, including the feed-forward blocks and the affinity
head, exactly as trained.  The degradation is then attributable to the pathway
rather than to a different training run reaching a different optimum.

This is not a substitute for the retraining ablation and does not claim to be.
A retrained model without cross-attention could compensate; this model cannot,
because it was not given the chance.  Both are reported.

Outputs
-------
  results/attention_ablation_summary.json
  results/attention_ablation_predictions.csv

Run from anywhere:  python analysis/analysis_attention_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from scipy import stats

from config_profiles import get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import make_unlabeled_prediction_loader, predict_unlabeled
from case_studies_results_generation import load_publication_ensemble

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_guard import apply_batch_limit

TEST_CSV = ROOT / "data" / "raw" / "test_kiba.csv"
CACHE_DIR = ROOT / "results" / "cache" / "attention_ablation"
OUT_JSON = ROOT / "results" / "attention_ablation_summary.json"
OUT_CSV = ROOT / "results" / "attention_ablation_predictions.csv"

SEED = 1337
N_BOOT = 2000
# The full test partition takes hours of CPU per pass and three passes are
# needed.  A seeded subsample keeps the comparison paired and exact.
N_SUBSAMPLE = 1200


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def concordance_index(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(y)
    y, p = y[order], p[order]
    total = hits = 0.0
    for i in range(len(y)):
        greater = y[i + 1:] > y[i]
        if not greater.any():
            continue
        d = p[i + 1:][greater] - p[i]
        total += d.size
        hits += float((d > 0).sum()) + 0.5 * float((d == 0).sum())
    return hits / total if total else float("nan")


def gate_values(model) -> dict:
    vals = {"atom_cross_gate": [], "residue_cross_gate": []}
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        return vals
    for block in fusion.blocks:
        vals["atom_cross_gate"].append(float(block.atom_cross_gate.detach()))
        vals["residue_cross_gate"].append(float(block.residue_cross_gate.detach()))
    return vals


def set_gates(model, value: float | None = None, scale: float | None = None) -> None:
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        raise SystemExit("checkpoint has no fusion module to ablate")
    with torch.no_grad():
        for block in fusion.blocks:
            if value is not None:
                block.atom_cross_gate.fill_(value)
                block.residue_cross_gate.fill_(value)
            elif scale is not None:
                block.atom_cross_gate.mul_(scale)
                block.residue_cross_gate.mul_(scale)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "pearson_r": float(stats.pearsonr(y, p)[0]),
        "spearman_rho": float(stats.spearmanr(y, p).correlation),
        "concordance_index": concordance_index(y, p),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    test = pd.read_csv(TEST_CSV)
    if N_SUBSAMPLE and N_SUBSAMPLE < len(test):
        sel = rng.choice(len(test), size=N_SUBSAMPLE, replace=False)
        sel.sort()
        test = test.iloc[sel].reset_index(drop=True)
    log(f"evaluating on {len(test)} held-out KIBA pairs")

    smiles = test["compound_iso_smiles"].astype(str).tolist()
    seqs = test["target_sequence"].astype(str).tolist()
    y = test["affinity"].to_numpy(float)

    cfg = get_config_profile("max_rmse_cluster_diffusion")
    cfg.build_caches_on_start = False
    cfg.ensemble_size = 1

    log("building caches")
    iso, gc_, pc_ = build_isolated_caches(
        cfg, sorted(set(smiles)), sorted(set(seqs)), str(CACHE_DIR),
        force_rebuild=False, cache_prefix="attention_ablation",
    )
    iso.num_workers = 0
    # KIBA targets run to several thousand residues, and the protein adapter's
    # self-attention grows with the square of sequence length times the batch,
    # so the profile default of 80 pairs per batch exhausts memory here.  Size
    # the batch from the longest target and the memory actually available.
    mem_info = apply_batch_limit(iso, seqs)

    frame = pd.DataFrame({"compound_iso_smiles": smiles, "target_sequence": seqs})
    results: dict = {"seed": SEED, "n_pairs": int(len(test)),
                     "n_bootstrap": N_BOOT, "subsampled": bool(N_SUBSAMPLE),
                     "memory": mem_info}
    preds: dict[str, np.ndarray] = {}

    conditions = [
        ("intact", None),
        ("cross_attention_off", 0.0),
        ("cross_attention_half", None),   # handled by scaling below
    ]

    for name, gate in conditions:
        models, norm = load_publication_ensemble(iso, checkpoint_dir=cfg.checkpoint_dir)
        model = models[0]
        if name == "intact":
            results["learned_gates"] = gate_values(model)
        elif name == "cross_attention_off":
            set_gates(model, value=0.0)
        elif name == "cross_attention_half":
            set_gates(model, scale=0.5)

        loader = make_unlabeled_prediction_loader(frame, gc_, pc_, iso)

        # The loader batches with a length-aware sampler that sorts pairs by
        # protein length, so predictions come back in that order, not in the
        # order of `frame`.  Recover the exact permutation from the sampler
        # itself rather than assuming it, and invert it before comparing
        # predictions with affinities.  Scripts that score a single target are
        # unaffected, which is why this only shows up here.
        order = [i for batch in loader.batch_sampler for i in batch]
        if len(order) != len(frame):
            raise SystemExit(f"sampler covered {len(order)} of {len(frame)} pairs")

        t0 = time.time()
        payload = predict_unlabeled([model], loader, iso, norm, collect_attention=False)
        p_sampler_order = np.asarray(payload["predictions"], dtype=float)
        p = np.empty_like(p_sampler_order)
        p[np.asarray(order, dtype=int)] = p_sampler_order
        preds[name] = p
        results[name] = metrics(y, p)
        results[name]["seconds"] = round(time.time() - t0, 1)
        log(f"{name:22s} CI {results[name]['concordance_index']:.4f} "
            f"RMSE {results[name]['rmse']:.4f} ({results[name]['seconds']:.0f}s)")

        # Sanity check: the unmodified checkpoint must reproduce roughly the
        # published held-out accuracy.  A concordance index near chance here
        # means predictions and affinities are misaligned, not that the model
        # is bad, and that mistake is otherwise easy to report as a result.
        if name == "intact" and results[name]["concordance_index"] < 0.75:
            raise SystemExit(
                f"intact checkpoint scored CI {results[name]['concordance_index']:.4f} "
                f"on held-out KIBA pairs, far below the expected ~0.86; "
                f"predictions are probably misaligned with affinities"
            )

    # Paired bootstrap on the shared pairs: the two rankers see identical data,
    # so their errors are correlated and marginal intervals would understate the
    # resolution of the difference.
    base = preds["intact"]
    for name in ("cross_attention_off", "cross_attention_half"):
        other = preds[name]
        d_ci, d_rmse = [], []
        n = len(y)
        brng = np.random.default_rng(SEED)
        for _ in range(N_BOOT):
            idx = brng.integers(0, n, n)
            yb = y[idx]
            d_rmse.append(np.sqrt(np.mean((yb - base[idx]) ** 2))
                          - np.sqrt(np.mean((yb - other[idx]) ** 2)))
        lo, hi = np.percentile(d_rmse, [2.5, 97.5])
        results[f"paired_{name}"] = {
            "delta_rmse_intact_minus_ablated": float(np.mean(d_rmse)),
            "delta_rmse_ci95": [float(lo), float(hi)],
            "delta_ci_intact_minus_ablated":
                results["intact"]["concordance_index"] - results[name]["concordance_index"],
            "delta_pearson_intact_minus_ablated":
                results["intact"]["pearson_r"] - results[name]["pearson_r"],
        }

    out = pd.DataFrame({"affinity": y})
    for name, p in preds.items():
        out[f"pred_{name}"] = p
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    print("\n--- learned cross-attention gates ---")
    print(json.dumps(results["learned_gates"], indent=2))
    print("\n--- inference-time ablation ---")
    for name in ("intact", "cross_attention_half", "cross_attention_off"):
        m = results[name]
        print(f"  {name:22s} CI {m['concordance_index']:.4f}  RMSE {m['rmse']:.4f}  "
              f"Pearson {m['pearson_r']:.4f}")
    for name in ("cross_attention_half", "cross_attention_off"):
        d = results[f"paired_{name}"]
        lo, hi = d["delta_rmse_ci95"]
        print(f"  intact minus {name}: dCI {d['delta_ci_intact_minus_ablated']:+.4f}, "
              f"dRMSE {d['delta_rmse_intact_minus_ablated']:+.4f} "
              f"(95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"\nwrote {OUT_JSON.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
