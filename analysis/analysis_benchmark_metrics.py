"""Expanded predictive-benchmark metrics for the KIBA standard test partition.

Adds to the regression metrics already reported: rank correlation, and binary
classification metrics at the conventional KIBA activity threshold, each
reported beside the positive-class base rate.  Reporting the base rate beside
every enrichment metric is the study's own recommended practice, applied here
to the study's own headline benchmark.

Consumes results/max_rmse_cluster_diffusion_member1_standard_predictions.csv
(target, prediction, residual) so no model inference is required.

Run from anywhere:  python analysis/analysis_benchmark_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "max_rmse_cluster_diffusion_member1_standard_predictions.csv"
OUT_JSON = ROOT / "results" / "benchmark_metrics_expanded.json"
OUT_TEX = ROOT / "results" / "table_benchmark_metrics_expanded.tex"

# Conventional KIBA binarization threshold used to define active pairs.
KIBA_ACTIVE_THRESHOLD = 12.1
N_BOOT = 2000
SEED = 1337


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of comparable pairs ranked correctly, ties counted as half.

    Pairs with equal true values are not comparable and are excluded; pairs
    with equal predictions count as half a concordance.  This is the standard
    convention, and the one the DeepDTA-lineage reference implementations use,
    so the value is comparable with the literature rows in the benchmark table.
    An earlier in-house evaluation scored prediction ties as zero instead,
    which is why that artifact reports 0.8621 where this reports 0.8625.
    """
    order = np.argsort(y_true)
    y_true, y_pred = y_true[order], y_pred[order]
    n = len(y_true)
    total = 0.0
    hits = 0.0
    for i in range(n):
        greater = y_true[i + 1:] > y_true[i]
        if not greater.any():
            continue
        diff = y_pred[i + 1:][greater] - y_pred[i]
        total += diff.size
        hits += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
    return hits / total if total else float("nan")


def _boot_ci(fn, *arrays, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    df = pd.read_csv(PRED)
    y = df["target"].to_numpy(float)
    p = df["prediction"].to_numpy(float)

    res = {
        "n_pairs": int(len(df)),
        "kiba_active_threshold": KIBA_ACTIVE_THRESHOLD,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }

    # --- regression -------------------------------------------------------
    res["rmse"] = float(np.sqrt(np.mean((y - p) ** 2)))
    res["mae"] = float(np.mean(np.abs(y - p)))
    res["pearson_r"] = float(stats.pearsonr(y, p)[0])
    res["spearman_rho"] = float(stats.spearmanr(y, p).correlation)
    res["r2"] = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    res["concordance_index"] = concordance_index(y, p)
    res["rmse_ci"] = _boot_ci(lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))), y, p)
    res["spearman_ci"] = _boot_ci(lambda a, b: float(stats.spearmanr(a, b).correlation), y, p)
    res["concordance_index_ci"] = _boot_ci(concordance_index, y, p)

    # --- classification at the activity threshold -------------------------
    lab = (y >= KIBA_ACTIVE_THRESHOLD).astype(int)
    pred_lab = (p >= KIBA_ACTIVE_THRESHOLD).astype(int)
    res["n_active"] = int(lab.sum())
    res["base_rate"] = float(lab.mean())
    res["auroc"] = float(roc_auc_score(lab, p))
    res["auprc"] = float(average_precision_score(lab, p))
    res["auprc_over_base_rate"] = res["auprc"] / res["base_rate"]
    res["balanced_accuracy"] = float(balanced_accuracy_score(lab, pred_lab))
    res["mcc"] = float(matthews_corrcoef(lab, pred_lab))
    res["auroc_ci"] = _boot_ci(lambda a, b: float(roc_auc_score(a, b)), lab, p)
    res["auprc_ci"] = _boot_ci(lambda a, b: float(average_precision_score(a, b)), lab, p)

    OUT_JSON.write_text(json.dumps(res, indent=2))

    # Grouped so a reader can see at a glance which rows are regression on the
    # continuous score and which are classification at the activity threshold,
    # and so the positive base rate sits beside the metrics it qualifies.
    regression = [
        ("RMSE", f"{res['rmse']:.4f}", f"[{res['rmse_ci'][0]:.4f}, {res['rmse_ci'][1]:.4f}]"),
        ("MAE", f"{res['mae']:.4f}", "--"),
        (r"Pearson $r$", f"{res['pearson_r']:.4f}", "--"),
        (r"Spearman $\rho$", f"{res['spearman_rho']:.4f}",
         f"[{res['spearman_ci'][0]:.4f}, {res['spearman_ci'][1]:.4f}]"),
        (r"$R^2$", f"{res['r2']:.4f}", "--"),
        ("Concordance index", f"{res['concordance_index']:.4f}",
         f"[{res['concordance_index_ci'][0]:.4f}, "
         f"{res['concordance_index_ci'][1]:.4f}]"),
    ]
    classification = [
        ("Positive base rate", f"{res['base_rate']:.4f}", "--"),
        ("AUROC", f"{res['auroc']:.4f}",
         f"[{res['auroc_ci'][0]:.4f}, {res['auroc_ci'][1]:.4f}]"),
        ("AUPRC", f"{res['auprc']:.4f}",
         f"[{res['auprc_ci'][0]:.4f}, {res['auprc_ci'][1]:.4f}]"),
        ("Balanced accuracy", f"{res['balanced_accuracy']:.4f}", "--"),
        ("Matthews correlation", f"{res['mcc']:.4f}", "--"),
    ]
    nl = "\\\\"
    thr = res["kiba_active_threshold"]
    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           r"Metric & Value & 95\% CI " + nl, r"\midrule",
           r"\multicolumn{3}{l}{\textit{Regression on the KIBA score}} " + nl]
    tex += [f"\\quad {n} & {v} & {c} " + nl for n, v, c in regression]
    tex += [r"\addlinespace",
            r"\multicolumn{3}{l}{\textit{Classification at KIBA $\geq$ "
            + f"{thr}" + r"}} " + nl]
    tex += [f"\\quad {n} & {v} & {c} " + nl for n, v, c in classification]
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    OUT_TEX.write_text("\n".join(tex))

    print(f"pairs {res['n_pairs']}, active base rate {res['base_rate']:.4f} "
          f"({res['n_active']} of {res['n_pairs']}) at KIBA >= {KIBA_ACTIVE_THRESHOLD}")
    for k in ("rmse", "mae", "pearson_r", "spearman_rho", "r2", "concordance_index",
              "auroc", "auprc", "auprc_over_base_rate", "balanced_accuracy", "mcc"):
        print(f"  {k:24s} {res[k]:.4f}")
    print(f"wrote {OUT_JSON.name}, {OUT_TEX.name}")


if __name__ == "__main__":
    main()
