"""Chance-referenced top-k contact overlap for the five-complex kinase panel.

The manuscript reports top-k overlap as an excess over the base-rate
expectation, but no script in the repository computed it.  This one does, from
the contact counts already released in results/interpretability_benchmark.csv,
so every overlap number in the text traces to code.

Two points of definition, both easy to misread:

1.  k is not 10.  In compute_structure_alignment_metrics, k is set to the
    number of true contacts in that complex, so the metric asks how many of
    the k highest-attention tokens are contacts when exactly k contacts exist.
2.  Under that definition, drawing k of n tokens uniformly at random gives an
    expected overlap of exactly k / n, which is the contact base rate.  The
    excess over chance is therefore observed overlap minus base rate.

For ligand atoms the base rate is at or near 1.0 in ATP-competitive complexes,
so the metric is saturated and the excess is pinned near zero by construction,
independent of the model.  That is the point, not a measurement.

Run from anywhere:  python analysis/analysis_overlap_chance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "interpretability_benchmark.csv"
OUT_CSV = ROOT / "results" / "overlap_chance_reference.csv"
OUT_JSON = ROOT / "results" / "overlap_chance_summary.json"


def _t_ci(x: np.ndarray, conf: float = 0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.5 + conf / 2.0, df=n - 1) * se
    return float(x.mean() - half), float(x.mean() + half)


def main() -> None:
    df = pd.read_csv(BENCH)

    df["residue_base_rate"] = df["n_contact_residues"] / df["n_residues"]
    df["atom_base_rate"] = df["n_contact_atoms"] / df["n_atoms"]
    df["residue_overlap_excess"] = df["residue_topk_overlap"] - df["residue_base_rate"]
    df["atom_overlap_excess"] = df["atom_topk_overlap"] - df["atom_base_rate"]
    # Enrichment relative to chance; 1.0 means indistinguishable from chance.
    df["residue_overlap_enrichment"] = df["residue_topk_overlap"] / df["residue_base_rate"]
    df["atom_overlap_enrichment"] = df["atom_topk_overlap"] / df["atom_base_rate"]
    df["atom_label_classes"] = np.where(
        df["n_contact_atoms"] == df["n_atoms"], 1, 2
    )

    cols = ["pdb_id", "protein", "ligand", "n_residues", "n_contact_residues",
            "residue_base_rate", "residue_topk_overlap", "residue_overlap_excess",
            "residue_overlap_enrichment", "residue_contact_auroc",
            "n_atoms", "n_contact_atoms", "atom_base_rate", "atom_topk_overlap",
            "atom_overlap_excess", "atom_overlap_enrichment", "atom_label_classes"]
    out = df[cols].copy()
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "n_complexes": int(len(df)),
        "note_k": "k equals the number of true contacts in each complex, not 10",
        "note_chance": "expected overlap for a random top-k draw equals k/n, the contact base rate",
        "atom_base_rate_min": float(df["atom_base_rate"].min()),
        "atom_base_rate_max": float(df["atom_base_rate"].max()),
        "n_complexes_atom_single_class": int((df["atom_label_classes"] == 1).sum()),
        "residue_base_rate_min": float(df["residue_base_rate"].min()),
        "residue_base_rate_max": float(df["residue_base_rate"].max()),
    }
    for level in ("residue", "atom"):
        obs = df[f"{level}_topk_overlap"].to_numpy(float)
        exc = df[f"{level}_overlap_excess"].to_numpy(float)
        base = df[f"{level}_base_rate"].to_numpy(float)
        lo, hi = _t_ci(exc)
        summary[f"{level}_overlap_mean"] = float(obs.mean())
        summary[f"{level}_base_rate_mean"] = float(base.mean())
        summary[f"{level}_excess_mean"] = float(exc.mean())
        summary[f"{level}_excess_ci95"] = [lo, hi]
        summary[f"{level}_excess_ci_excludes_zero"] = bool(lo > 0 or hi < 0)

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(out[["pdb_id", "protein", "n_atoms", "n_contact_atoms", "atom_base_rate",
               "n_residues", "n_contact_residues", "residue_base_rate",
               "residue_topk_overlap", "residue_overlap_excess"]].to_string(index=False))
    print()
    for level in ("residue", "atom"):
        lo, hi = summary[f"{level}_excess_ci95"]
        print(f"{level:8s} overlap {summary[f'{level}_overlap_mean']:.4f} "
              f"vs chance {summary[f'{level}_base_rate_mean']:.4f}, "
              f"excess {summary[f'{level}_excess_mean']:+.4f} "
              f"(95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"\natom contact base rate spans {summary['atom_base_rate_min']:.3f} to "
          f"{summary['atom_base_rate_max']:.3f}; "
          f"{summary['n_complexes_atom_single_class']} of {summary['n_complexes']} "
          f"complexes have a single-class atom label vector")
    print(f"wrote {OUT_CSV.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
