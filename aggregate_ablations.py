"""
Aggregate ablation results into publication-ready tables.

Reads results/ablations/raw_results.csv (written by run_ablations.py), or
falls back to scanning individual metrics.json files in the results/ablations/
tree.  Outputs:

  results/ablation_table.csv        — tidy: model_name, split_type, metric, mean, sd, n
  results/ablation_table_wide.csv   — wide: model_name, split_type, RMSE_mean, RMSE_sd, ...
  results/ablation_table.tex        — LaTeX booktabs table (mean±SD, best bolded)
  results/ablation_table.md         — Markdown table for quick review

Usage
-----
  python aggregate_ablations.py
  python aggregate_ablations.py --results-root results/ablations --output-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METRICS = ["RMSE", "MAE", "CI", "Pearson", "Spearman"]

# For each metric: True = higher is better, False = lower is better
METRIC_HIGHER_BETTER = {
    "RMSE": False,
    "MAE": False,
    "CI": True,
    "Pearson": True,
    "Spearman": True,
}

# Display order for model names (matches logical ablation progression)
MODEL_DISPLAY_ORDER = [
    "abl_base",
    "abl_no_fusion",
    "abl_no_ranking",
    "abl_no_diffusion",
    "abl_full",
]

MODEL_DISPLAY_NAMES = {
    "abl_base": "Base regressor",
    "abl_no_fusion": "No cross-attention",
    "abl_no_ranking": "No ranking loss",
    "abl_no_diffusion": "No diffusion",
    "abl_full": "Full model",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_results(results_root: Path) -> pd.DataFrame:
    """Load raw_results.csv or reconstruct from per-run metrics.json files."""
    csv_path = results_root / "raw_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        print(f"[load] {before} rows from {csv_path}")
        if removed:
            print(f"[load] removed {removed} exact duplicate rows before aggregation")
        return df

    print(f"[load] raw_results.csv not found — scanning metrics.json files under {results_root}")
    rows: List[Dict[str, Any]] = []
    for p in sorted(results_root.rglob("metrics.json")):
        with p.open() as fh:
            data = json.load(fh)
        if not all(k in data for k in ("model_name", "split_type", "seed", "RMSE")):
            continue
        row = {k: data[k] for k in ["model_name", "split_type", "seed"] + METRICS if k in data}
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No raw_results.csv and no metrics.json found under {results_root}. "
            "Run `python run_ablations.py` first."
        )
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[load] {before} rows reconstructed from metrics.json files")
    if removed:
        print(f"[load] removed {removed} exact duplicate rows before aggregation")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model_name, split_type), compute mean ± SD for each metric."""
    available = [m for m in METRICS if m in df.columns]
    records = []
    for (model, split), group in df.groupby(["model_name", "split_type"], sort=False):
        for metric in available:
            vals = group[metric].dropna()
            records.append({
                "model_name": model,
                "split_type": split,
                "metric": metric,
                "mean": vals.mean(),
                "sd": vals.std(ddof=1) if len(vals) > 1 else float("nan"),
                "n": len(vals),
            })
    return pd.DataFrame(records)


def pivot_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy aggregate to wide format with mean/sd columns per metric."""
    rows = []
    for (model, split), g in tidy.groupby(["model_name", "split_type"], sort=False):
        row: Dict[str, Any] = {"model_name": model, "split_type": split}
        for _, r in g.iterrows():
            row[f"{r['metric']}_mean"] = r["mean"]
            row[f"{r['metric']}_sd"] = r["sd"]
            row[f"{r['metric']}_n"] = int(r["n"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _fmt_cell(mean: float, sd: float, bold: bool) -> str:
    if pd.isna(mean):
        return "---"
    if pd.isna(sd):
        return f"$\\mathbf{{{mean:.3f}}}$" if bold else f"${mean:.3f}$"
    if bold:
        return f"$\\mathbf{{{mean:.3f}}}{{\\pm}}\\mathbf{{{sd:.3f}}}$"
    return f"${mean:.3f}{{\\pm}}{sd:.3f}$"


def make_latex_table(wide: pd.DataFrame, split_type: str) -> str:
    sub = wide[wide["split_type"] == split_type].copy()
    if sub.empty:
        return ""

    # Apply display order
    order = {name: i for i, name in enumerate(MODEL_DISPLAY_ORDER)}
    sub["_order"] = sub["model_name"].map(order).fillna(999)
    sub = sub.sort_values("_order").drop(columns="_order")

    available = [m for m in METRICS if f"{m}_mean" in sub.columns]

    # Identify best value per metric
    best_idx: Dict[str, int] = {}
    for metric in available:
        col = f"{metric}_mean"
        if METRIC_HIGHER_BETTER[metric]:
            best_idx[metric] = int(sub[col].idxmax())
        else:
            best_idx[metric] = int(sub[col].idxmin())

    header_metrics = " & ".join(
        f"{m} ${'\\uparrow' if METRIC_HIGHER_BETTER[m] else '\\downarrow'}$"
        for m in available
    )

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{\\textbf{{{split_type.capitalize()}-split ablation study.}} Mean $\\pm$ SD over {sub[f'{available[0]}_n'].max()} seeds on KIBA.}}",
        f"\\label{{tab:ablation_{split_type}}}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(available) + "}",
        "\\toprule",
        f"Model & {header_metrics} \\\\",
        "\\midrule",
    ]

    for idx, row in sub.iterrows():
        display_name = MODEL_DISPLAY_NAMES.get(row["model_name"], row["model_name"])
        cells = [display_name]
        for metric in available:
            bold = (idx == best_idx[metric])
            cells.append(_fmt_cell(row[f"{metric}_mean"], row.get(f"{metric}_sd", float("nan")), bold))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\endgroup",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def make_markdown_table(wide: pd.DataFrame, split_type: str) -> str:
    sub = wide[wide["split_type"] == split_type].copy()
    if sub.empty:
        return ""

    order = {name: i for i, name in enumerate(MODEL_DISPLAY_ORDER)}
    sub["_order"] = sub["model_name"].map(order).fillna(999)
    sub = sub.sort_values("_order").drop(columns="_order")

    available = [m for m in METRICS if f"{m}_mean" in sub.columns]
    header = "| Model | " + " | ".join(available) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(available)) + " |"

    rows_md = [header, sep]
    for _, row in sub.iterrows():
        display_name = MODEL_DISPLAY_NAMES.get(row["model_name"], row["model_name"])
        cells = [display_name]
        for metric in available:
            mean = row[f"{metric}_mean"]
            sd = row.get(f"{metric}_sd", float("nan"))
            if pd.isna(mean):
                cells.append("---")
            elif pd.isna(sd):
                cells.append(f"{mean:.3f}")
            else:
                cells.append(f"{mean:.3f}±{sd:.3f}")
        rows_md.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows_md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        default="results/ablations",
        help="Directory containing raw_results.csv or per-run metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where output tables are written.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_raw_results(results_root)

    # Ensure numeric
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    print(f"\n[data] {len(df)} total runs")
    counts = df.groupby(["model_name", "split_type"]).size().reset_index(name="n")
    print(counts.to_string(index=False))

    # Aggregate
    tidy = aggregate(df)
    wide = pivot_wide(tidy)

    # Write tidy CSV
    tidy_path = output_dir / "ablation_table.csv"
    tidy.to_csv(tidy_path, index=False, float_format="%.6f")
    print(f"\n[out]  {tidy_path}")

    # Write wide CSV
    wide_path = output_dir / "ablation_table_wide.csv"
    wide.to_csv(wide_path, index=False, float_format="%.6f")
    print(f"[out]  {wide_path}")

    # Write LaTeX (one table per split, combined into one file)
    tex_parts = ["% Ablation tables generated by aggregate_ablations.py\n"]
    for split in ["standard", "scaffold"]:
        tex = make_latex_table(wide, split)
        if tex:
            tex_parts.append(tex)
            tex_parts.append("")
    tex_path = output_dir / "ablation_table.tex"
    tex_path.write_text("\n".join(tex_parts), encoding="utf-8")
    print(f"[out]  {tex_path}")

    # Write Markdown
    md_parts = ["# Ablation Study — KIBA\n"]
    for split in ["standard", "scaffold"]:
        md = make_markdown_table(wide, split)
        if md:
            md_parts.append(f"## {split.capitalize()} split\n")
            md_parts.append(md)
            md_parts.append("")
    md_path = output_dir / "ablation_table.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"[out]  {md_path}")

    # Summary to stdout
    print("\n=== Quick summary (wide, scaffold split) ===")
    scaffold_wide = wide[wide["split_type"] == "scaffold"].copy()
    if not scaffold_wide.empty:
        order = {name: i for i, name in enumerate(MODEL_DISPLAY_ORDER)}
        scaffold_wide["_o"] = scaffold_wide["model_name"].map(order).fillna(999)
        scaffold_wide = scaffold_wide.sort_values("_o").drop(columns="_o")
        available = [m for m in METRICS if f"{m}_mean" in scaffold_wide.columns]
        cols = ["model_name"] + [f"{m}_mean" for m in available]
        print(scaffold_wide[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("  No scaffold results yet.")


if __name__ == "__main__":
    main()
