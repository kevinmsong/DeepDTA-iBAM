"""Rerun saved docking aggregation into an isolated audit directory.

No docking is performed. The source score table and panel are unchanged;
archived reports are hashed before and after the redirected aggregation.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import pandas as pd
import rdkit
import sklearn
from threadpoolctl import threadpool_limits

import analysis_docking_retrieval as aggregation

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "docking_revalidated"
TOLERANCE = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(old, new, prefix="") -> list[dict]:
    """Compare every numeric leaf, retaining exact values and differences."""
    rows = []
    if isinstance(old, dict):
        assert old.keys() == new.keys(), prefix
        for key in old:
            rows.extend(compare(old[key], new[key], f"{prefix}.{key}".strip(".")))
    elif isinstance(old, list):
        assert len(old) == len(new), prefix
        for i, (a, b) in enumerate(zip(old, new)):
            rows.extend(compare(a, b, f"{prefix}[{i}]"))
    elif isinstance(old, (int, float)):
        difference = float(new) - float(old)
        rows.append({"field": prefix, "archived": old, "revalidated": new,
                     "difference": difference,
                     "matches_within_tolerance": abs(difference) <= TOLERANCE})
    else:
        assert old == new, prefix
    return rows


def main() -> None:
    started = time.time()
    OUT.mkdir(exist_ok=True)
    protected = [aggregation.SCORES, aggregation.PANEL,
                 RES / "docking_retrieval_summary.json",
                 RES / "table_docking_retrieval.tex", RES / "docking_panel_scored.csv"]
    initial_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    source_hash = sha(Path(aggregation.__file__))
    old = json.loads((RES / "docking_retrieval_summary.json").read_text())
    # Retain original SCORES and PANEL paths; redirect all aggregation outputs.
    aggregation.RESULTS = OUT
    aggregation.OUT_JSON = OUT / "docking_retrieval_summary.json"
    aggregation.OUT_TEX = OUT / "table_docking_retrieval.tex"
    with threadpool_limits(limits=2):
        aggregation.main()
    new = json.loads(aggregation.OUT_JSON.read_text())

    original_panel = pd.read_csv(aggregation.PANEL)
    raw = pd.read_csv(aggregation.SCORES)
    panel_raw = raw.loc[raw["set"].eq("egfr_panel")]
    panel_keys = set(map(tuple, original_panel[["ident", "smiles"]].to_numpy()))
    attempted = set(map(tuple, panel_raw[["ident", "smiles"]].to_numpy()))
    successful = set(map(tuple, panel_raw.loc[panel_raw.status.eq("ok"), ["ident", "smiles"]].to_numpy()))
    assert len(panel_keys) == len(original_panel)
    assert successful <= attempted <= panel_keys
    assert len(attempted) == new["panel"]["n_attempted"] == 760
    assert len(successful) == new["panel"]["n_docked"] == 760
    assert len(attempted - successful) == new["panel"]["n_failed"] == 0
    assert len(panel_keys - successful) == new["panel"]["n_not_docked"] == 2540

    old_rows = pd.read_csv(RES / "docking_panel_scored.csv")
    new_rows = pd.read_csv(OUT / "docking_panel_scored.csv")
    assert list(old_rows.columns) == list(new_rows.columns)
    assert len(new_rows) == 760
    assert not new_rows[["ident", "smiles"]].duplicated().any()
    candidate_checks = {}
    for column in new_rows.columns:
        if column in ("ident", "smiles", "label"):
            assert old_rows[column].equals(new_rows[column]), column
            candidate_checks[column] = {"equal_values_and_order": True}
        else:
            delta = np.abs(old_rows[column].to_numpy() - new_rows[column].to_numpy())
            assert np.isfinite(delta).all() and delta.max() <= TOLERANCE, column
            candidate_checks[column] = {"max_absolute_difference": float(delta.max())}
    scientific_comparisons = []
    for section in ("metrics", "paired", "generated_analogs", "docking_cost"):
        scientific_comparisons.extend(compare(old[section], new[section], section))
    changes = [row for row in scientific_comparisons if not row["matches_within_tolerance"]]
    # Only rank-position metrics may change after removing arbitrary tie order.
    # AUROC, average precision, all paired intervals, timing and analog scores
    # must still reproduce the archived results.
    tie_metrics = (".bedroc20", ".ef_1pct", ".ef_5pct", ".recovery_10pct")
    assert all(row["field"].startswith("metrics.") and
               row["field"].endswith(tie_metrics) for row in changes)
    unchanged_panel = {k: v for k, v in old["panel"].items() if k != "n_failed"}
    assert all(new["panel"][k] == v for k, v in unchanged_panel.items())
    final_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert initial_hashes == final_hashes
    assert source_hash == sha(Path(aggregation.__file__))
    pd.DataFrame(scientific_comparisons).to_csv(OUT / "scientific_metric_comparison.csv", index=False)
    report = {
        "scope": "Aggregation of saved docking scores only; no docking or model inference.",
        "input_and_archived_output_sha256_before": initial_hashes,
        "input_and_archived_output_sha256_after": final_hashes,
        "source_script": str(Path(aggregation.__file__).relative_to(ROOT)),
        "source_script_sha256": source_hash,
        "audit_script_sha256": sha(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "rdkit": rdkit.__version__},
        "tolerance": TOLERANCE,
        "scientific_numeric_fields_compared": len(scientific_comparisons),
        "all_scientific_metrics_and_intervals_match": not changes,
        "all_auroc_average_precision_and_paired_intervals_match": True,
        "rank_metric_changes_from_explicit_tie_averaging": changes,
        "rank_tie_handling": new["rank_tie_handling"],
        "maximum_absolute_scientific_difference": max(abs(x["difference"]) for x in scientific_comparisons),
        "candidate_score_comparison": candidate_checks,
        "panel_count_correction": {"archived_n_failed": old["panel"]["n_failed"],
                                   "revalidated_n_failed": new["panel"]["n_failed"],
                                   "n_attempted": new["panel"]["n_attempted"],
                                   "n_not_docked": new["panel"]["n_not_docked"],
                                   "n_never_attempted": len(panel_keys - attempted)},
        "fingerprint_reference": "All 300 panel actives were docked, so within-subset and full-panel active reference sets coincide. The fingerprint ranker uses known active labels unavailable to the other rankers.",
        "outputs": ["docking_retrieval_summary.json", "table_docking_retrieval.tex",
                    "docking_panel_scored.csv", "scientific_metric_comparison.csv"],
        "seconds": round(time.time() - started, 3)}
    (OUT / "audit_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audit passed: {len(scientific_comparisons)} numerical fields, all candidate rows, and all protected hashes.")


if __name__ == "__main__":
    main()
