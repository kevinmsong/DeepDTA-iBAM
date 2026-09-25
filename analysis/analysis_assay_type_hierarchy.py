"""Sensitivity of the EGFR panel to the activity-record consolidation rule.

A reviewer objected that grouping Ki, Kd and IC50 under one label conflates a
relation operator with a measurement type, and that prior work consolidates
heterogeneous activity data by a measurement-type priority (Kd preferred over
Ki, Ki over IC50) rather than treating the three as interchangeable
(Valsson et al., Commun. Chem. 8:41, 2025, doi 10.1038/s42004-025-01428-y).

The panel as built keeps, for each parent molecule, the single record with the
highest pChEMBL value, ties broken by the lowest standard value in nM, with no
regard to measurement type.  This script re-derives the panel under the
type-priority rule and reports what changes.

Raw activity records are cached to results/egfr_activity_records_raw.csv on the
first successful fetch, so later runs need no network.  If ChEMBL is
unreachable and no cache exists, the script reports the composition that can be
established from the released panel files and exits without the counterfactual.

Run from anywhere:  python analysis/analysis_assay_type_hierarchy.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW_CACHE = RESULTS / "egfr_activity_records_raw.csv"
FAMILY_CSV = RESULTS / "egfr_interpolation_family.csv"
H1_CSV = RESULTS / "h1_active_library.csv"
OUT_JSON = RESULTS / "assay_type_hierarchy_summary.json"

TARGET = "CHEMBL203"
PAGE_LIMIT = 1000
MAX_PAGES = 25
# Panel thresholds, matching case_studies_results_generation.py
MIN_PCHEMBL = 7.0
MIN_SIMILARITY = 0.40
# Measurement-type priority: equilibrium binding constants before the
# condition-dependent functional readout.
TYPE_PRIORITY = {"KD": 0, "KI": 1, "IC50": 2}

FIELDS = ["molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles",
          "standard_type", "standard_relation", "standard_value",
          "standard_units", "pchembl_value", "assay_chembl_id"]


def _url(offset: int) -> str:
    return (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?target_chembl_id={TARGET}"
        "&assay_type=B"
        "&standard_type__in=Ki,IC50,Kd"
        "&standard_relation=%3D"
        "&pchembl_value__isnull=false"
        "&canonical_smiles__isnull=false"
        "&order_by=-pchembl_value"
        f"&limit={PAGE_LIMIT}&offset={offset}"
    )


def fetch_records() -> pd.DataFrame | None:
    rows: list[dict] = []
    for page in range(MAX_PAGES):
        url = _url(page * PAGE_LIMIT)
        try:
            with urllib.request.urlopen(url, timeout=90) as fh:
                payload = json.load(fh)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  fetch failed on page {page}: {exc}")
            return pd.DataFrame(rows) if rows else None
        acts = payload.get("activities", [])
        if not acts:
            break
        for a in acts:
            rows.append({k: a.get(k) for k in FIELDS})
        print(f"  page {page}: {len(acts)} records (running total {len(rows)})", flush=True)
        if len(acts) < PAGE_LIMIT:
            break
        time.sleep(0.4)
    return pd.DataFrame(rows) if rows else None


def load_records() -> pd.DataFrame | None:
    if RAW_CACHE.exists():
        print(f"using cached records: {RAW_CACHE.name}")
        return pd.read_csv(RAW_CACHE)
    print("fetching activity records from ChEMBL ...")
    df = fetch_records()
    if df is not None and len(df):
        df.to_csv(RAW_CACHE, index=False)
        print(f"cached {len(df)} records to {RAW_CACHE.name}")
    return df


def _consolidate_best_pchembl(df: pd.DataFrame) -> pd.DataFrame:
    """The rule the panel was built with: highest pChEMBL, ties by lowest nM."""
    return (df.sort_values(by=["key", "pchembl_value", "standard_value"],
                           ascending=[True, False, True])
              .drop_duplicates(subset=["key"])
              .reset_index(drop=True))


def _consolidate_type_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Kd before Ki before IC50; within the winning type, highest pChEMBL."""
    return (df.sort_values(by=["key", "type_rank", "pchembl_value", "standard_value"],
                           ascending=[True, True, False, True])
              .drop_duplicates(subset=["key"])
              .reset_index(drop=True))


def local_composition() -> dict:
    out = {}
    if FAMILY_CSV.exists():
        fam = pd.read_csv(FAMILY_CSV)
        out["family_size"] = int(len(fam))
        out["family_type_counts"] = {
            str(k): int(v) for k, v in fam["activity_type"].value_counts().items()
        }
    if H1_CSV.exists():
        h1 = pd.read_csv(H1_CSV)
        multi = h1["activity_types"].astype(str).str.contains(";")
        out["h1_size"] = int(len(h1))
        out["h1_multi_type_molecules"] = int(multi.sum())
        out["h1_type_strings"] = {
            str(k): int(v) for k, v in h1["activity_types"].value_counts().items()
        }
    return out


def main() -> None:
    summary = {
        "target": TARGET,
        "type_priority": "Kd > Ki > IC50",
        "current_rule": "highest pChEMBL per parent molecule, ties by lowest nM",
        "min_pchembl": MIN_PCHEMBL,
        "min_reference_similarity": MIN_SIMILARITY,
        "local_composition": local_composition(),
    }

    df = load_records()
    if df is None or not len(df):
        summary["status"] = "no_records"
        summary["note"] = (
            "ChEMBL activity endpoint unreachable and no local cache present; "
            "the counterfactual consolidation was not computed."
        )
        OUT_JSON.write_text(json.dumps(summary, indent=2))
        print("\nChEMBL unreachable and no cache. Wrote composition-only summary.")
        print(json.dumps(summary["local_composition"], indent=2))
        return

    df = df.dropna(subset=["pchembl_value"]).copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"])
    df["key"] = df["parent_molecule_chembl_id"].fillna(df["molecule_chembl_id"])
    df["type_upper"] = df["standard_type"].astype(str).str.upper()
    df["type_rank"] = df["type_upper"].map(TYPE_PRIORITY).fillna(99).astype(int)

    summary["n_records"] = int(len(df))
    summary["n_pairs"] = int(df["key"].nunique())
    summary["record_type_counts"] = {
        str(k): int(v) for k, v in df["type_upper"].value_counts().items()
    }

    per_pair_types = df.groupby("key")["type_upper"].nunique()
    summary["n_pairs_multi_type"] = int((per_pair_types > 1).sum())
    summary["frac_pairs_multi_type"] = float((per_pair_types > 1).mean())

    cur = _consolidate_best_pchembl(df).set_index("key")
    pri = _consolidate_type_priority(df).set_index("key")
    shared = cur.index.intersection(pri.index)

    changed_type = (cur.loc[shared, "type_upper"] != pri.loc[shared, "type_upper"])
    delta = (pri.loc[shared, "pchembl_value"] - cur.loc[shared, "pchembl_value"])
    summary["n_pairs_type_changed"] = int(changed_type.sum())
    summary["n_pairs_value_changed"] = int((delta.abs() > 1e-9).sum())
    summary["median_abs_pchembl_shift_when_changed"] = (
        float(delta[delta.abs() > 1e-9].abs().median()) if (delta.abs() > 1e-9).any() else 0.0
    )
    summary["max_abs_pchembl_shift"] = float(delta.abs().max())

    # Membership of the potency-gated set under each rule.
    cur_set = set(cur.index[cur["pchembl_value"] >= MIN_PCHEMBL])
    pri_set = set(pri.index[pri["pchembl_value"] >= MIN_PCHEMBL])
    summary["n_potent_current_rule"] = len(cur_set)
    summary["n_potent_priority_rule"] = len(pri_set)
    summary["n_gained_under_priority"] = len(pri_set - cur_set)
    summary["n_lost_under_priority"] = len(cur_set - pri_set)
    summary["jaccard_potent_sets"] = (
        len(cur_set & pri_set) / len(cur_set | pri_set) if (cur_set | pri_set) else 1.0
    )
    summary["status"] = "ok"

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print("\n--- consolidation sensitivity ---")
    for k in ("n_records", "n_pairs", "record_type_counts", "n_pairs_multi_type",
              "n_pairs_type_changed", "n_pairs_value_changed",
              "max_abs_pchembl_shift", "n_potent_current_rule",
              "n_potent_priority_rule", "n_gained_under_priority",
              "n_lost_under_priority", "jaccard_potent_sets"):
        print(f"  {k:34s} {summary[k]}")
    print(f"wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
