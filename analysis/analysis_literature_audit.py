"""
Structured audit of published drug-target affinity and compound-protein
interaction papers, testing whether the two reporting practices this study
recommends are already standard.

Scope and method
----------------
Papers were drawn from Europe PMC, restricted to open-access full texts so that
every verdict can be checked by a reader against the same source we used. The
sample spans 2018 to 2026 and mixes the canonical architectures with recent
work; it is a convenience sample of the open-access literature, not a
systematic review, and it is not a random draw from all published models.

Each paper was read for four questions:

  interp_claim   Does the paper validate attention or importance scores against
                 annotated binding-site or contact residues, in any form,
                 including a purely visual case study?
  quantitative   If so, is that validation quantitative (an overlap, enrichment,
                 accuracy, IoU or AUROC number) rather than a picture?
  base_rate      If quantitative, does the paper anywhere report the positive-
                 class base rate, that is, the fraction of residues that are
                 pocket or contact residues, alongside the metric?
  similarity_panel
                 Does the paper define a positive or active set using a chemical
                 similarity threshold or analog series AND also score a
                 fingerprint or similarity baseline on that same set?

The verdicts record what each paper reports in its full text as retrieved. A
paper that computes a base rate internally but does not print it is recorded as
not reporting it, because the check we propose is a reporting practice.

Run from the submission directory:  python analysis_literature_audit.py
"""

# --- path resolution -------------------------------------------------------
# Paths are anchored to this file, not the working directory, so the scripts
# run identically from a clean clone. LaTeX fragments are written to the
# manuscript directory when it is present (it is not part of this repository)
# and beside this script otherwise.
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_MS = _ROOT / "J_Cheminform_Submission"
OUT_DIR = _MS if _MS.is_dir() else _Path(__file__).resolve().parent


def _out(name):
    return str(OUT_DIR / name)
# ---------------------------------------------------------------------------

import csv
import json

SRC = str(_Path(__file__).resolve().parent / "literature_audit.csv")


def main():
    rows = list(csv.DictReader(open(SRC)))
    n = len(rows)

    claim = [r for r in rows if r["interp_claim"] == "Y"]
    quant = [r for r in claim if r["quantitative"] == "Y"]
    visual = [r for r in claim if r["quantitative"] == "N"]
    base_yes = [r for r in quant if r["base_rate"] == "Y"]
    base_no = [r for r in quant if r["base_rate"] == "N"]
    sim_yes = [r for r in rows if r["similarity_panel"] == "Y"]

    # ---------------------------------------------------------------- table
    with open(_out("table_literature_audit.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llrccc}\n\\toprule\n")
        fh.write("Model / study & Venue & Year & Residue claim & Quantitative "
                 "& Base rate \\\\\n\\midrule\n")
        mark = {"Y": "yes", "N": "no", "NA": "--"}
        for r in rows:
            fh.write(f"{r['short_name']} & {r['journal']} & {r['year']} & "
                     f"{mark[r['interp_claim']]} & {mark[r['quantitative']]} & "
                     f"{mark[r['base_rate']]} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{6}}{{l}}{{{len(claim)} of {n} validate "
                 f"attention against annotated residues; {len(quant)} of those "
                 f"do so quantitatively}} \\\\\n")
        fh.write(f"\\multicolumn{{6}}{{l}}{{Of the {len(quant)} quantitative "
                 f"evaluations, {len(base_yes)} report the positive-class base "
                 f"rate and {len(base_no)} do not}} \\\\\n")
        fh.write(f"\\multicolumn{{6}}{{l}}{{{len(sim_yes)} of {n} score a "
                 f"similarity baseline on a similarity-defined positive set}} "
                 f"\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    summary = dict(
        n=n,
        n_claim=len(claim), n_quantitative=len(quant), n_visual_only=len(visual),
        n_base_rate_reported=len(base_yes), n_base_rate_absent=len(base_no),
        n_similarity_panel=len(sim_yes),
        base_rate_reporting_fraction=round(len(base_yes) / len(quant), 3),
        base_rate_papers=[r["short_name"] for r in base_yes],
        visual_only_papers=[r["short_name"] for r in visual],
    )
    json.dump(summary, open("literature_audit_summary.json", "w"), indent=2)

    print("LITERATURE AUDIT")
    print(f"  papers audited                              {n}")
    print(f"  validate attention against annotated residues {len(claim)}")
    print(f"    of these, quantitatively                  {len(quant)}")
    print(f"    of these, visual case study only          {len(visual)}")
    print(f"  quantitative evaluations reporting base rate {len(base_yes)}"
          f" of {len(quant)}")
    print(f"    reporting:     {[r['short_name'] for r in base_yes]}")
    print(f"    not reporting: {[r['short_name'] for r in base_no]}")
    print(f"  similarity baseline on similarity-defined set {len(sim_yes)}"
          f" of {n}")


if __name__ == "__main__":
    main()
