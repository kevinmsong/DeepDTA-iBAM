"""
Typeset the manuscript's principal tables to a consistent house style.

The pipeline emits tables with mixed precision and left-aligned numeric columns.
This module regenerates the three most-read tables from the same source CSVs with
decimal-aligned siunitx columns, uniform significant figures, explicit units, and
footnotes marking rows that are reported for transparency but are not valid
comparators.

Run from the submission directory:  python make_tables.py
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

RES = str(_ROOT / "results") + "/"


def _fmt(v, dp=3):
    try:
        return f"{float(v):.{dp}f}"
    except (TypeError, ValueError):
        return "{" + str(v) + "}"


# ---------------------------------------------------------------------------
def benchmark_table():
    rows = list(csv.DictReader(open(RES + "table1_benchmark.csv")))
    with open(_out("table1_benchmark_pro.tex"), "w") as fh:
        fh.write("\\begin{threeparttable}\n")
        fh.write("\\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] l}\n\\toprule\n")
        fh.write("Model & {CI} & {MSE} & Architecture \\\\\n\\midrule\n")
        for i, r in enumerate(rows):
            name = r["model"]
            note = r["notes"]
            if i == 0:
                name = f"\\textbf{{{name}}}\\tnote{{a}}"
                note = "Unified multimodal (this work)"
            fh.write(f"{name} & {_fmt(r['CI'])} & {_fmt(r['MSE'])} & {note} \\\\\n")
            if i == 0:
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        fh.write("\\begin{tablenotes}[flushleft]\\footnotesize\n")
        fh.write("\\item[a] Locally rerun single checkpoint. Remaining rows are primary-source "
                 "values for the same split, reproduced from the cited publications and not "
                 "rerun in the present environment; the table is contextual rather than a "
                 "controlled comparison.\n")
        fh.write("\\end{tablenotes}\n\\end{threeparttable}\n")


# ---------------------------------------------------------------------------
def retrieval_table():
    rows = list(csv.DictReader(open(RES + "table_egfr_retrieval_metrics.csv")))
    inadmissible = {"Nearest-anchor ECFP", "Anchor-centroid ECFP"}
    with open(_out("table_egfr_retrieval_pro.tex"), "w") as fh:
        fh.write("\\begin{threeparttable}\n")
        fh.write("\\begin{tabular}{l S[table-format=1.3] l S[table-format=1.3] "
                 "S[table-format=1.3] S[table-format=1.3]}\n\\toprule\n")
        fh.write("Method & {AUROC} & {95\\% CI} & {AUPRC} & {BEDROC20} & {Rec.@10\\%} \\\\\n")
        fh.write("\\midrule\n")
        fh.write("\\multicolumn{6}{l}{\\textit{Admissible comparators}} \\\\\n")
        wrote_sep = False
        for r in rows:
            m = r["Method"]
            if m in inadmissible and not wrote_sep:
                fh.write("\\midrule\n\\multicolumn{6}{l}{\\textit{Not valid comparators on this "
                         "panel\\tnote{a}}} \\\\\n")
                wrote_sep = True
            label = f"{m}\\tnote{{a}}" if m in inadmissible else m
            ci = f"{float(r['AUROC CI low']):.3f}--{float(r['AUROC CI high']):.3f}"
            fh.write(f"{label} & {_fmt(r['AUROC'])} & {ci} & {_fmt(r['AUPRC'])} & "
                     f"{_fmt(r['BEDROC20'])} & {_fmt(r['Recovery@10%'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        fh.write("\\begin{tablenotes}[flushleft]\\footnotesize\n")
        fh.write("\\item[a] The positive class was defined by an ECFP similarity cutoff to the "
                 "reference binders from which the anchors were drawn, so these rows measure "
                 "recovery of the selection rule. Intervals are marginal; head-to-head "
                 "differences are tested by paired bootstrap in the text.\n")
        fh.write("\\end{tablenotes}\n\\end{threeparttable}\n")


# ---------------------------------------------------------------------------
def generation_table():
    rows = list(csv.DictReader(open(RES + "table_generation_comparison_summary.csv")))
    with open(_out("table_generation_pro.tex"), "w") as fh:
        fh.write("\\begin{threeparttable}\n")
        fh.write("\\begin{tabular}{l S[table-format=3.0] S[table-format=1.3] "
                 "S[table-format=1.3] S[table-format=1.3] S[table-format=2.3] "
                 "S[table-format=1.3] S[table-format=1.3]}\n\\toprule\n")
        fh.write("Generator & {$n$} & {QED} & {SA} & {Tanimoto} & {Affinity} & "
                 "{Lipinski} & {Alert-free} \\\\\n\\midrule\n")
        pretty = {"diffusion": "Diffusion", "fragment_swap": "Fragment swap",
                  "random_edit": "Random atom edit", "seed_reference": "Dasatinib (start)"}
        for r in rows:
            gen = r["Generator"]
            gen = pretty.get(gen.strip().lower().replace(" ", "_"), gen.replace("_", " ").capitalize())
            fh.write(f"{gen} & {int(float(r['Unique valid analogs']))} & "
                     f"{_fmt(r['QED mean'])} & {_fmt(r['SA mean'])} & "
                     f"{_fmt(r['Tanimoto mean'])} & {_fmt(r['PredAffinity mean'])} & "
                     f"{_fmt(r['Lipinski pass rate'])} & {_fmt(r['Alert-free rate'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        fh.write("\\begin{tablenotes}[flushleft]\\footnotesize\n")
        fh.write("\\item $n$ is unique valid analogs recovered under a common budget of 4,000 "
                 "decode attempts. Tanimoto is similarity to the dasatinib starting structure. "
                 "Affinity is model-predicted and rescored by the same checkpoint that generated "
                 "the candidates, so it is not independent evidence of potency. Lipinski and "
                 "alert-free are pass fractions.\n")
        fh.write("\\end{tablenotes}\n\\end{threeparttable}\n")


if __name__ == "__main__":
    benchmark_table()
    retrieval_table()
    generation_table()
    print("wrote table1_benchmark_pro.tex, table_egfr_retrieval_pro.tex, table_generation_pro.tex")
