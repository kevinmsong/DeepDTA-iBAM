"""Per-section body word counts for a LaTeX manuscript.

Shows where the length budget is actually going, so trimming targets the long
sections rather than the visible ones.  Floats are excluded, matching
wordcount_tex.py.

Usage:  python analysis/section_wordcount.py path/to/main.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FLOAT_ENVS = ["figure", "figure*", "table", "table*", "abstract", "IEEEkeywords"]


def strip(s: str) -> str:
    for env in FLOAT_ENVS:
        esc = re.escape(env)
        s = re.sub(r"\\begin\{" + esc + r"\}.*?\\end\{" + esc + r"\}", " ", s, flags=re.S)
    s = re.sub(r"(?<!\\)%.*", " ", s)
    s = re.sub(r"\$[^$]*\$", " x ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}\[\]~&\\]", " ", s)
    return s


def main() -> None:
    path = Path(sys.argv[1])
    tex = path.read_text(encoding="utf-8", errors="replace")
    start = tex.find(r"\section")
    ends = [tex.find(m) for m in (r"\bibliography{", r"\section*{Acknowledgment}")]
    ends = [e for e in ends if e > start]
    body = tex[start:min(ends) if ends else len(tex)]

    pat = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}")
    marks = [(m.start(), m.group(1), m.group(2)) for m in pat.finditer(body)]
    marks.append((len(body), "end", ""))

    total = 0
    rows = []
    for i in range(len(marks) - 1):
        pos, kind, name = marks[i]
        chunk = body[pos:marks[i + 1][0]]
        n = len(strip(chunk).split())
        indent = {"section": "", "subsection": "  ", "subsubsection": "    "}[kind]
        rows.append((indent, name, n, kind))
        if kind == "section":
            total += 0
    # section totals
    sec_tot = {}
    cur = None
    for indent, name, n, kind in rows:
        if kind == "section":
            cur = name
            sec_tot[cur] = sec_tot.get(cur, 0)
        if cur:
            sec_tot[cur] = sec_tot.get(cur, 0) + n

    for indent, name, n, kind in rows:
        label = f"{indent}{name}"
        if kind == "section":
            print(f"{label:<52s} {sec_tot[name]:>6,}  (section total)")
        else:
            print(f"{label:<52s} {n:>6,}")
    print("-" * 62)
    print(f"{'TOTAL BODY':<52s} {sum(sec_tot.values()):>6,}")


if __name__ == "__main__":
    main()
