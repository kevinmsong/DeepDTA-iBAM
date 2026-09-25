"""Count body words in a LaTeX manuscript, excluding floats and back matter.

Used to track the manuscript against its length target.  Captions, tables,
figures, the bibliography and the declarations are excluded, because the target
applies to the running text a reader actually reads.

Usage:  python analysis/wordcount_tex.py path/to/main.tex [more.tex ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FLOAT_ENVS = ["figure", "figure*", "table", "table*", "smalltable",
              "abstract", "IEEEkeywords", "thebibliography"]


def body_words(tex: str) -> tuple[int, int]:
    total_all = len(_strip(tex).split())

    # Restrict to the running text: from the first \section to the bibliography
    # or the declarations, whichever comes first.
    start = tex.find(r"\section")
    if start < 0:
        start = 0
    ends = [tex.find(m) for m in (r"\bibliography{", r"\section*{Declarations}",
                                  r"\begin{thebibliography}", r"\end{document}")]
    ends = [e for e in ends if e > start]
    end = min(ends) if ends else len(tex)
    body = tex[start:end]

    for env in FLOAT_ENVS:
        esc = re.escape(env)
        body = re.sub(r"\\begin\{" + esc + r"\}.*?\\end\{" + esc + r"\}",
                      " ", body, flags=re.S)
    return len(_strip(body).split()), total_all


def _strip(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", " ", s)              # comments
    s = re.sub(r"\$[^$]*\$", " x ", s)             # inline math -> one token
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)         # control sequences
    s = re.sub(r"[{}\[\]~&\\]", " ", s)            # braces and specials
    return s


def abstract_words(tex: str) -> int | None:
    """Word count of the abstract, which IEEE Access caps at 250."""
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        return None
    a = m.group(1).replace(r"\%", "%").replace("--", "-")
    return len([w for w in _strip(a).split() if re.search(r"[A-Za-z0-9]", w)])


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    for p in paths:
        if not p.exists():
            print(f"{p}: missing")
            continue
        tex = p.read_text(encoding="utf-8", errors="replace")
        body, whole = body_words(tex)
        print(f"{p.name}: body {body:,} words (whole file {whole:,})")
        n = abstract_words(tex)
        if n is not None:
            flag = "ok" if n <= 250 else "OVER the IEEE Access 250-word limit"
            print(f"{' ' * len(p.name)}  abstract {n} words ({flag})")


if __name__ == "__main__":
    main()
