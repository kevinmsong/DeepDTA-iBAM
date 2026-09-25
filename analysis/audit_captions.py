"""Audit figure and table captions in a LaTeX manuscript.

Reports every caption with its length and its opening phrase, so captions can be
checked for a self-contained opening statement and for lengths that are neither
a bare label nor a second results section.

Usage:  python analysis/audit_captions.py path/to/main.tex [more.tex ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_captions(tex: str):
    """Yield (kind, label, caption text) for each captioned float."""
    out = []
    i = 0
    while True:
        i = tex.find(r"\caption{", i)
        if i < 0:
            break
        j = i + len(r"\caption{")
        depth = 1
        while j < len(tex) and depth:
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
            j += 1
        body = tex[i + len(r"\caption{"): j - 1]
        tail = tex[j: j + 200]
        m = re.search(r"\\label\{(fig|tab):([^}]*)\}", tail)
        kind, lab = (m.group(1), m.group(2)) if m else ("?", "unlabeled")
        out.append((kind, lab, body))
        i = j
    return out


def words(body: str) -> int:
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    t = re.sub(r"[{}$~\\]", " ", t)
    return len(t.split())


def main() -> None:
    for path in [Path(a) for a in sys.argv[1:]]:
        tex = path.read_text(encoding="utf-8", errors="replace")
        caps = find_captions(tex)
        print(f"\n=== {path.name}: {len(caps)} captions ===")
        for kind, lab, body in caps:
            n = words(body)
            first = re.sub(r"\s+", " ", body).strip()
            flag = ""
            if n < 12:
                flag = "  <-- very short, may not stand alone"
            elif n > 130:
                flag = "  <-- long"
            print(f"[{kind}:{lab}] {n:3d}w{flag}")
            print(f"    {first[:150]}")


if __name__ == "__main__":
    main()
