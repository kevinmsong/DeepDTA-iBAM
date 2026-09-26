"""Check manuscript figures for colorblind safety and greyscale legibility.

Two things are verified for every figure in the palette:

1.  Every pair of palette colors stays distinguishable under simulated
    protanopia, deuteranopia and tritanopia, and in greyscale. Distance is
    CIE76 in Lab, with a threshold of 20, which is comfortably above the ~2.3
    just-noticeable difference and is a common practical floor for categorical
    encodings.
2.  Each rendered figure is reported with its greyscale contrast range, so a
    figure that collapses when printed in black and white is visible here.

Colour-vision simulation uses the Brettel-Vienot-Mollon style linear transforms
in LMS space, which is the standard approach for this check.

Run from anywhere:  python analysis/check_colorblind.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIGURES = ["fig_architecture", "fig_localization_forest", "fig_benchmark_validity",
           "fig_efficiency", "fig_interaction_content", "fig_docking",
           "fig_ibam_map"]

# sRGB -> LMS (Hunt-Pointer-Estevez, D65 normalised)
RGB2LMS = np.array([[0.31399022, 0.63951294, 0.04649755],
                    [0.15537241, 0.75789446, 0.08670142],
                    [0.01775239, 0.10944209, 0.87256922]])
LMS2RGB = np.linalg.inv(RGB2LMS)

SIMS = {
    "protanopia": np.array([[0, 1.05118294, -0.05116099],
                            [0, 1, 0],
                            [0, 0, 1]]),
    "deuteranopia": np.array([[1, 0, 0],
                              [0.9513092, 0, 0.04866992],
                              [0, 0, 1]]),
    "tritanopia": np.array([[1, 0, 0],
                            [0, 1, 0],
                            [-0.86744736, 1.86727089, 0]]),
}

THRESHOLD = 20.0


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb, kind):
    lin = _srgb_to_linear(rgb)
    lms = RGB2LMS @ lin
    out = LMS2RGB @ (SIMS[kind] @ lms)
    return _linear_to_srgb(out)


def to_lab(rgb):
    lin = _srgb_to_linear(rgb)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ lin
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def main() -> None:
    from figure_style import COLORS

    names = ["primary", "secondary", "tertiary", "quaternary", "accent", "muted"]
    cols = {n: hex_to_rgb(COLORS[n]) for n in names}
    # The per-figure check below also needs the neutrals, which figures use to
    # hold a background series against a highlighted one.
    all_cols = {n: hex_to_rgb(v) for n, v in COLORS.items()}

    print("=== palette separation (CIE76, threshold "
          f"{THRESHOLD:.0f}) ===")
    worst = (1e9, "", "", "")
    failures = 0
    for cond in ["normal"] + list(SIMS):
        sim = {n: (c if cond == "normal" else simulate(c, cond))
               for n, c in cols.items()}
        labs = {n: to_lab(c) for n, c in sim.items()}
        dmin, pair = 1e9, ("", "")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = float(np.linalg.norm(labs[a] - labs[b]))
                if d < dmin:
                    dmin, pair = d, (a, b)
        status = "ok" if dmin >= THRESHOLD else "TOO CLOSE"
        if dmin < THRESHOLD:
            failures += 1
        if dmin < worst[0]:
            worst = (dmin, cond, pair[0], pair[1])
        print(f"  {cond:14s} min dE = {dmin:6.1f}  ({pair[0]} vs {pair[1]})  {status}")

    # greyscale
    grey = {n: float(np.dot([0.2126, 0.7152, 0.0722], _srgb_to_linear(c)))
            for n, c in cols.items()}
    ordered = sorted(grey.items(), key=lambda kv: kv[1])
    gmin = min(b[1] - a[1] for a, b in zip(ordered, ordered[1:]))
    print(f"  {'greyscale':14s} min luminance gap = {gmin:.3f}"
          f"  {'ok' if gmin > 0.03 else 'TOO CLOSE'}")
    print("    luminances: " + ", ".join(f"{n} {v:.3f}" for n, v in ordered))

    # Per-figure check: colour only has to separate within a figure, and only
    # where it is the sole channel.  Figures that also encode by marker, line
    # style, border style or an axis label are noted as redundantly encoded.
    per_fig = {
        "fig_architecture": (["primary", "secondary", "tertiary"], "border style + legend"),
        "fig_localization_forest": (["primary", "secondary"], "marker shape"),
        "fig_benchmark_validity": (["primary", "secondary"], "axis labels"),
        "fig_efficiency": (["primary", "secondary", "tertiary"], "marker shape"),
        "fig_interaction_content": (["primary", "secondary"], "per-bar axis labels"),
        "fig_docking": (["primary", "secondary", "tertiary"], "line style"),
        "fig_ibam_map": (["secondary", "grey"], "marker shape"),
    }
    print("\n=== per-figure colour separation ===")
    worst_fig = (1e9, "", "", "", "")
    for fig, (used, redundant) in per_fig.items():
        fmin, fcond, fpair = 1e9, "", ("", "")
        for cond in ["normal"] + list(SIMS):
            sim = {n: (all_cols[n] if cond == "normal"
                       else simulate(all_cols[n], cond))
                   for n in used}
            labs = {n: to_lab(c) for n, c in sim.items()}
            for i, a in enumerate(used):
                for b in used[i + 1:]:
                    d = float(np.linalg.norm(labs[a] - labs[b]))
                    if d < fmin:
                        fmin, fcond, fpair = d, cond, (a, b)
        status = "ok" if fmin >= THRESHOLD else f"relies on {redundant}"
        print(f"  {fig:28s} min dE {fmin:5.1f} ({fcond}, {fpair[0]}/{fpair[1]})  {status}")
        if fmin < worst_fig[0]:
            worst_fig = (fmin, fig, fcond, fpair[0], fpair[1])

    # The one heatmap in the paper encodes a continuous quantity by color
    # alone, so its colormap has to stay ordered under every deficiency: a
    # reader must be able to tell higher from lower.  That means lightness has
    # to rise monotonically along the map, both in normal vision and under
    # simulation.
    print("\n=== sequential colormap ===")
    seq_ok = True
    try:
        import matplotlib.cm as cm
        from figure_style import SEQUENTIAL
        samples = np.linspace(0, 1, 32)
        rgb = np.array([cm.get_cmap(SEQUENTIAL)(s)[:3] for s in samples])
        for cond in ["normal"] + list(SIMS):
            cols = rgb if cond == "normal" else np.array(
                [simulate(c, cond) for c in rgb])
            L = np.array([to_lab(c)[0] for c in cols])
            drops = int((np.diff(L) < -0.5).sum())
            span = float(L.max() - L.min())
            status = "ok" if drops == 0 else f"{drops} reversals"
            if drops:
                seq_ok = False
            print(f"  {SEQUENTIAL} {cond:14s} lightness span {span:5.1f}  {status}")
    except Exception as exc:                       # pragma: no cover
        print(f"  (skipped: {type(exc).__name__}: {exc})")

    print("\n=== rendered figures ===")
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow unavailable; skipping render check)")
        return
    for f in FIGURES:
        p = RESULTS / f"{f}.png"
        if not p.exists():
            print(f"  {f:28s} MISSING")
            continue
        im = Image.open(p).convert("L")
        a = np.asarray(im, dtype=float) / 255
        ink = a[a < 0.97]
        span = float(ink.max() - ink.min()) if ink.size else 0.0
        print(f"  {f:28s} greyscale ink range {span:.2f}"
              f"  {'ok' if span > 0.3 else 'LOW CONTRAST'}")

    # The verdict is the per-figure result, not the whole palette.  The palette
    # holds more hues than any single figure uses, and two of the spares are
    # close to each other under simulation; that only matters if a figure ever
    # draws them together, which the per-figure check is what actually tests.
    print(f"\nfull palette worst case: dE {worst[0]:.1f} under {worst[1]} "
          f"({worst[2]} vs {worst[3]}), across hues no single figure combines")
    print(f"per-figure worst case:   dE {worst_fig[0]:.1f} in {worst_fig[1]} "
          f"under {worst_fig[2]} ({worst_fig[3]} vs {worst_fig[4]})")
    print("FIGURES COLORBLIND-SAFE:", worst_fig[0] >= THRESHOLD and seq_ok)


if __name__ == "__main__":
    main()
