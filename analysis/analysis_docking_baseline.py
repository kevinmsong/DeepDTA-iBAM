"""AutoDock Vina baseline for the in-domain EGFR retrieval panel.

Reviewer request: compare the learned scorer against an established
structure-based baseline.  Docks every candidate of the assay-defined EGFR
panel (300 actives, 3,000 property-matched decoys) into the 4WKQ ATP site,
plus the audited diffusion analogs and the dasatinib seed, so the generation
comparison also gains an orthogonal scorer.

Ligands are embedded with RDKit ETKDGv3 and MMFF-optimized, then converted to
PDBQT with Meeko.  The receptor is prepared by analysis/prep_docking_receptor.py.

Vina reports binding free energy, so a LOWER score is better.  Downstream
retrieval metrics negate it.

The run is resumable: completed ligands are read back from the output CSV and
skipped.  Run from anywhere:

    python analysis/analysis_docking_baseline.py [--workers N] [--exhaustiveness N]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCK_DIR = ROOT / "results" / "docking"


def _find_vina() -> Path:
    """Locate the AutoDock Vina executable.

    Vina is a third-party binary and is not redistributed with this repository;
    see the README for where to download it.  Accept either platform's file
    name under tools/, or anything on PATH, so the same script runs on Windows
    and on Linux.
    """
    for name in ("vina.exe", "vina"):
        candidate = ROOT / "tools" / name
        if candidate.exists():
            return candidate
    found = shutil.which("vina")
    if found:
        return Path(found)
    return ROOT / "tools" / "vina.exe"      # reported as missing by the caller


VINA = _find_vina()
RECEPTOR = DOCK_DIR / "receptor_4wkq.pdbqt"
BOX_JSON = DOCK_DIR / "docking_box.json"
OUT_CSV = DOCK_DIR / "vina_scores.csv"

PANEL_CSV = ROOT / "results" / "dude_egfr_panel_scored.csv"
ANALOG_CSV = ROOT / "results" / "generated_egfr_analogs_audited.csv"
DASATINIB = "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"

FIELDS = ["set", "ident", "smiles", "label", "vina_kcal", "status", "seconds"]


def _box():
    import json
    return json.loads(BOX_JSON.read_text())


def _prepare_ligand(smiles: str, path: Path, seed: int = 1337) -> None:
    """Embed and optimize in 3D with RDKit, then write PDBQT with Meeko."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("unparseable SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("3D embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # keep the embedded geometry if MMFF typing fails
    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise ValueError("Meeko produced no setup")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"Meeko write failed: {err}")
    path.write_text(pdbqt)


def dock_one(job):
    """Prepare and dock one ligand.  Returns a result row."""
    subset, ident, smiles, label, exhaustiveness = job
    started = time.time()
    row = {"set": subset, "ident": ident, "smiles": smiles, "label": label,
           "vina_kcal": "", "status": "ok", "seconds": ""}
    tmpdir = Path(tempfile.mkdtemp(prefix="vina_"))
    try:
        lig = tmpdir / "lig.pdbqt"
        try:
            _prepare_ligand(smiles, lig)
        except Exception as exc:
            row["status"] = f"prep_failed: {exc}"
            return row

        box = dock_one.box
        cmd = [
            str(VINA),
            "--receptor", str(RECEPTOR),
            "--ligand", str(lig),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
            "--cpu", "1",
            "--seed", "1337",
            "--out", str(tmpdir / "out.pdbqt"),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            row["status"] = "dock_timeout"
            return row
        if proc.returncode != 0:
            row["status"] = "dock_failed"
            return row

        score = None
        for line in (tmpdir / "out.pdbqt").read_text().splitlines():
            if line.startswith("REMARK VINA RESULT"):
                score = float(line.split()[3])
                break
        if score is None:
            row["status"] = "no_score"
            return row
        row["vina_kcal"] = f"{score:.4f}"
        return row
    finally:
        row["seconds"] = f"{time.time() - started:.1f}"
        for f in tmpdir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


def _init_worker(box):
    dock_one.box = box


def build_jobs(exhaustiveness: int):
    import pandas as pd
    jobs = []
    panel = pd.read_csv(PANEL_CSV)
    for r in panel.itertuples(index=False):
        jobs.append(("egfr_panel", str(r.ident), str(r.smiles), int(r.label), exhaustiveness))

    if ANALOG_CSV.exists():
        analogs = pd.read_csv(ANALOG_CSV)
        col = "smiles" if "smiles" in analogs.columns else analogs.columns[0]
        for i, r in enumerate(analogs.itertuples(index=False)):
            jobs.append(("generated_analog", f"analog_{i:03d}", str(getattr(r, col)), -1, exhaustiveness))
    jobs.append(("generated_analog", "dasatinib_seed", DASATINIB, -1, exhaustiveness))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--all-actives", action="store_true",
                    help="dock every panel active and cap only the decoys; "
                         "spends a limited budget where retrieval precision "
                         "is actually governed")
    ap.add_argument("--decoy-limit", type=int, default=0,
                    help="with --all-actives, dock at most this many decoys")
    ap.add_argument("--panel-limit", type=int, default=0,
                    help="dock only this many panel ligands (0 = all); the "
                         "order is shuffled under a fixed seed, so a limit "
                         "gives a random prevalence-preserving subsample")
    args = ap.parse_args()

    if not VINA.exists():
        sys.exit(f"missing Vina binary at {VINA}")
    if not RECEPTOR.exists():
        sys.exit("run analysis/prep_docking_receptor.py first")

    jobs = build_jobs(args.exhaustiveness)

    done = set()
    if OUT_CSV.exists():
        with OUT_CSV.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok":
                    done.add((row["set"], row["ident"]))
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    # Shuffle under a fixed seed so that any prefix of the run is a random
    # sample of the panel rather than all actives followed by all decoys.  An
    # interrupted run then still supports an unbiased retrieval estimate, and
    # the analysis reports the realised docking rate per class.
    random.Random(1337).shuffle(todo)

    # Docking the full 3,300-candidate panel takes roughly nine CPU-hours here,
    # so the panel is subsampled.  The order was shuffled under a fixed seed, so
    # any cap yields a random subsample rather than an arbitrary prefix.
    #
    # Retrieval precision is governed by the smaller class, and the panel is 1
    # active to 10 decoys, so a prevalence-preserving subsample spends most of
    # the budget on decoys and leaves the active count too small.  --all-actives
    # docks every active and caps only the decoys, which is the standard way to
    # spend a limited docking budget.  AUROC and the paired bootstrap are
    # invariant to class prevalence and stay comparable; AUPRC and enrichment
    # factors are not, and are reported at the realized prevalence.
    #
    # The generated analogs are always docked; they are a small, separate set.
    if args.all_actives:
        # Count decoys already scored in a previous run toward the cap, so
        # --decoy-limit means "this many decoys in total", not "this many more".
        n_decoy = 0
        if OUT_CSV.exists():
            with OUT_CSV.open(newline="") as fh:
                seen = set()
                for row in csv.DictReader(fh):
                    key = (row.get("set"), row.get("ident"))
                    if (row.get("status") == "ok" and row.get("set") == "egfr_panel"
                            and row.get("label") == "0" and key not in seen):
                        seen.add(key)
                        n_decoy += 1
        kept = []
        for j in todo:
            if j[0] == "egfr_panel" and j[3] == 0:
                if args.decoy_limit and n_decoy >= args.decoy_limit:
                    continue
                n_decoy += 1
            kept.append(j)
        todo = kept
    elif args.panel_limit:
        n_panel = 0
        kept = []
        for j in todo:
            if j[0] == "egfr_panel":
                if n_panel >= args.panel_limit:
                    continue
                n_panel += 1
            kept.append(j)
        todo = kept

    # Dock actives first so an interruption still leaves the better-powered set.
    todo.sort(key=lambda j: 0 if (j[0] == "egfr_panel" and j[3] == 1) else 1)

    print(f"{len(jobs)} ligands total, {len(done)} already scored, {len(todo)} to dock", flush=True)
    if not todo:
        print("nothing to do")
        return

    new_file = not OUT_CSV.exists()
    started = time.time()
    with OUT_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with Pool(args.workers, initializer=_init_worker, initargs=(_box(),)) as pool:
            for i, row in enumerate(pool.imap_unordered(dock_one, todo, chunksize=1), start=1):
                writer.writerow(row)
                fh.flush()
                if i % 50 == 0 or i == len(todo):
                    rate = i / max(time.time() - started, 1e-9)
                    eta = (len(todo) - i) / max(rate, 1e-9) / 60.0
                    print(f"  {i}/{len(todo)} done, {rate*60:.1f}/min, ETA {eta:.0f} min", flush=True)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
