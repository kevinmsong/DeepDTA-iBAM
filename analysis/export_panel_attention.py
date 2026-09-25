"""Export one interaction profile per ligand for the in-domain EGFR panel.

A reviewer asked whether the interaction maps carry information relevant to
affinity, and pointed out that crystallographic contacts exist only for
compounds that bind, so a contact benchmark scored against a crystal pose can
never include a non-binder.  The in-domain EGFR panel avoids that problem: the
protein is held fixed while the ligand varies across 300 assay-defined actives
and 3,000 property-matched decoys, and the model emits an interaction map for
every one of them whether it binds or not.

This script runs the released checkpoint over all 3,300 candidates and writes,
for each, the mean ligand-to-residue attention profile.  The downstream
analysis in analysis_contact_information.py then asks whether those profiles
differ between ligands at all, and whether affinity or binder status can be
modelled from them.

The existing predict_unlabeled helper averages attention over the batch, which
would collapse the per-ligand structure this analysis needs, so the forward
loop is reimplemented here.  Padding is respected: each profile averages over
that ligand's real atoms only.

Outputs
-------
  results/panel_attention_residue.npy    (N, R) float32 per-ligand profiles
  results/panel_attention_index.csv      one row per ligand, aligned to the npy
  results/panel_attention_meta.json      shapes and provenance

Run from anywhere:  python analysis/export_panel_attention.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

from config_profiles import get_config_profile
from data.cache_builders import build_isolated_caches
from training.inference import make_unlabeled_prediction_loader, _move_batch_to_device
from training.inference import runtime_device, _autocast_context
from case_studies_results_generation import load_publication_ensemble

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_guard import apply_batch_limit

PANEL = ROOT / "results" / "dude_egfr_panel_scored.csv"
SEQ_MANIFEST = (ROOT / "results" / "cache" / "egfr_interpolation" /
                "egfr_interpolation_proteins_manifest.json")
CACHE_DIR = ROOT / "results" / "cache" / "dude_egfr_panel"

OUT_NPY = ROOT / "results" / "panel_attention_residue.npy"
OUT_IDX = ROOT / "results" / "panel_attention_index.csv"
OUT_META = ROOT / "results" / "panel_attention_meta.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    panel = pd.read_csv(PANEL)
    smiles = panel["smiles"].astype(str).tolist()
    log(f"panel: {len(panel)} candidates, {int(panel['label'].sum())} actives")

    seq = next(iter(json.loads(SEQ_MANIFEST.read_text())["items"].values()))["sequence"]
    log(f"EGFR sequence length {len(seq)}")

    cfg = get_config_profile("max_rmse_cluster_diffusion")
    cfg.build_caches_on_start = False
    cfg.ensemble_size = 1

    log("building/loading ligand caches")
    iso, graph_cache, protein_cache = build_isolated_caches(
        cfg, smiles, [seq], str(CACHE_DIR),
        force_rebuild=False, cache_prefix="dude_egfr_panel",
    )
    iso.num_workers = 0  # Windows spawn guard, as in score_kinase_panel.py
    # Batching is by token budget, not by batch_size, and the attention tensors
    # grow with the square of target length.  Size the batch from the target and
    # the memory actually available rather than a fixed guess.
    mem_info = apply_batch_limit(iso, [seq])
    models, normalizer = load_publication_ensemble(iso, checkpoint_dir=cfg.checkpoint_dir)
    model = models[0]
    model.eval()

    frame = pd.DataFrame({"compound_iso_smiles": smiles,
                          "target_sequence": [seq] * len(smiles)})
    loader = make_unlabeled_prediction_loader(frame, graph_cache, protein_cache, iso)
    device = runtime_device(iso)

    profiles: list[np.ndarray] = []
    n_atoms_list: list[int] = []
    preds: list[float] = []
    reported = False
    started = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(iso):
                output, attention_maps, _ = model(
                    batch["drug_x"], batch["drug_adj"], batch["drug_mask"],
                    batch["protein_embeddings"], batch["protein_mask"],
                    drug_edge_features=batch["drug_edge_features"],
                    compute_diff_loss=False,
                )

            keys = [k for k in attention_maps
                    if "atom_to_residue" in k
                    and isinstance(attention_maps[k], torch.Tensor)
                    and attention_maps[k].ndim == 4]
            if not keys:
                raise SystemExit(
                    "no 4-D atom_to_residue attention found; keys were "
                    f"{list(attention_maps)}"
                )
            key = ",".join(keys)
            drug_mask = batch["drug_mask"].detach().float()        # (B, A) on device
            m = drug_mask.unsqueeze(-1)                            # (B, A, 1)
            denom = m.sum(dim=1).clamp(min=1.0)                    # (B, 1)

            if not reported:
                log(f"attention keys '{key}' shape "
                    f"{tuple(attention_maps[keys[0]].shape)}, "
                    f"drug_mask {tuple(drug_mask.shape)}, "
                    f"protein_mask {tuple(batch['protein_mask'].shape)}")
                reported = True

            # Average over fusion layers, matching the aggregation the released
            # benchmark uses in _attention_arrays_from_batches.  Reducing one
            # layer at a time keeps peak memory at a single (B, A, R) tensor;
            # stacking all layers first costs a few hundred MB per batch and
            # was enough to have the process killed.
            acc = None
            for k in keys:
                a = attention_maps[k].detach().float().mean(dim=1)  # (B, A, R)
                pk = (a * m).sum(dim=1) / denom                     # (B, R)
                acc = pk if acc is None else acc + pk
                del a, pk
            prof = (acc / float(len(keys))).cpu()                   # (B, R)
            del acc

            profiles.append(prof.numpy().astype(np.float32))
            n_atoms_list.extend(drug_mask.sum(dim=1).to(torch.int32).cpu().tolist())
            preds.extend(output.detach().float().cpu().view(-1).tolist())

            del attention_maps, output, batch, drug_mask, m, denom, prof

            if bi % 5 == 0:
                done = sum(p.shape[0] for p in profiles)
                rate = done / max(time.time() - started, 1e-9)
                log(f"  {done}/{len(panel)} ligands, {rate:.1f}/s, "
                    f"ETA {(len(panel)-done)/max(rate,1e-9)/60:.1f} min")
                # Checkpoint so a long run is never lost to an interruption.
                np.save(OUT_NPY.with_suffix(".partial.npy"),
                        np.concatenate(profiles, axis=0))
                gc.collect()

    prof_arr = np.concatenate(profiles, axis=0)
    if prof_arr.shape[0] != len(panel):
        raise SystemExit(f"profile count {prof_arr.shape[0]} != panel {len(panel)}")

    # Predictions come out of the model on the normalized scale; rescale so the
    # index file carries a directly comparable affinity value.
    raw = torch.tensor(preds, dtype=torch.float32)
    from training.inference import _denormalize_predictions
    denorm = _denormalize_predictions(raw, normalizer, iso.normalize_targets)
    denorm = np.asarray(denorm, dtype=float).reshape(-1)

    np.save(OUT_NPY, prof_arr)
    idx = panel[["role", "ident", "smiles", "label", "model_score", "ecfp_score"]].copy()
    idx["n_heavy_atoms_padded"] = n_atoms_list
    idx["model_pred_rerun"] = denorm
    idx.to_csv(OUT_IDX, index=False)

    OUT_META.write_text(json.dumps({
        "n_ligands": int(prof_arr.shape[0]),
        "n_residue_positions": int(prof_arr.shape[1]),
        "sequence_length": len(seq),
        "attention_key": key,
        "profile_definition": "mean over the ligand's real atoms of atom-to-residue attention, heads averaged",
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "seconds": round(time.time() - started, 1),
    }, indent=2))

    log(f"wrote {OUT_NPY.name} {prof_arr.shape}, {OUT_IDX.name}, {OUT_META.name}")
    corr = np.corrcoef(idx["model_score"], idx["model_pred_rerun"])[0, 1]
    log(f"rerun vs released model_score correlation: {corr:.6f}")


if __name__ == "__main__":
    main()
