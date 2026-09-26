"""Reference CLI behavior and map aggregation, without training or downloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reference/run_demo.py"


def invoke(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def test_random_mode_is_explicitly_untrained_and_saves_normalized_map(tmp_path):
    result = invoke("--mode", "random", "--smiles", "CCO", "--output-dir", tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "random"
    assert "UNTRAINED" in summary["status"]
    assert "predicted_affinity_kiba_units" not in summary
    assert summary["matches_audited_manuscript_checkpoint"] is False
    assert summary["interaction_map_shape"] == [3, 16]
    arrays = np.load(tmp_path / "interaction_map.npz")
    assert arrays["atom_to_residue"].shape == (3, 16)
    np.testing.assert_allclose(arrays["atom_to_residue"].sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(arrays["residue_profile"], arrays["atom_to_residue"].mean(axis=0))


def test_missing_trained_weights_fail_without_random_fallback(tmp_path):
    result = invoke("--checkpoint", tmp_path / "missing.safetensors")
    assert result.returncode == 2
    assert "Trained inference requires existing assets" in result.stderr
    assert not result.stdout.strip()


def test_invalid_smiles_is_rejected():
    result = invoke("--mode", "random", "--smiles", "not-a-molecule")
    assert result.returncode == 2
    assert "not a valid molecular structure" in result.stderr


def test_attention_average_excludes_padding_and_combines_blocks_and_heads():
    spec = importlib.util.spec_from_file_location("reference_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = torch.tensor([[[[0.2, 0.8, 99.0], [8., 9., 99.]],
                           [[0.4, 0.6, 99.0], [8., 9., 99.]]]])
    second = torch.tensor([[[[0.6, 0.4, 99.0], [8., 9., 99.]],
                            [[0.8, 0.2, 99.0], [8., 9., 99.]]]])
    matrix, keys = module.mean_interaction_map(
        {"layer_0_atom_to_residue": first, "layer_1_atom_to_residue": second,
         "layer_0_residue_to_atom": torch.zeros(1)},
        torch.tensor([[True, False]]), torch.tensor([[True, True, False]]))
    assert len(keys) == 2
    torch.testing.assert_close(matrix, torch.tensor([[0.5, 0.5]]))


ASSETS = [ROOT / "checkpoints/max_rmse_cluster_diffusion_member_0.safetensors",
          ROOT / "checkpoints/max_rmse_cluster_diffusion_member_0.json",
          ROOT / "data/cache/proteins.pt", ROOT / "data/cache/proteins_manifest.json"]


@pytest.mark.skipif(not all(p.is_file() for p in ASSETS), reason="original checkpoint/cache assets are not distributed")
def test_local_trained_assets_use_audited_weights_and_emit_affinity(tmp_path):
    result = invoke("--smiles", "CCO", "--output-dir", tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "trained"
    assert summary["matches_audited_manuscript_checkpoint"] is True
    assert np.isfinite(summary["predicted_affinity_kiba_units"])
    assert "untrained_raw_output_not_affinity" not in summary
    assert summary["interaction_map_shape"][0] == 3
    assert summary["interaction_map_shape"][1] > 16
    assert summary["maximum_attention_row_sum_error"] < 1e-5
