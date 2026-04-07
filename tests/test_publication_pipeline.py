from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from case_studies_results_generation import (
    PublicationContext,
    build_main_tex,
    decode_generated_analog,
    molecule_properties,
    rank_generated_analogs,
    run_benchmark_section,
    run_manuscript_section,
)
from config_profiles import get_config_profile
from data.cache_builders import GraphCacheBuilder, ProteinEmbeddingCacheBuilder
from models.rmse_model import DeepDTAGenIBAM, smiles_to_graph
from training.inference import make_unlabeled_prediction_loader, predict_unlabeled
from data.datasets import TargetNormalizer


class MockProteinEmbedder:
    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim

    def embed_chunks(self, chunks):
        outputs = []
        for chunk in chunks:
            rows = []
            for index, character in enumerate(chunk):
                base = float((ord(character) % 21) + index + 1)
                rows.append(torch.tensor([base, base / 10.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25], dtype=torch.float32))
            outputs.append(torch.stack(rows, dim=0))
        return outputs


class PublicationPipelineSmokeTests(unittest.TestCase):
    def test_benchmark_and_manuscript_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            args = argparse.Namespace(
                profile="max_rmse_cluster_diffusion",
                profile_name_override=None,
                checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
                member_count=1,
                fusion_mode=None,
                sections=["benchmark", "manuscript"],
                results_dir=str(results_dir),
                device="cpu",
                force_refresh=False,
                screen_lib_size=1000,
                num_decoy_proteins=4,
                skip_manuscript=False,
            )
            config = get_config_profile("max_rmse_cluster_diffusion")
            config.device = "cpu"
            config.checkpoint_dir = args.checkpoint_dir
            config.ensemble_size = 1
            ctx = PublicationContext(
                args=args,
                base_config=config,
                results_dir=results_dir,
                metrics_path=results_dir / "case_study_metrics.json",
                source_manifest_path=results_dir / "source_manifest.json",
            )
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / f"{config.profile_name}_member_0.safetensors").write_bytes(b"placeholder")
            fake_main_metrics = {"CI": 0.901, "RMSE": 0.131, "MAE": 0.089, "predictions": [1.0], "targets": [1.0]}
            ctx.metrics["ibam"] = {"predicted_affinity": 11.9}
            ctx.metrics["fishing"] = {"BEDROC20": 0.22}
            ctx.metrics["generation"] = {"best_pred_affinity": 12.1, "num_unique_valid_analogs": 100}
            ctx.metrics["diagnostics"] = {"bias_mean": -0.03}

            with mock.patch("case_studies_results_generation.evaluate_config_on_split", return_value=fake_main_metrics):
                run_benchmark_section(ctx)
            run_manuscript_section(ctx)
            self.assertTrue((results_dir / "table1_benchmark.csv").exists())
            self.assertIn("DeepDTA-iBAM", (results_dir / "table1_benchmark.csv").read_text(encoding="utf-8"))
            self.assertNotIn("DeepDTA-iBAM no fusion", (results_dir / "table1_benchmark.csv").read_text(encoding="utf-8"))
            manuscript = build_main_tex(ctx)
            self.assertIn("model_architecture.png", manuscript)
            self.assertNotIn("Caption pending.", manuscript)
            self.assertNotIn("scaffold-split", manuscript.lower())
            self.assertNotIn("no-fusion", manuscript.lower())
            self.assertTrue(Path("main.tex").exists())
            self.assertTrue(Path("supplementary.tex").exists())
            self.assertTrue(Path("references.bib").exists())

    def test_decoder_preserves_valid_molecule_and_properties(self):
        seed = Chem.MolFromSmiles("CCO")
        features, _, mask, _ = smiles_to_graph("CCO", max_atoms=None)
        decoded = decode_generated_analog(seed, features.numpy())
        self.assertIsNotNone(decoded)
        properties = molecule_properties(decoded)
        self.assertIn("QED", properties)
        self.assertIn("SA", properties)
        self.assertEqual(int(mask.sum().item()), decoded.GetNumAtoms())

    def test_rank_generated_analogs_uses_expected_priority(self):
        dataframe = pd.DataFrame(
            [
                {"smiles": "A", "LipinskiPass": 1, "AlertFree": 1, "QED": 0.70, "SA": 2.5, "PredAffinity": 11.0},
                {"smiles": "B", "LipinskiPass": 1, "AlertFree": 1, "QED": 0.72, "SA": 2.7, "PredAffinity": 10.9},
                {"smiles": "C", "LipinskiPass": 1, "AlertFree": 0, "QED": 0.95, "SA": 1.0, "PredAffinity": 12.0},
                {"smiles": "D", "LipinskiPass": 0, "AlertFree": 1, "QED": 0.99, "SA": 1.0, "PredAffinity": 12.5},
                {"smiles": "E", "LipinskiPass": 1, "AlertFree": 1, "QED": 0.72, "SA": 2.4, "PredAffinity": 10.8},
            ]
        )
        ranked = rank_generated_analogs(dataframe)
        self.assertEqual(ranked["smiles"].tolist(), ["E", "B", "A", "C", "D"])

    def test_unlabeled_prediction_loader_and_attention_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = get_config_profile("max_rmse_cluster")
            config.cache_root = str(tmp_path / "cache")
            config.device = "cpu"
            config.use_amp = False
            config.num_workers = 0
            config.normalize_targets = False
            config.protein_embedding_dim = 8
            config.protein_adapter_dim = 16
            config.protein_adapter_heads = 4
            config.protein_adapter_layers = 1
            config.protein_adapter_ff_mult = 2
            config.gat_hidden_dim = 64
            config.gat_layers = 2
            config.gat_heads = 8
            config.fusion_dim = 32
            config.fusion_heads = 4
            config.fusion_layers = 1
            config.fc_hidden_dim = 64
            config.diff_hidden_dim = 32
            config.diff_T = 16
            config.diff_inference_steps = 4
            config.max_pairs_per_batch = 2
            config.protein_token_budget = 128
            GraphCacheBuilder(config).build(["CCO"])
            ProteinEmbeddingCacheBuilder(config, embedder=MockProteinEmbedder(embedding_dim=8)).build(["ACDEFG"])

            from data.cache_builders import GraphCache, ProteinEmbeddingCache

            graph_cache = GraphCache(config.graph_cache_path, config.graph_manifest_path)
            protein_cache = ProteinEmbeddingCache(config.protein_cache_path, config.protein_manifest_path)
            dataframe = pd.DataFrame([{"compound_iso_smiles": "CCO", "target_sequence": "ACDEFG"}])
            loader = make_unlabeled_prediction_loader(dataframe, graph_cache, protein_cache, config)
            model = DeepDTAGenIBAM(
                gat_hidden_dim=64,
                gat_layers=2,
                gat_heads=8,
                protein_embedding_dim=8,
                protein_adapter_dim=16,
                protein_adapter_heads=4,
                protein_adapter_layers=1,
                fusion_dim=32,
                fusion_layers=1,
                fusion_heads=4,
                fc_hidden_dim=64,
                dropout=0.0,
                diff_hidden_dim=32,
                diff_T=16,
                diff_inference_steps=4,
            )
            normalizer = TargetNormalizer(mean=0.0, std=1.0)
            payload = predict_unlabeled([model], loader, config, normalizer, collect_attention=True)
            self.assertEqual(len(payload["predictions"]), 1)
            self.assertTrue(payload["attention"])

    def test_no_fusion_forward_uses_pooled_embeddings(self):
        model = DeepDTAGenIBAM(
            gat_hidden_dim=64,
            gat_layers=2,
            gat_heads=8,
            protein_embedding_dim=8,
            protein_adapter_dim=16,
            protein_adapter_heads=4,
            protein_adapter_layers=1,
            fusion_dim=32,
            fusion_layers=1,
            fusion_heads=4,
            fc_hidden_dim=64,
            dropout=0.0,
            diff_hidden_dim=32,
            diff_T=16,
            diff_inference_steps=4,
            fusion_mode="none",
        )
        drug_x = torch.randn(2, 5, 78)
        drug_adj = torch.eye(5).repeat(2, 1, 1)
        drug_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        protein_embeddings = torch.randn(2, 7, 8)
        protein_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        output, attention, diff_loss = model(
            drug_x,
            drug_adj,
            drug_mask,
            protein_embeddings,
            protein_mask,
            compute_diff_loss=True,
        )
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertIn("fusion_mode_none", attention)
        self.assertIsNotNone(diff_loss)


if __name__ == "__main__":
    unittest.main()
