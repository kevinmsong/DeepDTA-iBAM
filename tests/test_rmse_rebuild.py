"""Regression tests for the modular RMSE-focused DeepDTA-iBAM stack."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd
from safetensors.torch import load_file
import torch

from config_profiles import get_config_profile
from data.cache_builders import GraphCacheBuilder, ProteinEmbeddingCacheBuilder
from data.datasets import TargetGroupedTokenBudgetBatchSampler, TokenBudgetBatchSampler
from training.checkpoints import CheckpointMetadata, cache_hash_from_manifests, save_model_checkpoint
from training.engine import (
    EMA,
    build_dataloaders,
    build_model,
    build_ranking_pair_indices,
    evaluate_ensemble,
    make_loss,
    make_optimizer,
    make_scheduler,
    pairwise_logistic_ranking_loss,
    train_ensemble,
    train_one_epoch,
)
from models.rmse_model import DeepDTAGenIBAM, smiles_to_graph


class MockProteinEmbedder:
    """Deterministic protein embedder used for cache-builder tests."""

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim

    def embed_chunks(self, chunks):
        outputs = []
        for chunk in chunks:
            rows = []
            for character in chunk:
                base = float((ord(character) % 32) + 1)
                vec = torch.tensor(
                    [base, base / 10.0, math.sin(base), math.cos(base), 1.0, 0.5, -0.5, base / 100.0],
                    dtype=torch.float32,
                )
                rows.append(vec[: self.embedding_dim])
            outputs.append(torch.stack(rows, dim=0))
        return outputs




class RMSERebuildTests(unittest.TestCase):
    def _make_smoke_training_config(self, tmp_path: Path, ensemble_size: int = 1):
        rows = [
            {"compound_iso_smiles": "CCO", "target_sequence": "ACDEFGHIKLMN", "affinity": 10.1},
            {"compound_iso_smiles": "c1ccccc1", "target_sequence": "ACDEFGHIKLMNPQ", "affinity": 11.4},
            {"compound_iso_smiles": "CCN(CC)CC", "target_sequence": "QRSTVWYACDEFG", "affinity": 9.8},
            {"compound_iso_smiles": "CC(=O)O", "target_sequence": "QRSTVWYACDEFGH", "affinity": 12.0},
            {"compound_iso_smiles": "CCO", "target_sequence": "QRSTVWYACDEFG", "affinity": 10.3},
            {"compound_iso_smiles": "CCN(CC)CC", "target_sequence": "ACDEFGHIKLMN", "affinity": 9.9},
        ]
        dataframe = pd.DataFrame(rows)
        train_df = dataframe.iloc[:4].reset_index(drop=True)
        val_df = dataframe.iloc[4:5].reset_index(drop=True)
        test_df = dataframe.iloc[5:].reset_index(drop=True)
        train_path = tmp_path / "train.csv"
        val_path = tmp_path / "val.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        config = get_config_profile("max_rmse_cluster")
        config.train_file = str(train_path)
        config.val_file = str(val_path)
        config.test_file = str(test_path)
        config.cache_root = str(tmp_path / "cache")
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.graph_cache_name = "graphs.pt"
        config.graph_manifest_name = "graphs_manifest.json"
        config.protein_cache_name = "proteins.pt"
        config.protein_manifest_name = "proteins_manifest.json"
        config.num_workers = 0
        config.use_amp = False
        config.device = "cpu"
        config.max_pairs_per_batch = 2
        config.protein_token_budget = 64
        config.num_epochs = 1
        config.diffusion_max_weight = 0.02
        config.diffusion_warmup_epochs = 0
        config.diffusion_ramp_end_epoch = 1
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
        config.ensemble_size = ensemble_size
        config.dropout = 0.0

        GraphCacheBuilder(config).build(dataframe["compound_iso_smiles"].tolist())
        ProteinEmbeddingCacheBuilder(config, embedder=MockProteinEmbedder(embedding_dim=8)).build(
            dataframe["target_sequence"].tolist()
        )
        return config, dataframe, val_df

    def test_gasteiger_features_not_collapsed(self):
        node_features, _, mask, _ = smiles_to_graph("C[N+](C)(C)C", max_atoms=None)
        self.assertEqual(int(mask.sum().item()), node_features.size(0))
        charge_bins = node_features[:, 70:74]
        self.assertGreater(float(charge_bins.sum().item()), 0.0)
        non_default_bins = charge_bins[:, [0, 1, 3]].sum().item()
        self.assertGreater(non_default_bins, 0.0, "expected at least one non-default Gasteiger bin")

    def test_masked_padding_does_not_change_prediction(self):
        torch.manual_seed(7)
        model = DeepDTAGenIBAM(
            gat_hidden_dim=64,
            gat_layers=2,
            gat_heads=8,
            protein_embedding_dim=8,
            protein_adapter_dim=32,
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
        model.eval()

        graph = smiles_to_graph("CCO", max_atoms=None)
        drug_x = graph[0].unsqueeze(0)
        drug_adj = graph[1].unsqueeze(0)
        drug_mask = graph[2].unsqueeze(0)
        drug_edge = graph[3].unsqueeze(0)

        base_embeddings = torch.randn(1, 5, 8)
        base_mask = torch.ones(1, 5, dtype=torch.bool)
        padded_embeddings = torch.cat([base_embeddings, torch.randn(1, 4, 8)], dim=1)
        padded_mask = torch.tensor([[True, True, True, True, True, False, False, False, False]])

        base_out, base_attn, _ = model(
            drug_x,
            drug_adj,
            drug_mask,
            base_embeddings,
            base_mask,
            drug_edge_features=drug_edge,
            compute_diff_loss=False,
        )
        padded_out, padded_attn, _ = model(
            drug_x,
            drug_adj,
            drug_mask,
            padded_embeddings,
            padded_mask,
            drug_edge_features=drug_edge,
            compute_diff_loss=False,
        )

        self.assertTrue(torch.allclose(base_out, padded_out, atol=1e-5))
        self.assertTrue(
            torch.allclose(base_attn["residue_pool"], padded_attn["residue_pool"][:, :5], atol=1e-5)
        )

    def test_long_protein_cache_chunking_and_stitching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = get_config_profile("max_rmse_cluster")
            config.cache_root = str(tmp_path / "cache")
            config.protein_embedding_dim = 8
            config.protein_window_size = 16
            config.protein_window_overlap = 4
            config.protein_cache_dtype = "bfloat16"

            sequence = ("ACDEFGHIKLMNPQRSTVWY" * 7)[:103]
            builder = ProteinEmbeddingCacheBuilder(config, embedder=MockProteinEmbedder(embedding_dim=8))
            manifest = builder.build([sequence])

            cache = torch.load(config.protein_cache_path, map_location="cpu", weights_only=True)
            self.assertEqual(len(manifest["items"]), 1)
            cached = next(iter(cache.values()))
            self.assertEqual(cached["embeddings"].size(0), len(sequence))
            self.assertEqual(int(cached["mask"].sum().item()), len(sequence))

            expected_first = MockProteinEmbedder(8).embed_chunks([sequence[:1]])[0][0]
            self.assertTrue(torch.allclose(cached["embeddings"][0].float(), expected_first, atol=5e-3))

    def test_token_budget_sampler_len_matches_actual_batches(self):
        lengths = [20, 20, 20, 1000, 1000, 1000, 2000]
        sampler = TokenBudgetBatchSampler(
            lengths,
            max_pairs_per_batch=4,
            protein_token_budget=2500,
            shuffle=True,
            seed=123,
            bucket_size=3,
        )
        sampler.set_epoch(2)
        batches = list(iter(sampler))
        self.assertEqual(len(sampler), len(batches))

    def test_target_grouped_sampler_keeps_batches_single_target_and_within_budget(self):
        lengths = [12, 12, 12, 18, 18, 30]
        group_keys = ["A", "A", "A", "B", "B", "C"]
        sampler = TargetGroupedTokenBudgetBatchSampler(
            lengths,
            group_keys,
            max_pairs_per_batch=2,
            protein_token_budget=36,
            shuffle=True,
            seed=99,
        )
        sampler.set_epoch(1)
        batches = list(iter(sampler))
        self.assertEqual(len(sampler), len(batches))
        for batch in batches:
            batch_groups = {group_keys[idx] for idx in batch}
            self.assertEqual(len(batch_groups), 1)
            self.assertLessEqual(len(batch), 2)
            max_length = max(lengths[idx] for idx in batch)
            self.assertLessEqual(len(batch) * max_length, 36)

    def test_build_ranking_pair_indices_filters_same_target_and_delta(self):
        targets = torch.tensor([[10.0], [10.2], [12.0], [8.0], [8.6]], dtype=torch.float32)
        group_keys = ["A", "A", "A", "B", "B"]
        left, right, signs = build_ranking_pair_indices(targets, group_keys, min_delta=0.3, max_pairs=10)

        self.assertEqual({(int(i), int(j)) for i, j in zip(left.tolist(), right.tolist())}, {(0, 2), (1, 2), (3, 4)})
        self.assertTrue(torch.all(signs < 0))

    def test_pairwise_ranking_loss_returns_zero_without_informative_pairs(self):
        predictions = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
        targets = torch.tensor([[10.0], [10.1], [11.0]], dtype=torch.float32)
        group_keys = ["A", "A", "B"]
        loss, pair_count = pairwise_logistic_ranking_loss(
            predictions,
            targets,
            group_keys,
            min_delta=0.3,
            max_pairs=32,
        )
        self.assertEqual(pair_count, 0)
        self.assertEqual(float(loss.item()), 0.0)

    def test_smoke_train_epoch_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config, dataframe, val_df = self._make_smoke_training_config(tmp_path, ensemble_size=1)

            from data.cache_builders import GraphCache, ProteinEmbeddingCache

            graph_cache = GraphCache(config.graph_cache_path, config.graph_manifest_path)
            protein_cache = ProteinEmbeddingCache(config.protein_cache_path, config.protein_manifest_path)
            train_loader, val_loader, _, normalizer = build_dataloaders(config, graph_cache, protein_cache)

            model = build_model(config)
            optimizer = make_optimizer(model, config)
            scheduler = make_scheduler(optimizer, total_steps=max(1, len(train_loader)), config=config)
            ema = EMA(model, config.ema_decay)
            metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=make_loss(config),
                config=config,
                normalizer=normalizer,
                epoch=1,
                scaler=None,
                ema=ema,
            )
            self.assertIn("RMSE", metrics)
            self.assertGreater(metrics["RMSE"], 0.0)

            ema.apply_shadow()
            ensemble_metrics = evaluate_ensemble([model], val_loader, config, normalizer)
            ema.restore()
            self.assertIn("predictions", ensemble_metrics)
            self.assertIn("AttentionEntropy", ensemble_metrics)
            self.assertEqual(len(ensemble_metrics["predictions"]), len(val_df))

            cache_hash = cache_hash_from_manifests([config.graph_manifest_path, config.protein_manifest_path])
            metadata = CheckpointMetadata(
                profile_name=config.profile_name,
                member_index=0,
                member_seed=config.seed,
                epoch=1,
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))},
                config_hash=config.config_hash(),
                cache_hash=cache_hash,
                target_mean=normalizer.mean,
                target_std=normalizer.std,
            )
            checkpoint_path = Path(config.checkpoint_dir) / "smoke_member_0.safetensors"
            save_model_checkpoint(
                model.state_dict(),
                metadata,
                checkpoint_path,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
            )

            loaded_state = load_file(str(checkpoint_path), device="cpu")
            self.assertTrue(loaded_state)
            with checkpoint_path.with_suffix(".json").open("r", encoding="utf-8") as handle:
                saved_metadata = json.load(handle)
            self.assertEqual(saved_metadata["cache_hash"], cache_hash)

    def test_train_ensemble_resume_continues_from_latest_complete_checkpoint_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config, _, _ = self._make_smoke_training_config(tmp_path, ensemble_size=2)
            config.patience = 2

            initial_summary = train_ensemble(config)
            self.assertIsNone(initial_summary["resumed_from"])
            self.assertEqual(initial_summary["history"][0]["epoch"], 1)

            resumed_summary = train_ensemble(config, resume=True)
            self.assertEqual(resumed_summary["resumed_from"]["epoch"], 1)
            self.assertEqual(len(resumed_summary["resumed_from"]["checkpoint_paths"]), 2)
            self.assertEqual([entry["epoch"] for entry in resumed_summary["history"]], [2])
            self.assertEqual(resumed_summary["cuda_gpu_count"], 0)

    def test_train_ensemble_reloads_best_checkpoint_before_test_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config, _, _ = self._make_smoke_training_config(tmp_path, ensemble_size=1)

            with mock.patch("training.engine._load_saved_best_ensemble", wraps=__import__("training.engine", fromlist=["_load_saved_best_ensemble"])._load_saved_best_ensemble) as loader_mock:
                summary = train_ensemble(config)

            self.assertTrue(loader_mock.called)
            self.assertIn("best_val_ci", summary)
            self.assertEqual(summary["selection_metric"], config.selection_metric)


if __name__ == "__main__":
    unittest.main()
