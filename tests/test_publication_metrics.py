from __future__ import annotations

import unittest

import numpy as np

from utils.metrics import (
    auprc,
    auroc,
    bedroc,
    enrichment_factor,
    paired_bootstrap_metric_delta,
    precision_recall_f1_at_fraction,
    topk_recovery,
)


class PublicationMetricTests(unittest.TestCase):
    def test_enrichment_factor_and_recovery(self):
        labels = np.array([1, 0, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
        self.assertAlmostEqual(enrichment_factor(labels, scores, 0.2), 2.5, places=6)
        self.assertAlmostEqual(topk_recovery(labels, scores, 0.2), 0.5, places=6)

    def test_precision_recall_f1_at_fraction(self):
        labels = np.array([1, 0, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
        metrics = precision_recall_f1_at_fraction(labels, scores, 0.4)
        self.assertEqual(metrics["top_n"], 2)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=6)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=6)
        self.assertAlmostEqual(metrics["f1"], 0.5, places=6)

    def test_bedroc_and_auc_reward_early_ranking(self):
        labels = np.array([1, 0, 0, 1, 0, 0])
        strong_scores = np.array([0.95, 0.8, 0.7, 0.6, 0.2, 0.1])
        weak_scores = np.array([0.2, 0.95, 0.8, 0.1, 0.7, 0.6])
        self.assertGreater(bedroc(labels, strong_scores, alpha=20.0), bedroc(labels, weak_scores, alpha=20.0))
        self.assertGreater(auroc(labels, strong_scores), auroc(labels, weak_scores))
        self.assertGreater(auprc(labels, strong_scores), auprc(labels, weak_scores))

    def test_paired_bootstrap_delta(self):
        truth = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pred_a = np.array([0.1, 1.0, 2.1, 2.9, 4.0])
        pred_b = np.array([1.0, 0.0, 2.5, 1.5, 2.5])
        delta = paired_bootstrap_metric_delta(
            truth,
            pred_a,
            pred_b,
            lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2),
            n_boot=200,
            seed=7,
        )
        self.assertGreater(delta["delta"], 0.0)
        self.assertGreaterEqual(delta["ci_high"], delta["ci_low"])


if __name__ == "__main__":
    unittest.main()
