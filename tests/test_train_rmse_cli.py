from __future__ import annotations

import unittest

from train_rmse import build_parser, prepare_training_config


class TrainRmseCliTests(unittest.TestCase):
    def test_enable_diffusion_on_base_profile_applies_expected_schedule(self):
        args = build_parser().parse_args(["--profile", "max_rmse_cluster", "--enable-diffusion"])
        config = prepare_training_config(args)
        self.assertEqual(config.profile_name, "max_rmse_cluster_diffusion")
        self.assertAlmostEqual(config.diffusion_max_weight, 0.02)
        self.assertEqual(config.diffusion_warmup_epochs, 2)
        self.assertEqual(config.diffusion_ramp_end_epoch, 6)

    def test_disable_diffusion_zeroes_diffusion_schedule(self):
        args = build_parser().parse_args(["--profile", "diffusion_egfr_seed", "--disable-diffusion"])
        config = prepare_training_config(args)
        self.assertEqual(config.profile_name, "diffusion_egfr_seed")
        self.assertEqual(config.diffusion_max_weight, 0.0)
        self.assertEqual(config.diffusion_warmup_epochs, 0)
        self.assertEqual(config.diffusion_ramp_end_epoch, 0)


if __name__ == "__main__":
    unittest.main()
