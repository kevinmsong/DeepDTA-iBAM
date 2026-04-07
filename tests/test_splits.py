"""Tests for hardened dataset split generation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from data.splits import prepare_split_artifacts


def _write_existing_split_files(root: Path, dataframe: pd.DataFrame) -> tuple[str, str, str]:
    train_path = root / "train_existing.csv"
    val_path = root / "val_existing.csv"
    test_path = root / "test_existing.csv"

    dataframe.iloc[: len(dataframe) // 2].reset_index(drop=True).to_csv(train_path, index=False)
    dataframe.iloc[len(dataframe) // 2 : -2].reset_index(drop=True).to_csv(val_path, index=False)
    dataframe.iloc[-2:].reset_index(drop=True).to_csv(test_path, index=False)
    return str(train_path), str(val_path), str(test_path)


def _scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


class SplitGenerationTests(unittest.TestCase):
    def test_scaffold_split_has_no_scaffold_overlap(self):
        rows = [
            {"compound_iso_smiles": "c1ccccc1", "target_sequence": "T1", "affinity": 9.1},
            {"compound_iso_smiles": "Cc1ccccc1", "target_sequence": "T2", "affinity": 9.2},
            {"compound_iso_smiles": "c1ccncc1", "target_sequence": "T3", "affinity": 9.3},
            {"compound_iso_smiles": "Cc1ccncc1", "target_sequence": "T4", "affinity": 9.4},
            {"compound_iso_smiles": "C1CCCCC1", "target_sequence": "T5", "affinity": 9.5},
            {"compound_iso_smiles": "CC1CCCCC1", "target_sequence": "T6", "affinity": 9.6},
            {"compound_iso_smiles": "c1ccc2[nH]ccc2c1", "target_sequence": "T7", "affinity": 9.7},
            {"compound_iso_smiles": "Cc1ccc2[nH]ccc2c1", "target_sequence": "T8", "affinity": 9.8},
        ]
        dataframe = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_file, val_file, test_file = _write_existing_split_files(Path(tmpdir), dataframe)
            artifacts = prepare_split_artifacts(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file,
                mode="scaffold",
                output_root=str(Path(tmpdir) / "generated"),
                seed=23,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
            )

            train_df = pd.read_csv(artifacts.train_file)
            val_df = pd.read_csv(artifacts.val_file)
            test_df = pd.read_csv(artifacts.test_file)

            self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(dataframe))

            train_scaffolds = {_scaffold(smiles) for smiles in train_df["compound_iso_smiles"].unique()}
            val_scaffolds = {_scaffold(smiles) for smiles in val_df["compound_iso_smiles"].unique()}
            test_scaffolds = {_scaffold(smiles) for smiles in test_df["compound_iso_smiles"].unique()}

            self.assertFalse(train_scaffolds & val_scaffolds)
            self.assertFalse(train_scaffolds & test_scaffolds)
            self.assertFalse(val_scaffolds & test_scaffolds)

    def test_non_scaffold_modes_are_rejected(self):
        dataframe = pd.DataFrame(
            [
                {"compound_iso_smiles": "c1ccccc1", "target_sequence": "T1", "affinity": 9.1},
                {"compound_iso_smiles": "c1ccncc1", "target_sequence": "T2", "affinity": 9.2},
                {"compound_iso_smiles": "C1CCCCC1", "target_sequence": "T3", "affinity": 9.3},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            train_file, val_file, test_file = _write_existing_split_files(Path(tmpdir), dataframe)
            with self.assertRaisesRegex(ValueError, "Only 'scaffold' and 'standard' are supported"):
                prepare_split_artifacts(
                    train_file=train_file,
                    val_file=val_file,
                    test_file=test_file,
                    mode="cold-target",
                    output_root=str(Path(tmpdir) / "generated"),
                )

    def test_standard_mode_returns_existing_split_paths_without_materialization(self):
        dataframe = pd.DataFrame(
            [
                {"compound_iso_smiles": "c1ccccc1", "target_sequence": "T1", "affinity": 9.1},
                {"compound_iso_smiles": "c1ccncc1", "target_sequence": "T2", "affinity": 9.2},
                {"compound_iso_smiles": "C1CCCCC1", "target_sequence": "T3", "affinity": 9.3},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            train_file, val_file, test_file = _write_existing_split_files(Path(tmpdir), dataframe)
            artifacts = prepare_split_artifacts(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file,
                mode="standard",
                output_root=str(Path(tmpdir) / "generated"),
            )
            self.assertEqual(artifacts.train_file, train_file)
            self.assertEqual(artifacts.val_file, val_file)
            self.assertEqual(artifacts.test_file, test_file)
            self.assertIsNone(artifacts.manifest_path)
            self.assertEqual(artifacts.summary["mode"], "standard")


if __name__ == "__main__":
    unittest.main()
