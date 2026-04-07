# DeepDTA-iBAM

**Interpretable Cross-Attention for Affinity Prediction, Target-Conditioned Retrieval, and Generative Drug Design**

DeepDTA-iBAM is a multimodal deep learning framework that unifies drug-target affinity (DTA) prediction, interpretable interaction mapping, target-conditioned ligand retrieval, and seeded molecular design within a single architecture. The model combines graph-based ligand encoding, cached ESM-C protein embeddings, and bidirectional atom-residue cross-attention with a diffusion auxiliary head for target-conditioned molecular generation.

## Architecture

DeepDTA-iBAM has five main components:

1. **Ligand encoder** — Multi-head graph attention over atom-bond graphs with edge-feature bias (78 atom features, 12 bond features)
2. **Protein adapter** — Learned projection of cached ESM-C residue embeddings into the shared fusion space
3. **Bidirectional cross-attention** — Atom-to-residue and residue-to-atom attention producing interpretable interaction maps (iBAM)
4. **Affinity prediction head** — KIBA score regression from the fused multimodal state
5. **Diffusion auxiliary head** — Target-conditioned denoising for seeded, topology-preserving molecular design

## Installation

```bash
git clone https://github.com/kevinmsong/DeepDTA-iBAM.git
cd DeepDTA-iBAM
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.4+, RDKit, ESM

## Quick Start

### 1. Download and parse KIBA data

```bash
python download_kaggle_kiba.py
python parse_kiba.py
```

### 2. Build caches (graph and protein embeddings)

```bash
python build_caches.py --device cuda
```

### 3. Train a model

```bash
python train_rmse.py \
  --profile max_rmse_cluster_diffusion \
  --ensemble-size 1 \
  --split-mode standard \
  --device cuda
```

### 4. Evaluate

```bash
python evaluate_best_model.py \
  --profile max_rmse_cluster_diffusion \
  --member-count 1 \
  --data-split standard \
  --eval-split both \
  --device cuda
```

### 5. Reproduce all manuscript results

```bash
python case_studies_results_generation.py \
  --profile max_rmse_cluster_diffusion \
  --member-count 1 \
  --sections ibam fishing generation interpolation ablation diagnostics benchmark manuscript \
  --results-dir results \
  --device cuda
```

This generates all figures (300 dpi PNG + PDF), tables (CSV + LaTeX), captions, metrics, and the manuscript source files.

## Repository Structure

```
DeepDTA-iBAM/
├── train_rmse.py                         # Main training entrypoint
├── evaluate_best_model.py                # Checkpoint evaluation
├── case_studies_results_generation.py    # Publication asset generation
├── config_profiles.py                    # Experiment configuration profiles
├── build_caches.py                       # Cache preprocessing
├── download_kaggle_kiba.py               # KIBA dataset downloader
├── parse_kiba.py                         # KIBA parser utility
├── run_ablations.py                      # Ablation study runner
├── run_generation_validation.py          # Generation validation
├── run_interpretability_benchmark.py     # Interpretability analysis
├── reproduce_all.py                      # Full reproducibility orchestrator
├── aggregate_ablations.py                # Ablation result aggregation
├── requirements.txt                      # Python dependencies
├── models/
│   └── rmse_model.py                     # DeepDTA-iBAM architecture
├── training/
│   ├── engine.py                         # Training loop
│   ├── inference.py                      # Inference pipeline
│   └── checkpoints.py                    # Checkpoint I/O (SafeTensors)
├── data/
│   ├── datasets.py                       # Dataset classes
│   ├── cache_builders.py                 # Graph & protein caching
│   └── splits.py                         # Scaffold and standard splits
├── utils/
│   ├── metrics.py                        # CI, RMSE, MAE, AUROC, BEDROC
│   └── features.py                       # Feature engineering
├── tests/                                # Test suite
└── results/                              # Publication artifacts
    ├── fig*.png / fig*.pdf               # Manuscript figures (300 dpi)
    ├── table*.csv / table*.tex           # Manuscript tables
    ├── *_caption.txt                     # Figure and table captions
    ├── case_study_metrics.json           # Aggregated metrics
    └── source_manifest.json              # Provenance tracking
```

## Configuration Profiles

Defined in [`config_profiles.py`](config_profiles.py):

| Profile | Description |
|---------|-------------|
| `max_rmse_cluster_diffusion` | Full integrated model with diffusion head |
| `max_rmse_cluster` | Affinity-only model (no diffusion) |
| `max_rmse_cluster_no_fusion` | Ablation: no cross-attention fusion |
| `diffusion_egfr_seed` | EGFR-conditioned generation configuration |
| `inference` | Minimal config for deployment |

## Data

The repository expects KIBA CSV files in `data/raw/`. Raw data, cached embeddings, model checkpoints, and the ZINC archive are not included in this repository due to size constraints. They can be obtained or regenerated using the documented workflow above.

## Testing

```bash
python -m pytest -q tests/
```

## Citation

If you use DeepDTA-iBAM in your research, please cite:

```bibtex
@article{song2026deepdta_ibam,
  title   = {DeepDTA-iBAM: Interpretable Cross-Attention for Affinity Prediction, Target-Conditioned Retrieval, and Generative Drug Design},
  author  = {Song, Kevin M.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Acknowledgements

This study was supported in part by the National Heart, Lung, and Blood Institute under grant numbers U01HL134764, P01 HL160476, R01HL131017, and R01HL149137.

## License

[MIT License](LICENSE)
