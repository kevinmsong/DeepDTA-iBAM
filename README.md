# DeepDTA-iBAM

**DeepDTA-iBAM: Cross-Attention Interaction Mapping for Affinity Prediction, Ligand Retrieval, and Analog Generation**

DeepDTA-iBAM is a multimodal deep learning framework that combines drug-target affinity (DTA) prediction, attention-based interaction mapping, target-conditioned ligand retrieval, and seeded molecular design within a single architecture. The model combines graph-based ligand encoding, cached ESM-C protein embeddings, and bidirectional atom-residue cross-attention with a diffusion auxiliary head for target-conditioned molecular generation.

This release includes the [main manuscript and supplement](IEEE_Access_Submission/),
analysis code and saved results, and a [minimal CPU reference](reference/README.md).
The reference reuses the production implementation. Original model weights and
protein caches are not included, and no public download location is provided.
An explicit untrained demonstration is available without those assets; its
outputs are not affinity predictions.

## Architecture

DeepDTA-iBAM has five main components:

1. **Ligand encoder** — Multi-head graph attention over atom-bond graphs with edge-feature bias (78 atom features, 12 bond features)
2. **Protein adapter** — Learned projection of cached ESM-C residue embeddings into the shared fusion space
3. **Bidirectional cross-attention**: Atom-to-residue and residue-to-atom attention producing interaction maps; these are not validated physical contacts or causal explanations
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

### Minimal reference

```bash
python -m pip install -r reference/requirements.txt
python reference/run_demo.py --mode random --output-dir tmp/reference_random
```

This verifies the full production architecture's tensor flow with random weights
and synthetic protein embeddings. If the original checkpoint, metadata, and
protein cache are available locally, run trained inference instead:

```bash
python reference/run_demo.py --output-dir tmp/reference_trained
```

The default target is the shortest cached sequence, not necessarily EGFR.
See [reference/README.md](reference/README.md) for target selection, required
assets, output formats, and the distinction between trained and untrained modes.

### Full research workflow

The following commands require the broader dependencies and external data.
Training produces new weights; it does not recover the original checkpoint's
missing training history or guarantee the published numerical results.

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

This generates the original case-study figures, tables (CSV + LaTeX), captions and metrics.
The manuscript figures are built separately by `analysis/make_figures_ieee.py` at 600 dpi.

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
├── reference/                            # Minimal CPU entrypoint and asset documentation
├── IEEE_Access_Submission/                # Current manuscript, supplement, and supporting sources
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
├── analysis/                             # Reproduces every reported statistic
│   ├── analysis_revision.py              # Meta-analysis, panel separation, ablation and generator tests
│   ├── revalidate_contact_probes.py      # Nested ridge and fold-local descriptor adjustment
│   ├── export_residue_level.py           # Per-residue attention and contact export
│   ├── analysis_power.py                 # Power derivation for the localization panel
│   ├── audit_generated_set.py            # Analog-set audit and scaffold diversity
│   ├── build_dude_egfr_panel.py          # In-domain assay-defined EGFR panel
│   ├── score_kinase_panel.py             # Model and fingerprint scoring of that panel
│   ├── analysis_literature_audit.py      # Audit of reporting practice in published work
│   ├── literature_audit.csv              # Per-paper verdicts with PMC identifiers
│   └── make_figures_ieee.py             # Current manuscript figures
└── results/                              # Derived artifacts
    ├── fig*.pdf / fig*.png               # Figures (vector where available)
    ├── interpretability_benchmark.csv    # Per-complex localization metrics
    ├── interpretability_residue_level.csv# 1,385 residues: attention and contact labels
    ├── dude_egfr_panel_scored.csv        # In-domain panel, both rankers per candidate
    ├── egfr_assay_actives.csv            # 6,060 unfiltered ChEMBL EGFR actives
    ├── generated_egfr_analogs_audited.csv# Analogs surviving the structural audit
    ├── case_study_metrics.json           # Aggregated metrics
    └── source_manifest.json              # Provenance tracking
```

The current manuscript sources are in `IEEE_Access_Submission/`. Analysis scripts
resolve paths relative to their own location. Their required inputs must be
present; some audits use external checkpoint, structure, or cache assets:

```bash
# Benchmarks and statistics
python analysis/analysis_benchmark_metrics.py    # KIBA regression and classification metrics
python analysis/analysis_power.py                # conditional panel-level power; stratified fields are withdrawn
python analysis/analysis_overlap_chance.py       # chance-referenced top-k contact overlap
python analysis/revalidate_contact_probes.py      # current nested/fold-local probe validation
python analysis/analysis_attention_ablation.py   # inference-time ablation of the fusion gates
python analysis/analysis_efficiency.py           # CPU latency, throughput and parameter profile
python analysis/revalidate_egfr_curation.py       # curation counts, property ranges, and hierarchy sensitivity
python analysis/revalidate_docking_aggregation.py # tie-aware retrieval metrics from saved docking scores
python analysis/analysis_literature_audit.py     # reporting practice in published work
python analysis/audit_generated_set.py           # analog-set audit and scaffold diversity

# Figures and manuscript checks
python analysis/make_figures_ieee.py             # current main-text figures, 600 dpi PDF + PNG
python analysis/make_figure_ibam_map.py          # the interaction-map figure
python analysis/make_figure_generation.py        # the analog-generation figure
python analysis/make_figure_residuals.py         # supplementary residual diagnostics
python analysis/make_figure_scaffold_ablation.py # the supplementary ablation figure
python analysis/check_colorblind.py              # color-vision-deficiency and greyscale audit
python analysis/verify_manuscript_numbers.py     # every headline value against its artifact
python analysis/audit_captions.py <manuscript.tex>
python analysis/wordcount_tex.py <manuscript.tex>
```

The original `analysis_contact_information.py` is retained only to reproduce
exploratory estimates with non-nested penalty selection and full-panel
residualization. Use `revalidate_contact_probes.py` for current estimates.
The binding-mode analyses in `analysis_mixed_effects.py`, the legacy forest
plot in `make_figures.py`, and mode annotations in `analysis_revision.py` were
withdrawn because 4RJ3 was mislabeled. They are not current biological evidence.

Scripts that run model inference require the checkpoint and cached embeddings,
which are not redistributed. These include `export_residue_level.py`,
`score_kinase_panel.py`, `analysis_attention_ablation.py`,
`analysis_efficiency.py`, and `audit_checkpoint_provenance.py`.
`export_residue_level.py` regenerates residue-level maps, and
`score_kinase_panel.py` rescores the in-domain panel.
`export_panel_attention.py` regenerates `results/panel_attention_residue.npy`,
the 3,300 x 1,210 attention profile matrix, which is released here so the
interaction-map analysis can be rerun without a GPU.

`export_residue_level.py --dump-matrix` additionally writes the full
atom-by-residue interaction map per complex to `results/ibam_matrix_{pdb_id}.npz`,
which is what the interaction-map figure is drawn from. That script is
deterministic within one environment but not across them: rerunning the released
checkpoint on a CPU workstation reproduces the released per-residue attention to
approximately 1.1e-3 in absolute attention weight. The 1KE6 revalidation records
residue AUROC 0.6056 versus archived 0.6042. The manuscript retains the archived
residue results and separately reports the corrected graph-order atom AUROC.
See `results/atom_contacts_revalidated.*`; results from different runs should
not be silently substituted.

Scripts that run inference size their batch from the longest target sequence
and the memory available, via `analysis/memory_guard.py`, because memory is
governed by target length rather than by library size.

### Docking

The docking comparison uses AutoDock Vina 1.2.7, which is a third-party binary
and is not redistributed here. Download it from
<https://github.com/ccsb-scripps/AutoDock-Vina/releases> and place the
executable at `tools/vina` (`tools/vina.exe` on Windows). Receptor preparation
and the search box are released:

```bash
python analysis/prep_docking_receptor.py   # writes results/docking/receptor_4wkq.pdbqt
                                           # and results/docking/docking_box.json
python analysis/analysis_docking_baseline.py --all-actives --decoy-limit 460 --workers 14
python analysis/revalidate_docking_aggregation.py
```

Vina regenerates its grid maps from the receptor and box on each run; the maps
themselves are 146 MB and are not committed. Per-ligand scores from the run
reported in the manuscript are in `results/docking/vina_scores.csv`, so the
retrieval analysis can be reproduced without re-docking.

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

The full workflow expects KIBA CSV files in `data/raw/`. Raw KIBA data, cached
embeddings, model checkpoints, and the ZINC archive are not included. The
preprocessing scripts build new caches from available source data. Original
trained weights have no public download link here; training new weights is not
equivalent to reproducing the audited checkpoint. The saved numerical artifacts
and targeted revalidation reports document what can be checked without retraining.
Audit file hashes identify the local input bytes used in each run. Git may
normalize line endings in historical CSV files across platforms; the validation
scripts check parsed records and numerical values as well as recording hashes.

## Testing

```bash
python -m pytest -q tests/
```

## Citation

If you use DeepDTA-iBAM in your research, please cite:

```bibtex
@article{song2026deepdta_ibam,
  title   = {DeepDTA-iBAM: Cross-Attention Interaction Mapping for Affinity Prediction, Ligand Retrieval, and Analog Generation},
  author  = {Song, Kevin and Zhang, John and Ye, Lei and Zhang, Jianyi},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Acknowledgments

This study was supported in part by the National Heart, Lung, and Blood Institute under grant numbers U01HL134764, P01 HL160476, R01HL131017, and R01HL149137.

The authors acknowledge the University of Alabama at Birmingham IT Research Computing group for high-performance computing support and CPU/GPU time on the Cheaha compute cluster, which was used for model training and evaluation in this study.

## License

[MIT License](LICENSE) for the project software. The official IEEE template
assets retain their original ownership and terms; see
[template provenance](IEEE_Access_Submission/TEMPLATE_SOURCE.md).
