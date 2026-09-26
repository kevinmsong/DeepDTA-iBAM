# Minimal reference

This CPU entrypoint uses the production model in `models/rmse_model.py`, the
`max_rmse_cluster_diffusion` configuration, the production graph builder, and
the strict checkpoint loader. It scores one ligand and exports its
atom-by-residue attention map. It does not train a model or download assets.

## Install

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -r reference/requirements.txt
```

The entrypoint needs PyTorch, NumPy, pandas, RDKit, SafeTensors, and tqdm.
ESM-C is not required when using cached embeddings. The repository's broader
training and analysis workflows have additional dependencies in the root
`requirements.txt` and their script imports.

## Trained checkpoint inference

The original weights and protein caches are **not included in Git**, and this
repository does not currently provide a public download location for them.
Trained inference requires these existing assets:

| Asset | Default path |
|---|---|
| Model weights | `checkpoints/max_rmse_cluster_diffusion_member_0.safetensors` |
| Weight metadata and target normalization | Same basename with `.json` |
| Cached ESM-C 600M embeddings | `data/cache/proteins.pt` |
| Cache manifest containing sequences and keys | `data/cache/proteins_manifest.json` |

If these assets are available locally:

```bash
python reference/run_demo.py --output-dir tmp/reference_trained
```

The default ligand is gefitinib. The default target is the **shortest sequence
in the supplied cache**, which is not necessarily EGFR. Select a specific
target using its key under `items` in the protein manifest:

```bash
python reference/run_demo.py --protein-key YOUR_CACHE_KEY --smiles "CCO" --output-dir tmp/reference_pair
```

Use `--checkpoint`, `--protein-cache`, and `--protein-manifest` for other local
asset locations. Only the full diffusion configuration is supported. The JSON
reports whether the weights match the audited manuscript checkpoint:

```text
369be983160a49950c44ba9fcfe2334a8fd0a3c5ab084a27aff3f38aad203213
```

The audited optimizer metadata identifies epoch 47; the separate training
summary ends at epoch 44 and does not establish the checkpoint-selection
history. New training runs are not guaranteed to reproduce those weights.
See `results/checkpoint_provenance_audit.md` for the audit and numerical checks.

## Untrained tensor demonstration

A fresh clone can exercise the production architecture without private assets:

```bash
python reference/run_demo.py --mode random --output-dir tmp/reference_random
```

This explicit mode uses random model weights and 16 synthetic protein-embedding
rows. Its raw scalar is labeled **not an affinity estimate**. It checks the
forward pass and map dimensions only. Missing trained assets never silently
trigger this mode.

## Outputs and interpretation

The command prints JSON. `--output-dir` additionally writes:

- `summary.json`: mode, input selection, parameter count, map dimensions, and,
  in trained mode, asset hashes and predicted affinity in KIBA units.
- `interaction_map.npz`: `atom_to_residue` and its atom-averaged
  `residue_profile`. Attention is averaged over all heads and fusion blocks;
  padding is removed.

Scores use cached-target CPU float32 inference. Different cached embeddings or
numerical environments can change outputs slightly. KIBA predictions are not
experimental affinities. Attention weights are internal model couplings;
they are not validated physical contacts or causal explanations.

## Ranking and generation

- `analysis/score_kinase_panel.py` scores the saved EGFR candidate panel using
  the same affinity model and reports a fingerprint reference.
- `run_generation_validation.py --help` describes the comparison of diffusion
  proposals, fragment swaps, and random atom edits.
- `case_studies_results_generation.py` contains the latent-retrieval and seeded
  analog-generation workflows. Generation uses the pooled protein adapter
  before fusion and preserves the seed graph's topology.

These workflows require additional data and dependencies. They do not become
available by running the untrained demonstration. Saved manuscript results can
be inspected without the original weights; `analysis/revalidate_contact_probes.py`
revalidates the released attention profiles using nested ridge selection and
training-fold descriptor adjustment, with the broader analysis dependencies.

## Smoke checks

```bash
python -m pip install pytest
python -m pytest -q tests/test_reference_demo.py
```

The tests verify explicit untrained labeling, failure without trained assets,
invalid-SMILES rejection, and correct masked attention aggregation. An additional
trained smoke check runs only when the original weights and caches exist locally.
