# Checkpoint provenance audit

## Established

- Weights: `checkpoints/max_rmse_cluster_diffusion_member_0.safetensors`.
- SHA256: `369be983160a49950c44ba9fcfe2334a8fd0a3c5ab084a27aff3f38aad203213`.
- The optimizer archive's metadata exactly matches the epoch-47 JSON sidecar.
- Optimizer steps: [85916]; scheduler last step: 85916.
- Current profile and training CSV reconstruct 1828 batches per epoch, so 47 complete epochs give exactly 85916 optimizer steps. This supports the sidecar's epoch count without recovering the missing history.
- Cache manifest hash matches checkpoint: True.
- All 5,913 archived target values match the test CSV after reconstructing its stable length-sorted inference order.
- Fresh inference: 24 pairs from 24 targets, lengths 215 to 4128; 24 compounds.
- Maximum absolute prediction difference: 0.00680541992 KIBA units; mean absolute difference: 0.00154729684; correlation: 0.999998991718.
- 0/24 differences are at most 0.0001 KIBA units.
- All 5913 archived standard outputs lie exactly on the denormalized bfloat16 grid; fresh CPU arithmetic uses float32.
- A second check uses 5 overlapping gate-ablation pairs and that analysis's own archived cache: maximum prediction difference 9.53674318e-07 KIBA units. The base and ablation protein caches differ numerically, with per-target maximum embedding differences [0.01708984375, 0.01611328125, 0.017578125, 0.01953125, 0.048828125].
- All original input hashes remain unchanged: True.

## Still unresolved

The training summary covers epochs 1 through 44, names epoch 36 as best, and records no resume. The weights file carries no embedded training metadata. The epoch-47 sidecar and optimizer metadata agree, but neither supplies the missing epoch-by-epoch history. No audited artifact links the old summary's best-epoch metrics to the epoch-47 weights. The original resolved configuration is not saved, so its hash cannot be used to recover omitted settings.

This audit establishes current checkpoint identity and prediction reproducibility. It does not independently establish the training chronology, the sidecar's validation metrics, or the original checkpoint-selection decision. Retain general validation-based selection wording and disclose the unreconciled training log.

## Reproduction

Run `python analysis/audit_checkpoint_provenance.py` from the project environment. It reads existing weights and caches, performs 24 CPU forward passes plus 5 corroborating passes using two threads, and writes this report, JSON, and per-pair CSV. It does not retrain or modify original artifacts.
