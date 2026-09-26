# Atom-contact revalidation

All element/bond-preserving PDB-to-cached-graph mappings; no attention-based mapping selection.

Original files and cached tensors were unchanged (SHA-256 verified).
Only 1KE6 required inference; the other four atom-label vectors are single-class.
All 1,385 residue labels, including 100 contacts, match the released labels.

| PDB | Mappings | Label vectors | Archived atom AUROC | Corrected AUROC range | Corrected top-k overlap range |
|---|---:|---:|---:|---|---|
| 1KE6 | 4 | 1 | 0.192308 | 0.269231 to 0.269231 | 0.961538 to 0.961538 |
| 2HYY | 4 | 1 | 0.000000 | undefined | 1.000000 to 1.000000 |
| 4RJ3 | 4 | 1 | 0.000000 | undefined | 1.000000 to 1.000000 |
| 4WKQ | 2 | 1 | 0.000000 | undefined | 1.000000 to 1.000000 |
| 6YOJ | 12 | 1 | 0.000000 | undefined | 1.000000 to 1.000000 |

Archived zero AUROCs in saturated cases are numerical fallbacks, not measurements.
The JSON records every atom mapping, symmetry-related label vector, and conditional panel result.
The NPZ retains the new forward-pass attention arrays, enabling metric recomputation without inference.

## Panel overlap and chance reference

```json
[
  {
    "atom_overlap_mean": 0.9923076923076923,
    "atom_base_rate_mean": 0.9925925925925926,
    "atom_excess_mean": -0.0002849002849002691,
    "atom_excess_ci95": [
      -0.0010759102863810795,
      0.0005061097165805412
    ]
  }
]
```
