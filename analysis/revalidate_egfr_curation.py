"""Audit saved EGFR curation cohorts without replacing archived data.

Recompute consolidation sensitivity and molecular properties, identify the
membership differences behind the 5,942 and 6,060 counts, and check what the
saved evidence supports. No network requests or model inference are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
OUT = RES / 'egfr_curation_revalidated'
RDLogger.DisableLog('rdApp.*')
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PROPS = ['MW', 'cLogP', 'HBD', 'HBA', 'RotB', 'charge']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def properties(frame, id_column, smiles_column):
    rows = []
    for ident, smi in zip(frame[id_column], frame[smiles_column]):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, ident
        rows.append([ident, Descriptors.MolWt(mol), Crippen.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol), Chem.GetFormalCharge(mol)])
    return pd.DataFrame(rows, columns=['id'] + PROPS)


def main():
    OUT.mkdir(exist_ok=True)
    names = ['egfr_activity_records_raw.csv', 'egfr_assay_actives.csv',
        'egfr_assay_panel.csv', 'zinc_decoy_pool.csv', 'egfr_interpolation_family.csv']
    report = {'inputs': {n: {'sha256': sha(RES/n)} for n in names},
        'software': {'python': platform.python_version(), 'pandas': pd.__version__,
            'numpy': np.__version__, 'rdkit': rdBase.rdkitVersion},
        'network_used': False}
    raw = pd.read_csv(RES/names[0])
    exploratory = pd.read_csv(RES/names[1])
    panel = pd.read_csv(RES/names[2])
    pool = pd.read_csv(RES/names[3])
    family = pd.read_csv(RES/names[4])
    raw['key'] = raw.parent_molecule_chembl_id.fillna(raw.molecule_chembl_id)
    raw['type_upper'] = raw.standard_type.str.upper()
    raw['type_rank'] = raw.type_upper.map({'KD': 0, 'KI': 1, 'IC50': 2})
    assert raw.type_rank.notna().all()
    assert raw.standard_relation.eq('=').all()
    assert raw.pchembl_value.notna().all()
    cur = raw.sort_values(['key', 'pchembl_value', 'standard_value'],
        ascending=[True, False, True], kind='stable').drop_duplicates('key').set_index('key')
    pri = raw.sort_values(['key', 'type_rank', 'pchembl_value', 'standard_value'],
        ascending=[True, True, False, True], kind='stable').drop_duplicates('key').set_index('key')
    delta = pri.pchembl_value - cur.pchembl_value
    cur_set = set(cur.index[cur.pchembl_value >= 7])
    pri_set = set(pri.index[pri.pchembl_value >= 7])
    sensitivity = dict(n_records=len(raw), n_pairs=len(cur),
        record_type_counts=raw.type_upper.value_counts().to_dict(),
        n_pairs_multi_type=int((raw.groupby('key').type_upper.nunique()>1).sum()),
        n_pairs_type_changed=int((cur.type_upper != pri.type_upper).sum()),
        n_pairs_value_changed=int((delta.abs()>1e-9).sum()),
        max_abs_pchembl_shift=float(delta.abs().max()),
        n_potent_current_rule=len(cur_set), n_potent_priority_rule=len(pri_set),
        n_lost_under_priority=len(cur_set-pri_set), n_gained_under_priority=len(pri_set-cur_set),
        jaccard_potent_sets=len(cur_set&pri_set)/len(cur_set|pri_set))
    old = json.loads((RES/'assay_type_hierarchy_summary.json').read_text())
    assert all(sensitivity[k] == old[k] for k in sensitivity)
    report['consolidation'] = sensitivity
    cur.loc[sorted(cur_set-pri_set)].reset_index().to_csv(OUT/'lost_under_priority.csv', index=False)

    # Explicit molecule and parent identifiers account for different grouping.
    raw_molecule_max = raw.groupby('molecule_chembl_id').pchembl_value.max()
    potent_mol = set(raw_molecule_max.index[raw_molecule_max>=7])
    exp_ids = set(exploratory.molecule_chembl_id)
    extra = sorted(exp_ids - potent_mol)
    missing = sorted(potent_mol - exp_ids)
    assert not missing
    assert len(exploratory) == len(exp_ids) == exploratory.smiles.nunique()
    assert exploratory.pchembl.ge(7).all()
    rows = exploratory.set_index('molecule_chembl_id').loc[extra].copy()
    rows['raw_cache_max_pchembl'] = raw_molecule_max.reindex(extra)
    rows['reason_relative_to_raw_cache'] = np.where(rows.raw_cache_max_pchembl.isna(),
        'molecule absent from raw cache', 'raw cache maximum below 7')
    rows.reset_index().to_csv(OUT/'exploratory_additional_molecules.csv', index=False)
    shared = exploratory.set_index('molecule_chembl_id').loc[sorted(potent_mol)]
    report['cohort_comparison'] = dict(exploratory_rows=len(exploratory),
        exploratory_unique_molecule_ids=len(exp_ids),
        raw_cache_potent_molecule_ids=len(potent_mol),
        raw_cache_potent_parent_ids=len(cur_set),
        molecule_to_parent_count_reduction=len(potent_mol)-len(cur_set),
        exploratory_extra_molecule_ids=len(extra),
        extras_absent_from_raw_cache=int(rows.raw_cache_max_pchembl.isna().sum()),
        extras_with_raw_cache_potency_below_7=int(rows.raw_cache_max_pchembl.notna().sum()),
        shared_molecules_with_different_max_pchembl=int((shared.pchembl-raw_molecule_max.reindex(shared.index)).abs().gt(1e-9).sum()),
        interpretation='The additional molecules and differing grouping keys explain the count arithmetic. The historical query or database-version cause of differing records is not established by these files.')

    print('Recomputing properties of saved exploratory actives and ZINC pool...', flush=True)
    ap = properties(exploratory, 'molecule_chembl_id', 'smiles')
    zp = properties(pool, 'zinc_id', 'smiles')
    ap.to_csv(OUT/'exploratory_active_properties.csv', index=False)
    zp.to_csv(OUT/'zinc_pool_properties.csv', index=False)
    ranges = zp[PROPS].agg(['min','max']).to_dict()
    within = ap[PROPS].ge(zp[PROPS].min()).all(axis=1) & ap[PROPS].le(zp[PROPS].max()).all(axis=1)
    report['property_audit'] = dict(n_actives=len(ap), n_zinc_pool=len(zp),
        active_means=ap[PROPS].mean().to_dict(), zinc_means=zp[PROPS].mean().to_dict(),
        zinc_ranges=ranges, n_actives_in_six_descriptor_minmax_envelope=int(within.sum()),
        ids_in_envelope=ap.loc[within,'id'].tolist(),
        saved_matched_panel_counts=panel.role.value_counts().to_dict(),
        saved_matched_active_ids=panel.loc[panel.role.eq('active'),'ident'].tolist(),
        interpretation='The saved matched panel demonstrates its own composition, but the historical matching code and precise envelope criteria are not available in the retained scripts. No new matching is claimed.')

    # Reapply the archived fingerprint rule to each consolidation of THIS cache.
    # This is not assumed to reconstruct the original 347-molecule query sample.
    refs = [FP.GetFingerprint(Chem.MolFromSmiles(cur.loc[k,'canonical_smiles']))
        for k in ['CHEMBL939','CHEMBL553']]
    def filtered(df, keys):
        keep = set()
        for k in sorted(keys):
            mol = Chem.MolFromSmiles(df.loc[k,'canonical_smiles'])
            if mol is not None and max(DataStructs.TanimotoSimilarity(FP.GetFingerprint(mol), f) for f in refs) >= .4:
                keep.add(k)
        return keep
    fc, fp = filtered(cur, cur_set), filtered(pri, pri_set)
    released = set(family.parent_molecule_chembl_id)
    report['same_cache_similarity_sensitivity'] = dict(n_current=len(fc), n_priority=len(fp),
        n_lost=len(fc-fp), n_gained=len(fp-fc), lost_ids=sorted(fc-fp),
        jaccard=len(fc&fp)/len(fc|fp), n_released_family=len(released),
        n_shared_with_released=len(fc&released),
        released_only=sorted(released-fc), cache_only=sorted(fc-released),
        interpretation='Both rules use one fixed cached record set and the same reference fingerprints. The original family builder fetched at most four 1,000-record pages; this sensitivity cache contains 19,527 records. Differences in extraction history remain, so this is a sensitivity analysis on the cache, not a rerun of original retrieval predictions.')
    (RES/'egfr_curation_revalidated.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs','same_cache_similarity_sensitivity']}, indent=2), flush=True)
    print('similarity current/priority/released:',len(fc),len(fp),len(released), flush=True)


if __name__ == '__main__':
    main()
