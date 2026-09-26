"""Map coordinate-derived ligand contacts to RDKit graph atom order.

Every element/bond-preserving isomorphism is considered. Molecular symmetry
must never be resolved by selecting the mapping with the best attention score.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def ligand_mol_from_pdb(pdb_text, ligand_atoms):
    """Reconstruct the selected coordinate ligand and verify its atom order."""
    serials = [atom["serial"] for atom in ligand_atoms]
    if len(set(serials)) != len(serials):
        raise ValueError("Selected ligand has duplicate PDB atom serials")
    selected = set(serials)
    lines = [line for line in pdb_text.splitlines()
             if line.startswith("HETATM") and int(line[6:11]) in selected]
    bound = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n",
                                 sanitize=False, removeHs=False)
    if bound is None or bound.GetNumAtoms() != len(ligand_atoms):
        raise ValueError("Coordinate ligand could not be reconstructed exactly")
    for i, atom in enumerate(ligand_atoms):
        rd_atom = bound.GetAtomWithIdx(i)
        info = rd_atom.GetPDBResidueInfo()
        position = bound.GetConformer().GetAtomPosition(i)
        if (info is None or info.GetSerialNumber() != atom["serial"]
                or rd_atom.GetSymbol().upper() != atom["element"].upper()
                or not np.allclose([position.x, position.y, position.z],
                                   [atom["x"], atom["y"], atom["z"]],
                                   rtol=0, atol=1e-6)):
            raise ValueError("Coordinate ligand atom order differs from contact-label order")
    return bound


def mapped_contact_vectors(graph, bound, contacts, *, max_matches=100000):
    """Return all PDB-to-graph mappings and distinct graph-order label vectors.

    Chirality constraints are omitted conservatively. The caller must retain
    all resulting label vectors or refuse a scalar metric if they differ.
    """
    contacts = np.asarray(contacts)
    if (contacts.shape != (bound.GetNumAtoms(),)
            or not np.isin(contacts, [0, 1]).all()):
        raise ValueError("Contact labels must be one binary value per coordinate atom")
    # PDB supplies coordinates/connectivity; the graph supplies bond orders.
    # The template assignment may pick one symmetric assignment. Enumerating
    # all isomorphisms below prevents a performance-based mapping selection.
    assigned = AllChem.AssignBondOrdersFromTemplate(graph, bound)
    if assigned.GetNumAtoms() != graph.GetNumAtoms():
        raise ValueError("PDB and graph atom counts differ")
    if assigned.GetNumBonds() != graph.GetNumBonds():
        raise ValueError("PDB and graph bond counts differ")
    matches = graph.GetSubstructMatches(assigned, uniquify=False,
                                        useChirality=False, maxMatches=max_matches)
    if not matches or len(matches) >= max_matches:
        raise ValueError("No complete mapping, or mapping enumeration truncated")
    vectors = {}
    for mapping in matches:
        if len(set(mapping)) != graph.GetNumAtoms():
            raise ValueError("Atom mapping is not bijective")
        for i, j in enumerate(mapping):
            a, b = assigned.GetAtomWithIdx(i), graph.GetAtomWithIdx(j)
            if (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic()) != (
                    b.GetAtomicNum(), b.GetFormalCharge(), b.GetIsAromatic()):
                raise ValueError("Atom chemistry failed mapping verification")
        for bond in assigned.GetBonds():
            mapped = graph.GetBondBetweenAtoms(mapping[bond.GetBeginAtomIdx()],
                                               mapping[bond.GetEndAtomIdx()])
            if mapped is None or (bond.GetBondType(), bond.GetIsAromatic()) != (
                    mapped.GetBondType(), mapped.GetIsAromatic()):
                raise ValueError("Bond chemistry failed mapping verification")
        labels = np.empty(len(contacts), dtype=int)
        labels[np.asarray(mapping)] = contacts
        vectors[tuple(labels.tolist())] = labels
    return list(matches), list(vectors.values())


def unique_graph_contact_vector(graph, bound, contacts):
    """Return invariant labels, or refuse an ambiguous scalar evaluation."""
    mappings, vectors = mapped_contact_vectors(graph, bound, contacts)
    if len(vectors) != 1:
        raise ValueError(
            f"Molecular symmetry yields {len(vectors)} distinct contact-label vectors; "
            "report metrics over all mappings instead of choosing one")
    return vectors[0], mappings
