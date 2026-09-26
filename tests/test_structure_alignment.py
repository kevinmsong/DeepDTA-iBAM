"""Regression checks for coordinate-to-graph contacts and undefined metrics."""

import numpy as np
import pytest
from rdkit import Chem

from case_studies_results_generation import compute_structure_alignment_metrics
from utils.structure_alignment import mapped_contact_vectors, unique_graph_contact_vector


def test_contacts_follow_graph_order_after_coordinate_permutation():
    graph = Chem.MolFromSmiles("CCO")
    bound = Chem.RenumberAtoms(graph, [2, 1, 0])
    labels, mappings = unique_graph_contact_vector(graph, bound, [1, 0, 0])
    assert mappings == [(2, 1, 0)]
    np.testing.assert_array_equal(labels, [0, 0, 1])


def test_symmetry_keeps_all_labels_and_refuses_ambiguous_scalar():
    graph = Chem.MolFromSmiles("CC")
    mappings, vectors = mapped_contact_vectors(graph, graph, [1, 0])
    assert len(mappings) == 2
    assert {tuple(v) for v in vectors} == {(1, 0), (0, 1)}
    with pytest.raises(ValueError, match="distinct contact-label vectors"):
        unique_graph_contact_vector(graph, graph, [1, 0])
    labels, mappings = unique_graph_contact_vector(graph, graph, [1, 1])
    assert len(mappings) == 2
    np.testing.assert_array_equal(labels, [1, 1])


def test_single_class_auroc_and_zero_contact_overlap_are_undefined():
    a2r = np.asarray([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    r2a = np.asarray([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    mixed = compute_structure_alignment_metrics(a2r, r2a, np.array([1, 0]),
                                                 np.array([0, 0, 1]))
    assert mixed["atom_contact_auroc"] == 1.0
    assert mixed["atom_topk_overlap"] == 1.0
    saturated = compute_structure_alignment_metrics(a2r, r2a, np.ones(2), np.ones(3))
    assert np.isnan(saturated["atom_contact_auroc"])
    assert np.isnan(saturated["residue_contact_auroc"])
    assert saturated["atom_topk_overlap"] == 1.0
    empty = compute_structure_alignment_metrics(a2r, r2a, np.zeros(2), np.zeros(3))
    assert np.isnan(empty["atom_contact_auroc"])
    assert np.isnan(empty["atom_topk_overlap"])
