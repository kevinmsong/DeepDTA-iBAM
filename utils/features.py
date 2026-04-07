"""Sequence-encoding utilities for SMILES strings and protein sequences.

Provides integer encoding (character → index) with configurable padding
and truncation, plus an RDKit-based SMILES-to-graph converter and a
Morgan fingerprint helper.

Classes
-------
FeatureEncoder    Base character-to-integer encoder.
SMILESEncoder     Encoder pre-configured for the SMILES character set.
ProteinEncoder    Encoder pre-configured for the amino-acid alphabet.

Functions
---------
encode_smiles              Convenience wrapper around ``SMILESEncoder``.
encode_protein             Convenience wrapper around ``ProteinEncoder``.
smiles_to_graph            Lightweight SMILES → graph dict (node features + edge index).
compute_morgan_fingerprint Bit-vector Morgan fingerprint via RDKit.
"""

import numpy as np
from typing import List, Tuple


# Character sets for encoding
SMILES_CHARSET = ['#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u']

PROTEIN_CHARSET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']


class FeatureEncoder:
    """Base encoder: maps characters to integer indices with padding/truncation."""
    
    def __init__(self, charset, max_length):
        self.charset = charset
        self.max_length = max_length
        self.char_to_int = {c: i for i, c in enumerate(charset)}
        self.int_to_char = {i: c for i, c in enumerate(charset)}
        self.vocab_size = len(charset)
    
    def encode(self, sequence: str, padding='post') -> np.ndarray:
        """Encode a character sequence to an integer array.

        Args:
            sequence: Input string.
            padding: ``'post'`` (default) or ``'pre'`` padding.

        Returns:
            Integer array of shape ``(max_length,)``.
        """
        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Encode characters (unknown chars map to 0)
        encoded = []
        for char in sequence:
            if char in self.char_to_int:
                encoded.append(self.char_to_int[char])
            else:
                encoded.append(0)
        
        # Pad to max_length
        pad_length = self.max_length - len(encoded)
        if padding == 'post':
            encoded = encoded + [0] * pad_length
        else:
            encoded = [0] * pad_length + encoded
        
        return np.array(encoded, dtype=np.int32)
    
    def decode(self, encoded: np.ndarray) -> str:
        """Decode an integer array back to a character string."""
        return ''.join([self.int_to_char.get(i, '') for i in encoded if i > 0])
    
    def batch_encode(self, sequences: List[str], padding='post') -> np.ndarray:
        """Encode a batch of sequences into a 2-D integer array."""
        return np.array([self.encode(seq, padding) for seq in sequences])


class SMILESEncoder(FeatureEncoder):
    """Integer encoder for SMILES strings (default max length 100)."""
    
    def __init__(self, max_length=100):
        super().__init__(SMILES_CHARSET, max_length)


class ProteinEncoder(FeatureEncoder):
    """Integer encoder for amino-acid sequences (default max length 1000)."""
    
    def __init__(self, max_length=1000):
        super().__init__(PROTEIN_CHARSET, max_length)


def encode_smiles(smiles: str, max_length: int = 100) -> np.ndarray:
    """Encode a SMILES string to an integer array.

    Args:
        smiles: SMILES string.
        max_length: Maximum length for padding / truncation.

    Returns:
        Integer array of shape ``(max_length,)``.
    """
    encoder = SMILESEncoder(max_length)
    return encoder.encode(smiles)


def encode_protein(sequence: str, max_length: int = 1000) -> np.ndarray:
    """Encode a protein sequence to an integer array.

    Args:
        sequence: Amino-acid sequence string.
        max_length: Maximum length for padding / truncation.

    Returns:
        Integer array of shape ``(max_length,)``.
    """
    encoder = ProteinEncoder(max_length)
    return encoder.encode(sequence)


def smiles_to_graph(smiles: str):
    """Convert a SMILES string to a basic graph dict (node features + edge index).

    This is a lightweight alternative to ``models.rmse_model.smiles_to_graph``;
    the model-level version produces the full 78-atom + 12-edge feature tensors
    used during training.  This version is mainly useful for quick inspection.

    Returns ``None`` if the SMILES cannot be parsed or RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Node features (atoms)
        num_atoms = mol.GetNumAtoms()
        node_features = []
        
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
            ]
            node_features.append(features)
        
        # Edge index
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # Add reverse edge
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]  # Self-loop for single atom
        
        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_index': np.array(edge_index, dtype=np.int64).T,
            'num_nodes': num_atoms,
        }
    
    except ImportError:
        # RDKit not available, return None
        return None


def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute a Morgan (circular) fingerprint for a SMILES string.

    Args:
        smiles: SMILES string.
        radius: Fingerprint radius (default 2 ≈ ECFP4).
        n_bits: Bit-vector length.

    Returns:
        Binary array of shape ``(n_bits,)``.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    
    except ImportError:
        # RDKit not available
        return np.zeros(n_bits, dtype=np.float32)


if __name__ == "__main__":
    # Test encoding
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    protein = "MKTIIALSYIFCLVFA"
    
    print("Testing SMILES encoding...")
    encoded_smiles = encode_smiles(smiles, max_length=100)
    print("Encoded shape: {}".format(encoded_smiles.shape))
    print("Sample values: {}".format(encoded_smiles[:10]))
    
    print("\nTesting protein encoding...")
    encoded_protein = encode_protein(protein, max_length=1000)
    print("Encoded shape: {}".format(encoded_protein.shape))
    print("Sample values: {}".format(encoded_protein[:10]))
