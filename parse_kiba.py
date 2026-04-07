"""Parse the KIBA dataset from a pre-downloaded Kaggle text file.

Expects a space-separated file with columns:
    DRUG_ID  TARGET_ID  SMILES  PROTEIN_SEQUENCE  AFFINITY

Splits the data 90/5/5 (train/val/test) and writes CSVs to ``data/raw/``.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_kiba():
    """Parse the space-separated KIBA file and write train/val/test CSVs.

    Reads ``data/kaggle/kiba.txt`` (columns: DRUG_ID, TARGET_ID, SMILES,
    PROTEIN_SEQUENCE, AFFINITY), splits 90/5/5, and writes CSVs to
    ``data/raw/``.

    Returns:
        Tuple of ``(train_df, val_df, test_df)`` DataFrames, or
        ``(None, None, None)`` if the source file is missing.
    """
    print("="*70)
    print("PARSING REAL KIBA DATASET")
    print("="*70)
    
    kiba_file = 'data/kaggle/kiba.txt'
    
    if not os.path.exists(kiba_file):
        print(f"✗ File not found: {kiba_file}")
        return None, None, None
    
    print(f"\nReading {kiba_file}...")
    
    # Read as space-separated with 5 columns
    data = []
    with open(kiba_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 5:
                drug_id = parts[0]
                target_id = parts[1]
                smiles = parts[2]
                protein = parts[3]
                affinity = parts[4]
                
                data.append({
                    'compound_iso_smiles': smiles,
                    'target_sequence': protein,
                    'affinity': float(affinity)
                })
            
            if (i + 1) % 20000 == 0:
                print(f"  Processed {i+1:,} lines...")
    
    df = pd.DataFrame(data)
    print(f"\n✓ Loaded {len(df):,} drug-target pairs")
    print(f"  Unique SMILES: {df['compound_iso_smiles'].nunique():,}")
    print(f"  Unique proteins: {df['target_sequence'].nunique():,}")
    
    # Split 90/5/5
    print("\n" + "="*70)
    print("SPLITTING: 90% train / 5% val / 5% test")
    print("="*70)
    
    os.makedirs('data/raw', exist_ok=True)
    
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=4221, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=4221, shuffle=True)
    
    # Save
    train_df.to_csv('data/raw/train_kiba.csv', index=False)
    val_df.to_csv('data/raw/val_kiba.csv', index=False)
    test_df.to_csv('data/raw/test_kiba.csv', index=False)
    
    print(f"\n✓ Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"✓ Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Statistics
    print("\n" + "="*70)
    print("REAL KIBA DATASET STATISTICS")
    print("="*70)
    print(f"Total: {len(df):,}")
    print(f"Unique drugs (SMILES): {df['compound_iso_smiles'].nunique():,}")
    print(f"Unique proteins: {df['target_sequence'].nunique():,}")
    print(f"\nAffinity:")
    print(f"  Mean: {df['affinity'].mean():.2f}")
    print(f"  Std: {df['affinity'].std():.2f}")
    print(f"  Range: [{df['affinity'].min():.2f}, {df['affinity'].max():.2f}]")
    print(f"  Median: {df['affinity'].median():.2f}")
    
    # Verify it's real data
    print("\n" + "="*70)
    print("DATA VERIFICATION")
    print("="*70)
    print(f"Sample SMILES (first 3):")
    for s in df['compound_iso_smiles'].unique()[:3]:
        print(f"  {s[:80]}...")
    print(f"\nProtein length stats:")
    print(f"  Min: {df['target_sequence'].str.len().min()}")
    print(f"  Max: {df['target_sequence'].str.len().max()}")
    print(f"  Mean: {df['target_sequence'].str.len().mean():.0f}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = parse_kiba()
    
    if train_df is not None:
        print("\n" + "="*70)
        print("✓ REAL KIBA DATASET READY!")
        print("="*70)
        print("\nSample training data (first 2 rows):")
        pd.set_option('display.max_colwidth', 50)
        print(train_df.head(2))
        print("\n✓ Files saved to data/raw/")
