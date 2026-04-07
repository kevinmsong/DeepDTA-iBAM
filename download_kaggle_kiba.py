"""Download and split the KIBA dataset from Kaggle.

Configures Kaggle API credentials, downloads the
``christang0002/davis-and-kiba`` dataset, identifies the KIBA file,
normalises column names, and writes 90/5/5 train/val/test CSVs to
``data/raw/``.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import subprocess
import sys
import zipfile

def setup_kaggle():
    """Write a placeholder ``~/.kaggle/kaggle.json`` credentials file.

    Replace the dummy API key with your own before running.
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json, 'w') as f:
        f.write('{"username":"_","key":"KGAT_6eefaa24f712e9a04909b2be7fdf425f"}')
    
    # Set permissions (Windows doesn't need this but doesn't hurt)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass
    
    print("✓ Kaggle credentials configured")

def download_kiba():
    """Download the KIBA dataset from Kaggle, normalise columns, and split.

    Downloads ``christang0002/davis-and-kiba``, identifies the KIBA file,
    standardises column names, splits 90/5/5, and writes CSVs to ``data/raw/``.

    Returns:
        Tuple of ``(train_df, val_df, test_df)`` DataFrames, or
        ``(None, None, None)`` on failure.
    """
    print("="*70)
    print("DOWNLOADING REAL KIBA DATASET FROM KAGGLE")
    print("="*70)
    
    # Setup credentials
    setup_kaggle()
    
    # Install kaggle if needed
    print("\nInstalling kaggle CLI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=True)
    
    # Create data directory
    os.makedirs('data/kaggle', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Download dataset
    print("\nDownloading dataset from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "christang0002/davis-and-kiba", "-p", "data/kaggle", "--unzip"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        # Try alternative command
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", "christang0002/davis-and-kiba", "-p", "data/kaggle", "--unzip"],
            capture_output=True,
            text=True
        )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk('data/kaggle'):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath) / 1024 / 1024
            print(f"  {filepath} ({size:.2f} MB)")
    
    # Find and load KIBA file
    kiba_file = None
    for root, dirs, files in os.walk('data/kaggle'):
        for f in files:
            if 'kiba' in f.lower():
                kiba_file = os.path.join(root, f)
                print(f"\n✓ Found KIBA file: {kiba_file}")
                break
    
    if not kiba_file:
        print("✗ KIBA file not found!")
        return None, None, None
    
    # Load and parse KIBA data
    print("\nLoading KIBA data...")
    
    # Try different parsing approaches
    try:
        # First try as regular CSV
        df = pd.read_csv(kiba_file)
        print(f"Loaded as CSV: {len(df)} rows, columns: {list(df.columns)}")
    except:
        # Try space-separated
        df = pd.read_csv(kiba_file, sep=r'\s+', header=None)
        print(f"Loaded as space-separated: {len(df)} rows")
    
    # Check column structure
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # Standardize column names based on content
    if len(df.columns) >= 3:
        # Find columns by content type
        for i, col in enumerate(df.columns):
            sample = str(df[col].iloc[0])
            if len(sample) > 100:  # Likely protein sequence
                df = df.rename(columns={col: 'target_sequence'})
            elif any(c in sample for c in ['C', 'c', '(', ')', '=', '#']) and len(sample) < 500:  # SMILES
                df = df.rename(columns={col: 'compound_iso_smiles'})
            elif sample.replace('.', '').replace('-', '').isdigit():  # Affinity
                df = df.rename(columns={col: 'affinity'})
    
    # Ensure we have required columns
    required = ['compound_iso_smiles', 'target_sequence', 'affinity']
    if not all(col in df.columns for col in required):
        # Manual assignment if needed
        if len(df.columns) == 5:
            df.columns = ['drug_id', 'target_id', 'compound_iso_smiles', 'target_sequence', 'affinity']
        elif len(df.columns) == 3:
            df.columns = ['compound_iso_smiles', 'target_sequence', 'affinity']
    
    # Keep only required columns
    df = df[['compound_iso_smiles', 'target_sequence', 'affinity']]
    
    # Convert affinity to float
    df['affinity'] = pd.to_numeric(df['affinity'], errors='coerce')
    df = df.dropna()
    
    print(f"\n✓ Processed {len(df):,} drug-target pairs")
    print(f"  Unique SMILES: {df['compound_iso_smiles'].nunique():,}")
    print(f"  Unique proteins: {df['target_sequence'].nunique():,}")
    
    # Split 90/5/5
    print("\n" + "="*70)
    print("SPLITTING: 90% train / 5% val / 5% test")
    print("="*70)
    
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=4221, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=4221, shuffle=True)
    
    # Save
    train_df.to_csv('data/raw/train_kiba.csv', index=False)
    val_df.to_csv('data/raw/val_kiba.csv', index=False)
    test_df.to_csv('data/raw/test_kiba.csv', index=False)
    
    print(f"\n✓ Train: {len(train_df):,} samples -> data/raw/train_kiba.csv")
    print(f"✓ Val:   {len(val_df):,} samples -> data/raw/val_kiba.csv")
    print(f"✓ Test:  {len(test_df):,} samples -> data/raw/test_kiba.csv")
    
    # Statistics
    print("\n" + "="*70)
    print("REAL KIBA DATASET STATISTICS")
    print("="*70)
    print(f"Total: {len(df):,}")
    print(f"Unique drugs: {df['compound_iso_smiles'].nunique():,}")
    print(f"Unique proteins: {df['target_sequence'].nunique():,}")
    print(f"Affinity: {df['affinity'].mean():.2f} ± {df['affinity'].std():.2f}")
    print(f"Range: [{df['affinity'].min():.2f}, {df['affinity'].max():.2f}]")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = download_kiba()
    
    if train_df is not None:
        print("\n" + "="*70)
        print("✓ REAL KIBA DATASET READY!")
        print("="*70)
        print("\nSample data:")
        print(train_df.head(2))
