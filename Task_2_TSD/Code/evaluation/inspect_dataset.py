# ================================================================
# Description:
# This script performs an initial audit and exploratory data
# analysis (EDA) of the BloodNet dataset used for Task 2 (Time
# Since Deposition) model training and evaluation.
#
# Objective:
# Inspect the raw dataset structure and metadata to verify
# integrity before model development, including class balance,
# missing values, and file counts across data splits.
#
# Checks Performed:
#   1. Metadata CSV inspection (shape, columns, unique values,
#      value counts, missing values)
#   2. File count audit across data splits:
#      train1, test, outside_test (per aging-interval subfolder)
#
# Dataset:
#   Bloodstain images grouped into five deposition intervals:
#   1d, 7d, 14d, 21d, and 28d.
#
# ================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== BloodNet Dataset Audit & EDA ===")
    
    # Paths
    base_dir = "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Text/BloodNet_50k_Images"
    csv_path = os.path.join(base_dir, "bloodstain_information.csv")
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Read CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nCSV Shape: {df.shape}")
        print("\nColumns:", list(df.columns))
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Unique value counts
        print("\nUnique values in key columns:")
        for col in ['group', 'rabbit_ID', 'rabbit_ gender', 'TSD']:
            if col in df.columns:
                print(f"  {col}: {df[col].unique()}")
                print(df[col].value_counts(dropna=False))
                print("-" * 30)
                
        # Missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
    else:
        print(f"CSV not found at: {csv_path}")
        df = None

    # 2. Count files in directories
    print("\nFile Counts in Directories:")
    splits = {
        'train1': os.path.join(data_dir, 'train1'),
        'test': os.path.join(data_dir, 'test'),
        'outside_test': os.path.join(data_dir, 'outside_test')
    }
    
    for split_name, split_path in splits.items():
        if os.path.exists(split_path):
            print(f"\nSplit: {split_name}")
            subdirs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
            total_files = 0
            for subdir in subdirs:
                subdir_path = os.path.join(split_path, subdir)
                files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
                print(f"  {subdir}: {len(files)} files")
                total_files += len(files)
            print(f"Total files in {split_name}: {total_files}")
        else:
            print(f"Directory not found: {split_path}")

if __name__ == "__main__":
    main()
