# =============================================================================
# BPA DATASET DISTRIBUTION AUDIT
#
# This script performs a quality-control audit of the bloodstain pattern
# classification dataset before model training and evaluation. It analyzes the
# number of samples available in each class, calculates class distribution
# percentages, and checks for potential class imbalance.
#
# The audit is performed on:
#   1. Original organized dataset before augmentation.
#   2. Parent-split augmented training dataset to verify class balancing.
#   3. Clean unaugmented test dataset to confirm natural evaluation
#      distribution.
#
# The imbalance analysis identifies classes with insufficient representation
# that may negatively affect model learning and evaluation reliability.
# This ensures that dataset preparation, augmentation, and splitting steps have
# produced a suitable and consistent dataset for CNN-based bloodstain pattern
# classification.
# =============================================================================
import os

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
ORGANIZED_DIR = os.path.join(BASE_DIR, "Data", "Organized")
AUGMENTED_DIR = os.path.join(BASE_DIR, "Data", "Augmented")

def audit_data(data_dir, label="Data"):
    if not os.path.exists(data_dir):
        print(f"\n[!] Directory not found: {data_dir}")
        return

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"\n--- Audit for {label} ---")
    print(f"{'Class':<20} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 45)
    
    counts = {}
    total = 0
    for cls in sorted(classes):
        cls_path = os.path.join(data_dir, cls)
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f)) and not f.startswith(".")])
        counts[cls] = count
        total += count
    
    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{cls:<20} | {count:<10} | {percentage:>9.2f}%")
        
    print("-" * 45)
    print(f"{'Total':<20} | {total:<10} | 100.00%")
    
    # Check for imbalance
    print("\nImbalance Check:")
    if total > 0:
        for cls in sorted(counts.keys()):
            count = counts[cls]
            if count < (total * 0.05):
                print(f"[!] WARNING: Class '{cls}' is severely underrepresented (< 5%).")
            elif count < (total * 0.10):
                print(f"[-] NOTE: Class '{cls}' is underrepresented (< 10%).")
            else:
                print(f"[+] OK: Class '{cls}' has sufficient representation.")

if __name__ == "__main__":
    audit_data(ORGANIZED_DIR, "Original Organized Data")
    audit_data(os.path.join(AUGMENTED_DIR, "train"), "New Parent-Split TRAIN Set (Balanced)")
    audit_data(os.path.join(AUGMENTED_DIR, "test"), "New Clean TEST Set (Unaugmented & Natural Distribution)")
