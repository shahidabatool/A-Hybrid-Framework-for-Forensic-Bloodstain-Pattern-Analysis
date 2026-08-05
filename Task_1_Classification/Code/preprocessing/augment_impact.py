# =============================================================================
# IMPACT SPATTER DATASET PREPARATION
#
# This script prepares the Impact Spatter dataset by grouping images according
# to their original parent source to prevent data leakage between training and
# testing sets. An 80/20 split is performed at the parent-image level, ensuring
# that tiles or related images from the same source are not present in both
# subsets. Training images are augmented using Albumentations until the target
# size is reached, while test images are sampled with a per-parent limit to
# maintain balanced evaluation.
# =============================================================================
import os
import cv2
import numpy as np
import random
import shutil
import re
import albumentations as A

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
CLEANED_DIR = os.path.join(BASE_DIR, "Data", "Cleaned", "Impact_Spatter")
AUGMENTED_DIR = os.path.join(BASE_DIR, "Data", "Augmented")
TRAIN_DST = os.path.join(AUGMENTED_DIR, "train", "Impact_Spatter")
TEST_DST = os.path.join(AUGMENTED_DIR, "test", "Impact_Spatter")

TRAIN_TARGET = 1000
TEST_CAP_PER_PARENT = 25

aug_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.GaussNoise(p=0.3),
])

def get_parent_id(filename):
    if "hemospat" in filename.lower():
        match = re.match(r"(hemospat_Pattern\d+)", filename)
        if match:
            return match.group(1)
        return "hemospat_unknown"
    if "zenodo" in filename.lower():
        # Strip indexing like (1), (2), (3) and the tile/stitch suffix
        base = re.sub(r"\(\d+\)", "", filename)
        base = base.split("_tile_")[0]
        base = base.split("_stitch")[0]
        return base
    if "_tile_" in filename:
        return filename.split("_tile_")[0]
    return filename

def main():
    random.seed(42)
    
    os.makedirs(TRAIN_DST, exist_ok=True)
    os.makedirs(TEST_DST, exist_ok=True)
    
    # Clean output directories
    for f in os.listdir(TRAIN_DST):
        os.unlink(os.path.join(TRAIN_DST, f))
    for f in os.listdir(TEST_DST):
        os.unlink(os.path.join(TEST_DST, f))
        
    images = [f for f in os.listdir(CLEANED_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print("[!] No cleaned Impact Spatter images found.")
        return
        
    print(f"Impact Spatter: splitting and augmenting {len(images)} clean images/tiles...")
    
    # Group by parent card
    parent_groups = {}
    for img_name in images:
        pid = get_parent_id(img_name)
        if pid not in parent_groups:
            parent_groups[pid] = []
        parent_groups[pid].append(img_name)
        
    parent_ids = list(parent_groups.keys())
    random.shuffle(parent_ids)
    
    split_idx = int(len(parent_ids) * 0.8)
    if split_idx == 0:
        split_idx = 1
    if split_idx == len(parent_ids):
        split_idx = len(parent_ids) - 1
        
    train_parents = parent_ids[:split_idx]
    test_parents = parent_ids[split_idx:]
    
    train_list = []
    for pid in train_parents:
        train_list.extend(parent_groups[pid])
        
    print(f"  Parent Split: {len(train_parents)} parents for Train, {len(test_parents)} parents for Test")
    print(f"  Initial Split: {len(train_list)} train images, {len(images) - len(train_list)} test images")
    
    # Generate Train Set (Balanced to TRAIN_TARGET)
    current_count = 0
    # Add original/pre-tiled files first
    random.shuffle(train_list)
    for img_name in train_list:
        if current_count >= TRAIN_TARGET: break
        shutil.copy(os.path.join(CLEANED_DIR, img_name), os.path.join(TRAIN_DST, f"train_{current_count}_orig_{img_name}"))
        current_count += 1
        
    # Generate augmented fillers
    while current_count < TRAIN_TARGET:
        img_name = random.choice(train_list)
        img = cv2.imread(os.path.join(CLEANED_DIR, img_name))
        if img is None: continue
        
        augmented = aug_pipeline(image=img)["image"]
        cv2.imwrite(os.path.join(TRAIN_DST, f"train_{current_count}_aug_{img_name}"), augmented)
        current_count += 1
        
    # Generate Test Set (With Cap per Parent to prevent imbalance)
    test_count = 0
    for pid in test_parents:
        pid_images = parent_groups[pid]
        sampled_images = pid_images
        if len(pid_images) > TEST_CAP_PER_PARENT:
            sampled_images = random.sample(pid_images, TEST_CAP_PER_PARENT)
            
        for img_name in sampled_images:
            shutil.copy(os.path.join(CLEANED_DIR, img_name), os.path.join(TEST_DST, f"test_capped_{img_name}"))
            test_count += 1
            
    print(f"  Impact Spatter Completed: Saved {current_count} train images and {test_count} test images.")

if __name__ == "__main__":
    main()
