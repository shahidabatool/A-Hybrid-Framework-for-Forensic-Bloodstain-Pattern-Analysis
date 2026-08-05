# =============================================================================
# COUGH SPATTER DATASET PREPARATION
#
# This script prepares the Cough Spatter dataset by performing a sequential
# 80/20 train-test split based on video frame order to reduce temporal leakage.
# Original training frames are copied first, then augmented using Albumentations
# until the target number of training images is reached. Test images are copied
# without augmentation to ensure unbiased model evaluation.
# =============================================================================
import os
import cv2
import numpy as np
import random
import shutil
import re
import albumentations as A

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
CLEANED_DIR = os.path.join(BASE_DIR, "Data", "Cleaned", "Cough_Spatter")
AUGMENTED_DIR = os.path.join(BASE_DIR, "Data", "Augmented")
TRAIN_DST = os.path.join(AUGMENTED_DIR, "train", "Cough_Spatter")
TEST_DST = os.path.join(AUGMENTED_DIR, "test", "Cough_Spatter")

TRAIN_TARGET = 1000

aug_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.GaussNoise(p=0.3),
])

def extract_frame_index(filename):
    match = re.search(r"_frame_(\d+)\.jpg", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def main():
    random.seed(42)
    
    os.makedirs(TRAIN_DST, exist_ok=True)
    os.makedirs(TEST_DST, exist_ok=True)
    
    # Remove existing files in target folders to avoid duplicates
    for f in os.listdir(TRAIN_DST):
        os.unlink(os.path.join(TRAIN_DST, f))
    for f in os.listdir(TEST_DST):
        os.unlink(os.path.join(TEST_DST, f))
        
    images = [f for f in os.listdir(CLEANED_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print("[!] No cleaned Cough Spatter images found.")
        return
        
    print(f"Cough Spatter: splitting and augmenting {len(images)} frames...")
    
    # Sequential frame split to prevent leakage in video frames
    images.sort(key=extract_frame_index)
    split_idx = int(len(images) * 0.8)
    train_list = images[:split_idx]
    test_list = images[split_idx:]
    
    print(f"  Split: {len(train_list)} frames for Train, {len(test_list)} frames for Test")
    
    # Generate Train Set (Augmented to TRAIN_TARGET)
    current_count = 0
    # Copy original train frames first
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
        
    # Generate Test Set (Clean copy)
    test_count = 0
    for img_name in test_list:
        shutil.copy(os.path.join(CLEANED_DIR, img_name), os.path.join(TEST_DST, f"test_orig_{img_name}"))
        test_count += 1
        
    print(f"  Cough Spatter Completed: Saved {current_count} train images and {test_count} test images.")

if __name__ == "__main__":
    main()
