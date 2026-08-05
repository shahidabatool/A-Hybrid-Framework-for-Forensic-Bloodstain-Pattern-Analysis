# =============================================================================
# COUGH SPATTER IMAGE CLEANING AND PREPROCESSING
#
# This script preprocesses the Cough Spatter images by automatically detecting
# bloodstain regions and removing irrelevant background information. Bloodstains
# are identified using HSV color segmentation, while skin-colored regions and
# non-blood objects are suppressed using YCrCb skin detection and additional
# color-based filtering.
#
# After detecting valid bloodstain contours, the script computes a bounding box,
# applies padding, crops the region of interest, and replaces the remaining
# background with a uniform white background. Images that do not contain a valid
# bloodstain are discarded.
#
# To improve processing efficiency, large images are downsampled before
# analysis, and multiple images are processed in parallel using multiprocessing.
# The cleaned images are saved for subsequent augmentation, dataset splitting,
# and CNN model training.


# NOTE:
# Although the Cough Spatter class was excluded from the final model evaluation
# due to parent-level data leakage arising from a single source video, this
# preprocessing script is retained as part of the complete data preparation
# pipeline. It may be reused in future work if additional independent Cough
# Spatter data become available, allowing the class to be incorporated into the
# classification model without leakage concerns.
# =============================================================================
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
SRC_DIR = os.path.join(BASE_DIR, "Data", "Organized", "Cough_Spatter")
DST_DIR = os.path.join(BASE_DIR, "Data", "Cleaned", "Cough_Spatter")

def detect_blood_and_crop(img):
    if img is None:
        return None
    
    # Downsample extremely large images to a max dimension of 2048 to speed up processing
    h, w = img.shape[:2]
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red blood ranges (Standard for camera/video captures)
    lower_red1 = np.array([0, 40, 20])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 40, 20])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    blood_mask = cv2.bitwise_or(mask1, mask2)
    
    # Skin color ranges in YCrCb to filter out hands/forearms
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    base_mask = cv2.bitwise_and(blood_mask, cv2.bitwise_not(skin_mask))
    
    # Pixel-level color constraint to remove neutral/black gloves in camera photos
    diff_rg = cv2.subtract(img[:, :, 2], img[:, :, 1])
    diff_rb = cv2.subtract(img[:, :, 2], img[:, :, 0])
    color_mask = (diff_rg > 15) & (diff_rb > 15)
    base_mask = cv2.bitwise_and(base_mask, base_mask, mask=color_mask.astype(np.uint8))
            
    # Find contours
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 10
    kept_mask = np.zeros(base_mask.shape, dtype=np.uint8)
    
    # Find global bounding box of all valid contours
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    has_points = False
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(kept_mask, [cnt], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            has_points = True
                
    if not has_points:
        return None
        
    # Add crop padding
    pad = 30
    x_start = max(0, x_min - pad)
    y_start = max(0, y_min - pad)
    x_end = min(img.shape[1], x_max + pad)
    y_end = min(img.shape[0], y_max + pad)
    
    # Crop and mask out non-blood background pixels to solid white
    cropped_img = img[y_start:y_end, x_start:x_end]
    cropped_mask = kept_mask[y_start:y_end, x_start:x_end]
    
    white_bg = np.full(cropped_img.shape, 255, dtype=np.uint8)
    white_bg[cropped_mask > 0] = cropped_img[cropped_mask > 0]
    
    return white_bg

def process_single_image(args):
    src_path, dst_path = args
    img = cv2.imread(src_path)
    if img is None:
        return False
        
    cleaned = detect_blood_and_crop(img)
    if cleaned is not None:
        cv2.imwrite(dst_path, cleaned)
        return True
    return False

def main():
    if not os.path.exists(SRC_DIR):
        print(f"[!] Source folder not found: {SRC_DIR}")
        return
        
    os.makedirs(DST_DIR, exist_ok=True)
    images = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Cleaning Cough Spatter: processing {len(images)} images...")
    
    tasks = []
    for img_name in images:
        tasks.append((os.path.join(SRC_DIR, img_name), os.path.join(DST_DIR, img_name)))
        
    num_workers = min(3, max(1, multiprocessing.cpu_count()))
    saved_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_single_image, tasks)
        for res in results:
            if res:
                saved_count += 1
                
    print(f"Finished Cough_Spatter: {saved_count}/{len(images)} images kept.")

if __name__ == "__main__":
    main()
