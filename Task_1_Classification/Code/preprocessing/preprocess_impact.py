import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
SRC_DIR = os.path.join(BASE_DIR, "Data", "Organized", "Impact_Spatter")
DST_DIR = os.path.join(BASE_DIR, "Data", "Cleaned", "Impact_Spatter")

def process_massive_image(img, filename, dst_dir):
    """Tiles a massive image, checks for valid tiles, selects at most 50 of them, cleans, and saves."""
    h, w = img.shape[:2]
    crop_size = 1024
    
    # Relaxed HSV double hue red range for dried low-saturation bloodstains
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 20])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([155, 40, 20])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    blood_mask = cv2.bitwise_or(mask1, mask2)
    
    # Collect coordinates of all valid tiles
    valid_coords = []
    for y in range(0, h - crop_size + 1, crop_size):
        for x in range(0, w - crop_size + 1, crop_size):
            tile_mask = blood_mask[y:y+crop_size, x:x+crop_size]
            blood_pixels = np.sum(tile_mask > 0)
            if blood_pixels > 500:
                valid_coords.append((y, x))
                
    if not valid_coords:
        cleaned = detect_blood_and_crop(img)
        if cleaned is not None:
            cv2.imwrite(os.path.join(dst_dir, f"{filename}_fallback.png"), cleaned)
            return True
        return False
        
    # Limit to max 50 tiles randomly per massive card to avoid folder bloating
    max_tiles = 50
    if len(valid_coords) > max_tiles:
        random.seed(42)  # Set seed for reproducibility per card
        valid_coords = random.sample(valid_coords, max_tiles)
        
    idx = 0
    saved_count = 0
    for y, x in valid_coords:
        tile_mask = blood_mask[y:y+crop_size, x:x+crop_size]
        tile_img = img[y:y+crop_size, x:x+crop_size]
        
        contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept_mask = np.zeros(tile_mask.shape, dtype=np.uint8)
        has_points = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                cv2.drawContours(kept_mask, [cnt], -1, 255, -1)
                has_points = True
                
        if has_points and np.sum(kept_mask > 0) > 300:
            white_bg = np.full(tile_img.shape, 255, dtype=np.uint8)
            white_bg[kept_mask > 0] = tile_img[kept_mask > 0]
            
            tile_name = f"{filename}_tile_{idx}.png"
            cv2.imwrite(os.path.join(dst_dir, tile_name), white_bg)
            idx += 1
            saved_count += 1
            
    if saved_count == 0:
        cleaned = detect_blood_and_crop(img)
        if cleaned is not None:
            cv2.imwrite(os.path.join(dst_dir, f"{filename}_fallback.png"), cleaned)
            return True
        return False
        
    return True

def detect_blood_and_crop(img):
    """For small images: relaxed HSV color masking and crop to ROI on white background."""
    h, w = img.shape[:2]
    max_dim = 2048
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 20])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([155, 40, 20])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    blood_mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 10
    kept_mask = np.zeros(blood_mask.shape, dtype=np.uint8)
    
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
        
    pad = 30
    x_start = max(0, x_min - pad)
    y_start = max(0, y_min - pad)
    x_end = min(img.shape[1], x_max + pad)
    y_end = min(img.shape[0], y_max + pad)
    
    cropped_img = img[y_start:y_end, x_start:x_end]
    cropped_mask = kept_mask[y_start:y_end, x_start:x_end]
    
    white_bg = np.full(cropped_img.shape, 255, dtype=np.uint8)
    white_bg[cropped_mask > 0] = cropped_img[cropped_mask > 0]
    return white_bg

def process_single_image(args):
    src_path, dst_path, img_name, dst_dir = args
    img = cv2.imread(src_path)
    if img is None:
        return False
        
    h, w = img.shape[:2]
    if h > 2048 * 2 or w > 2048 * 2:
        return process_massive_image(img, img_name, dst_dir)
    else:
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
    images = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.png', '.tiff', '.jpeg'))]
    
    print(f"Cleaning Impact Spatter: processing {len(images)} images...")
    
    tasks = []
    for img_name in images:
        tasks.append((os.path.join(SRC_DIR, img_name), os.path.join(DST_DIR, img_name), img_name, DST_DIR))
        
    num_workers = min(3, max(1, multiprocessing.cpu_count()))
    saved_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_single_image, tasks)
        for res in results:
            if res:
                saved_count += 1
                
    print(f"Finished Impact_Spatter: {saved_count}/{len(images)} images kept (some tiled into multiple).")

if __name__ == "__main__":
    main()
