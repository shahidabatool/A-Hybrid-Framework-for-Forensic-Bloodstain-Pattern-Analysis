# =============================================================================
# BPA DATASET ORGANIZATION AND FRAME EXTRACTION
#
# This script consolidates bloodstain pattern images from multiple public
# datasets into a unified directory structure for CNN-based classification.
# Static image datasets are copied into their corresponding class folders,
# while video datasets are processed by extracting representative frames at
# regular intervals.
#
# The script supports multiple data sources, including Kaggle, Zenodo,
# HemoSpat, and Mendeley, and maps each sample to one of the five bloodstain
# pattern classes (Gunshot, Impact Spatter, Passive Drip, Transfer Wipe, and
# Cough Spatter). Files are renamed where necessary to prevent filename
# collisions and preserve source information.
#
# Before processing, previously organized files are removed to ensure a clean
# and reproducible dataset preparation pipeline.
# =============================================================================
import cv2
import os
import shutil

# Paths
BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
RAW_DATA_DIR = os.path.join(BASE_DIR, "Text")
ORGANIZED_DIR = os.path.join(BASE_DIR, "Data", "Organized")

# Mendeley mapping (Prefix to Class)
MENDELEY_MAP = {
    "15": "Cough_Spatter",
    "10": "Transfer_Wipe",
    "1": "Passive_Drip",
    "3": "Passive_Drip"
}

def extract_frames(video_path, output_folder, target_count=150):
    """Extracts a target number of frames from a video."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Calculate sampling rate to get roughly target_count frames
    sampling_rate = max(1, total_frames // target_count)
    
    count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it matches sampling rate and we haven't exceeded target
        if count % sampling_rate == 0 and saved_count < target_count:
            frame_name = f"{video_name}_frame_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_count += 1
        
        count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_name}")

def organize_static_data():
    """Copies Kaggle, Zenodo, and HemoSpat images to organized folders."""
    # Kaggle -> Gunshot
    kaggle_dir = os.path.join(RAW_DATA_DIR, "1_BPA_Scans_Teaching_Kaggle")
    for root, dirs, files in os.walk(kaggle_dir):
        for file in files:
            if file.endswith(".jpg"):
                # Use source folder name to avoid collision (guns vs rifles)
                source_type = "guns" if "experiments_with_g" in root else "rifles"
                new_name = f"kaggle_{source_type}_{file}"
                shutil.copy(os.path.join(root, file), os.path.join(ORGANIZED_DIR, "Gunshot", new_name))
    
    # Zenodo -> Impact_Spatter
    zenodo_dir = os.path.join(RAW_DATA_DIR, "5_High_Res_Impact_Spatter_Zenodo")
    for file in os.listdir(zenodo_dir):
        if file.endswith(".tiff") or file.endswith(".png"):
            shutil.copy(os.path.join(zenodo_dir, file), os.path.join(ORGANIZED_DIR, "Impact_Spatter", f"zenodo_{file}"))
            
    # HemoSpat -> Impact_Spatter
    hemospat_dir = os.path.join(RAW_DATA_DIR, "SupplementaryMaterials")
    for root, dirs, files in os.walk(hemospat_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                # Include Pattern folder and subfolder to avoid collision
                parts = root.split(os.sep)
                pattern_folder = next((p for p in parts if "Pattern" in p), "unknown")
                sub_folder = parts[-1]
                new_name = f"hemospat_{pattern_folder}_{sub_folder}_{file}"
                shutil.copy(os.path.join(root, file), os.path.join(ORGANIZED_DIR, "Impact_Spatter", new_name))

def process_mendeley():
    """Processes Mendeley videos and extracts frames."""
    mendeley_dir = os.path.join(RAW_DATA_DIR, "4_Blood_Stain_Pattern_Videos_Mendeley")
    for file in os.listdir(mendeley_dir):
        if file.endswith(".avi"):
            prefix = file.split(" ")[0][:2] if file[0].isdigit() else ""
            class_key = "".join([c for c in prefix if c.isdigit()])
            
            target_class = MENDELEY_MAP.get(class_key, "Unknown")
            if target_class != "Unknown":
                video_path = os.path.join(mendeley_dir, file)
                output_folder = os.path.join(ORGANIZED_DIR, target_class)
                extract_frames(video_path, output_folder, target_count=150)

if __name__ == "__main__":
    # Clear organized folders first to ensure clean run
    for folder in os.listdir(ORGANIZED_DIR):
        path = os.path.join(ORGANIZED_DIR, folder)
        if os.path.isdir(path):
            for f in os.listdir(path):
                file_path = os.path.join(path, f)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

    print("Organizing static data...")
    organize_static_data()
    print("Processing Mendeley videos...")
    process_mendeley()
    print("Done!")
