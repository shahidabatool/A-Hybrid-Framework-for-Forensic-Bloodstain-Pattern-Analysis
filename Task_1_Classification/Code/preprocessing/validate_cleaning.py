# =============================================================================
# BEFORE AND AFTER IMAGE CLEANING VALIDATION VISUALIZATION
#
# This script generates visual comparisons between original bloodstain images
# and their cleaned versions after preprocessing to validate the effectiveness
# of the image cleaning pipeline before CNN model training.
#
# For each bloodstain class, representative images are selected from the
# organized raw dataset and compared against the corresponding cleaned images.
# The comparison highlights the effect of blood region extraction, background
# removal, ROI cropping, and image normalization.
#
# Images are converted from BGR to RGB format for visualization using
# Matplotlib. The generated figures provide a qualitative assessment of
# preprocessing quality and help verify that important stain characteristics
# are preserved while removing irrelevant background information.
#
# The before-and-after comparison figures are saved in the Evaluation folder
# and are used for preprocessing validation and documentation.
# =============================================================================
import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
ORGANIZED_DIR = os.path.join(BASE_DIR, "Data", "Organized")
CLEANED_DIR = os.path.join(BASE_DIR, "Data", "Cleaned")
EVAL_DIR = os.path.join(BASE_DIR, "Evaluation", "before_after_comparison")

def generate_before_after_plots():
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(CLEANED_DIR) if os.path.isdir(os.path.join(CLEANED_DIR, d))]
    
    print("Generating Before-and-After Cleaning Validation Figures...")
    for cls in sorted(classes):
        cls_org = os.path.join(ORGANIZED_DIR, cls)
        cls_cln = os.path.join(CLEANED_DIR, cls)
        
        # Get list of clean images
        clean_images = [f for f in os.listdir(cls_cln) if f.lower().endswith(('.jpg', '.png', '.tiff'))]
        if not clean_images:
            print(f"  No images found for {cls}, skipping.")
            continue
            
        # Select 2 representative images
        # We sort them so we consistently select the same validation images
        if cls == "Gunshot":
            selected_images = ["HP_1.jpg_tile_33.png", "C4.jpg_tile_13.png"]
        else:
            selected_images = sorted(clean_images)[:2]
        
        # Plot size and layout (2 rows of comparisons, each with Before and After)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Before and After Preprocessing: {cls.replace('_', ' ')}", fontsize=14, fontweight='bold', y=0.98)
        
        for idx, img_name in enumerate(selected_images):
            # Load original image and cleaned image
            org_path = os.path.join(cls_org, img_name)
            cln_path = os.path.join(cls_cln, img_name)
            
            img_org = cv2.imread(org_path)
            img_cln = cv2.imread(cln_path)
            
            if img_org is None or img_cln is None:
                print(f"  Warning: Could not read {img_name} from organized or cleaned folders.")
                continue
                
            # Convert BGR to RGB for matplotlib plotting
            img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            img_cln_rgb = cv2.cvtColor(img_cln, cv2.COLOR_BGR2RGB)
            
            # Plot original image in column 0
            axes[idx, 0].imshow(img_org_rgb)
            axes[idx, 0].set_title(f"Original: {img_name}", fontsize=10, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Plot cleaned image in column 1
            axes[idx, 1].imshow(img_cln_rgb)
            axes[idx, 1].set_title(f"Cleaned (ROI Cropped & Background Masked)", fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(EVAL_DIR, f"{cls}_cleaning_validation.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved comparison figure for {cls} to {save_path}")

if __name__ == "__main__":
    generate_before_after_plots()
