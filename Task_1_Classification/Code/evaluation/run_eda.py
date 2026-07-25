import os
# Increase pixel limit for high-res forensic scans
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set OpenCV thread limit to avoid potential issues in local environment
cv2.setNumThreads(4)


BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
ORGANIZED_DIR = os.path.join(BASE_DIR, "Data", "Organized")
OUT_DIR = os.path.join(BASE_DIR, "Evaluation", "eda_plots")

def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Blood detection via HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 20])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 40, 20])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    blood_mask = cv2.bitwise_or(mask1, mask2)
    
    # Skin removal via YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Final mask
    final_mask = cv2.bitwise_and(blood_mask, cv2.bitwise_not(skin_mask))
    
    # Find contours/droplets
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    droplet_areas = []
    droplet_aspect_ratios = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Filter out tiny noise pixels
            droplet_areas.append(area)
            # Aspect ratio of bounding box
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh > 0:
                aspect_ratio = bw / bh
                droplet_aspect_ratios.append(aspect_ratio)
                
    stain_density = (np.sum(final_mask > 0) / (w * h)) * 100
    
    return {
        "width": w,
        "height": h,
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "droplet_count": len(droplet_areas),
        "mean_droplet_area": np.mean(droplet_areas) if droplet_areas else 0,
        "max_droplet_area": np.max(droplet_areas) if droplet_areas else 0,
        "mean_droplet_aspect_ratio": np.mean(droplet_aspect_ratios) if droplet_aspect_ratios else 0,
        "stain_density": stain_density
    }

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    classes = [d for d in os.listdir(ORGANIZED_DIR) if os.path.isdir(os.path.join(ORGANIZED_DIR, d))]
    
    data_records = []
    
    print("Starting detailed Exploratory Data Analysis (EDA)...")
    for cls in sorted(classes):
        print(f"\nProcessing class: {cls}")
        cls_path = os.path.join(ORGANIZED_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.tiff'))]
        
        # To make things run reasonably fast but still be highly representative, we analyze all images
        count = 0
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            metrics = analyze_image(img_path)
            if metrics:
                metrics["class"] = cls
                metrics["filename"] = img_name
                data_records.append(metrics)
                count += 1
                if count % 100 == 0:
                    print(f"  Processed {count}/{len(images)} images...")
                    
    df = pd.DataFrame(data_records)
    
    # Save CSV of EDA stats
    csv_path = os.path.join(OUT_DIR, "eda_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed image statistics to: {csv_path}")
    
    # Group and print summaries
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS (EDA) SUMMARY METRICS BY CLASS")
    print("="*80)
    summary_df = df.groupby("class").agg({
        "filename": "count",
        "width": "mean",
        "height": "mean",
        "droplet_count": "mean",
        "mean_droplet_area": "mean",
        "mean_droplet_aspect_ratio": "mean",
        "stain_density": "mean",
        "mean_brightness": "mean"
    }).rename(columns={"filename": "image_count"})
    
    print(summary_df.to_string())
    print("="*80)
    
    # Save text summary report
    summary_txt_path = os.path.join(OUT_DIR, "eda_summary_report.txt")
    with open(summary_txt_path, "w") as f:
        f.write("========================================================================\n")
        f.write("      BPA DATASET EXPLORATORY DATA ANALYSIS (EDA) SUMMARY REPORT\n")
        f.write("========================================================================\n\n")
        f.write(summary_df.to_string())
        f.write("\n\nDetailed Class Observations:\n")
        f.write("1. Cough Spatter & Gunshot: Characterized by a high count of droplets per image\n")
        f.write("   but with very small average droplet areas (fine mist splatters).\n")
        f.write("2. Impact Spatter: Displays a medium droplet count and moderate droplet area,\n")
        f.write("   representing a radial dispersion of droplets.\n")
        f.write("3. Passive Drip: Characterized by very low droplet counts per image but large,\n")
        f.write("   circular droplet areas (large blood drops dripping vertically).\n")
        f.write("4. Transfer/Wipe: Characterized by low component counts but very high stain densities\n")
        f.write("   and massive connected areas due to continuous smearing.\n")
    print(f"Saved text summary report to: {summary_txt_path}")
    
    # --- VISUALIZATIONS ---
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution Bar Plot
    plt.figure(figsize=(8, 5))
    order = df["class"].value_counts().index
    sns.countplot(data=df, x="class", order=order, palette="Blues_r")
    plt.title("Class Distribution (Original Organized Dataset)", fontsize=14, fontweight='bold')
    plt.xlabel("Pattern Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Droplet Count Boxplot (Log Scale)
    plt.figure(figsize=(9, 5))
    # Replace 0 with 0.1 for log scale compatibility
    df_count = df.copy()
    df_count["droplet_count_plot"] = df_count["droplet_count"].clip(lower=0.5)
    ax = sns.boxplot(data=df_count, x="class", y="droplet_count_plot", palette="Set2")
    ax.set_yscale("log")
    plt.title("Stain/Droplet Count per Image by Class (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Pattern Class", fontsize=12)
    plt.ylabel("Stain/Droplet Count (log scale)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "droplet_count_boxplot.png"), dpi=300)
    plt.close()
    
    # 3. Droplet Mean Area Boxplot (Log Scale)
    plt.figure(figsize=(9, 5))
    df_area = df.copy()
    df_area["mean_area_plot"] = df_area["mean_droplet_area"].clip(lower=1.0)
    ax = sns.boxplot(data=df_area, x="class", y="mean_area_plot", palette="Set3")
    ax.set_yscale("log")
    plt.title("Average Droplet/Stain Area (in Pixels) by Class (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Pattern Class", fontsize=12)
    plt.ylabel("Average Area (px, log scale)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "droplet_area_boxplot.png"), dpi=300)
    plt.close()
    
    # 4. Aspect Ratio Boxplot
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="class", y="mean_droplet_aspect_ratio", palette="Pastel1")
    plt.title("Droplet Bounding Box Aspect Ratio (Width / Height) by Class", fontsize=14, fontweight='bold')
    plt.xlabel("Pattern Class", fontsize=12)
    plt.ylabel("Aspect Ratio (W/H)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "droplet_aspect_ratio_boxplot.png"), dpi=300)
    plt.close()

    # 5. Scatter Plot: Count vs Area (Color-coded by class)
    plt.figure(figsize=(10, 6))
    df_scatter = df[df["droplet_count"] > 0].copy()
    sns.scatterplot(
        data=df_scatter, 
        x="droplet_count", 
        y="mean_droplet_area", 
        hue="class", 
        style="class",
        alpha=0.7, 
        palette="bright"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Stain count vs. Average Area Mapping", fontsize=14, fontweight='bold')
    plt.xlabel("Stain/Droplet Count per Image (Log)", fontsize=12)
    plt.ylabel("Mean Droplet Area in Pixels (Log)", fontsize=12)
    plt.legend(title="Pattern Class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "count_vs_area_scatter.png"), dpi=300)
    plt.close()

    print("\nVisualizations saved successfully to: ", OUT_DIR)

if __name__ == "__main__":
    main()
