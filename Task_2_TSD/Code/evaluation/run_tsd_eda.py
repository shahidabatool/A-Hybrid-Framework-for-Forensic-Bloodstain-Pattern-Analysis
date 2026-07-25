import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def clean_weight(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().replace('kg', '').strip()
    try:
        return float(val_str)
    except:
        return np.nan

def clean_age(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().replace('days', '').strip()
    try:
        return float(val_str)
    except:
        return np.nan

def main():
    print("=== Task 2 Time Since Deposition (TSD) EDA ===")
    
    # Paths
    base_dir = "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Text/BloodNet_50k_Images"
    csv_path = os.path.join(base_dir, "bloodstain_information.csv")
    data_dir = os.path.join(base_dir, "data")
    out_dir = "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Evaluation/eda_plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows.")
    
    # Clean rabbit variables
    df['rabbit_weight_numeric'] = df['rabbit_ weight'].apply(clean_weight)
    df['rabbit_age_numeric'] = df['rabbit_ age'].apply(clean_age)
    
    # 1. Rabbit-Level Split Verification Table
    print("\nRabbit ID distribution across groups:")
    rabbit_split = pd.crosstab(df['rabbit_ID'], df['group'])
    print(rabbit_split)
    
    # Save statistics
    stats_file = os.path.join(out_dir, "tsd_descriptive_statistics.txt")
    with open(stats_file, "w") as f:
        f.write("=== Task 2: Time Since Deposition (TSD) Descriptive Statistics ===\n\n")
        f.write("1. Metadata Shape:\n")
        f.write(f"Total Rows: {df.shape[0]}\nColumns: {list(df.columns)}\n\n")
        
        f.write("2. Split and Subject (Rabbit) Breakdown:\n")
        f.write(rabbit_split.to_string())
        f.write("\n\n")
        
        f.write("3. Rabbit Demographic Summaries (Unique Rabbits Only):\n")
        unique_rabbits = df.drop_duplicates(subset=['rabbit_ID'])
        f.write(f"Total unique rabbits: {len(unique_rabbits)}\n")
        f.write(f"Gender distribution:\n{unique_rabbits['rabbit_ gender'].value_counts().to_string()}\n\n")
        f.write(f"Weight (kg) - Mean: {unique_rabbits['rabbit_weight_numeric'].mean():.2f}, Std: {unique_rabbits['rabbit_weight_numeric'].std():.2f}, Min: {unique_rabbits['rabbit_weight_numeric'].min():.2f}, Max: {unique_rabbits['rabbit_weight_numeric'].max():.2f}\n")
        f.write(f"Age (days) - Mean: {unique_rabbits['rabbit_age_numeric'].mean():.2f}, Std: {unique_rabbits['rabbit_age_numeric'].std():.2f}, Min: {unique_rabbits['rabbit_age_numeric'].min():.2f}, Max: {unique_rabbits['rabbit_age_numeric'].max():.2f}\n\n")
        
        f.write("4. Drying Time Interval (TSD) Counts by Split:\n")
        tsd_split = pd.crosstab(df['TSD'], df['group'])
        f.write(tsd_split.to_string())
        f.write("\n")
        
    print(f"Descriptive statistics written to {stats_file}")

    # Set styles
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Split and Rabbit Breakdown
    plt.figure(figsize=(10, 6))
    rabbit_split_plot = rabbit_split.plot(kind='bar', stacked=True, color=['#4682B4', '#E68A8A', '#1B365D'], edgecolor='black', ax=plt.gca())
    plt.title("Distribution of Samples by Rabbit ID and Dataset Split", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Rabbit Subject ID", fontsize=12, labelpad=10)
    plt.ylabel("Sample Count", fontsize=12, labelpad=10)
    plt.legend(title="Dataset Split", labels=["Dev (Validation) Set", "Test Set (Holdout)", "Train Set"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rabbit_split_distribution.png"), dpi=300)
    plt.close()
    
    # Plot 2: TSD Interval counts
    plt.figure(figsize=(10, 6))
    tsd_counts = df['TSD'].value_counts()
    # Sort logically
    order = ['0-1d', '1-3d', '5-7d', '7-8d', '13-14d', '14-15d', '20-21d', '21-22d', '27-28d', '28-29d']
    order = [x for x in order if x in tsd_counts.index]
    sns.barplot(x=tsd_counts.loc[order].index, y=tsd_counts.loc[order].values, hue=tsd_counts.loc[order].index, palette="viridis", legend=False, edgecolor='black')
    plt.title("Time Since Deposition (TSD) Interval Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("TSD Class Interval", fontsize=12, labelpad=10)
    plt.ylabel("Image Count", fontsize=12, labelpad=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsd_class_distribution.png"), dpi=300)
    plt.close()

    # 3. Analyze Color Channel Degradation Trend (Physics/Chemical Validation)
    print("\nAnalyzing color channel degradation across drying times...")
    # We will sample 100 images per category in train1 and calculate their mean R, G, B channels
    train1_dir = os.path.join(data_dir, "train1")
    categories = ['1d_train', '7d_train', '14d_train', '21d_train', '28d_train']
    days_mapped = [1, 7, 14, 21, 28]
    
    color_stats = []
    
    for cat, days in zip(categories, days_mapped):
        cat_path = os.path.join(train1_dir, cat)
        if os.path.exists(cat_path):
            files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Sample up to 150 files for speed and representation
            sampled_files = np.random.choice(files, min(len(files), 150), replace=False)
            
            for file_name in sampled_files:
                img_path = os.path.join(cat_path, file_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # OpenCV loads BGR, convert to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Compute mean of non-background pixels (we assume background is black/white)
                    # Let's check: are images white-background or black-background?
                    # Let's count non-white pixels (where R, G, B are not all > 240)
                    # or non-black pixels (R, G, B > 15)
                    # Let's assume blood stains are dark red/brown, and background is white/light
                    # Let's find blood pixels (e.g. R > 20 and R > G + 10)
                    mask = (img_rgb[:, :, 0] > 10) & (img_rgb[:, :, 0] > img_rgb[:, :, 1])
                    if np.sum(mask) > 10:
                        blood_pixels = img_rgb[mask]
                        mean_r = np.mean(blood_pixels[:, 0])
                        mean_g = np.mean(blood_pixels[:, 1])
                        mean_b = np.mean(blood_pixels[:, 2])
                        ratio_rg = mean_r / (mean_g + 1e-5)
                        ratio_rb = mean_r / (mean_b + 1e-5)
                        
                        color_stats.append({
                            'days': days,
                            'R': mean_r,
                            'G': mean_g,
                            'B': mean_b,
                            'RG_ratio': ratio_rg,
                            'RB_ratio': ratio_rb
                        })
                        
    color_df = pd.DataFrame(color_stats)
    
    # Group by days and compute average channels
    grouped_color = color_df.groupby('days').mean().reset_index()
    print("\nMean color values across drying times:")
    print(grouped_color)
    
    # Save color stats
    with open(stats_file, "a") as f:
        f.write("\n\n5. Color Analysis of Bloodstains Over Time (RGB Mean of Blood Pixels):\n")
        f.write(grouped_color.to_string())
        f.write("\n")
        
    # Plot 3: RGB Intensities Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_color['days'], grouped_color['R'], marker='o', color='red', linewidth=2, label='Red Channel')
    plt.plot(grouped_color['days'], grouped_color['G'], marker='s', color='green', linewidth=2, label='Green Channel')
    plt.plot(grouped_color['days'], grouped_color['B'], marker='^', color='blue', linewidth=2, label='Blue Channel')
    plt.title("RGB Channel Intensities vs. Bloodstain Drying Time (TSD)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time Since Deposition (Days)", fontsize=12, labelpad=10)
    plt.ylabel("Average Pixel Intensity (0-255)", fontsize=12, labelpad=10)
    plt.xticks([1, 7, 14, 21, 28])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rgb_intensities_vs_time.png"), dpi=300)
    plt.close()
    
    # Plot 4: R/G and R/B Ratio (chemical indicators)
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_color['days'], grouped_color['RG_ratio'], marker='o', color='brown', linewidth=2, label='Red/Green Ratio')
    plt.plot(grouped_color['days'], grouped_color['RB_ratio'], marker='x', color='purple', linewidth=2, label='Red/Blue Ratio')
    plt.title("Red-to-Green & Red-to-Blue Ratio vs. Drying Time (TSD)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time Since Deposition (Days)", fontsize=12, labelpad=10)
    plt.ylabel("Intensity Ratio", fontsize=12, labelpad=10)
    plt.xticks([1, 7, 14, 21, 28])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "color_ratios_vs_time.png"), dpi=300)
    plt.close()

    # Create a 5-day visual sample grid
    print("\nCreating drying visual sample grid...")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for idx, (cat, days) in enumerate(zip(categories, days_mapped)):
        cat_path = os.path.join(train1_dir, cat)
        if os.path.exists(cat_path):
            files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Select first file
            img_path = os.path.join(cat_path, files[0])
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(f"{days} Day(s) Old", fontsize=12, fontweight='bold')
                axes[idx].axis('off')
    plt.suptitle("Visual Browning Process of Bloodstains (Hemoglobin Oxidation)", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsd_visual_drying_grid.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTask 2 TSD EDA successfully completed. Plots saved in:", out_dir)

if __name__ == "__main__":
    main()
