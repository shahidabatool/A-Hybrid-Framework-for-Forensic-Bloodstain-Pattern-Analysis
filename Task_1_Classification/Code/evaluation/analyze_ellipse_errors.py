import os
import cv2
import numpy as np
import pandas as pd

def calculate_iou(contour, ellipse_geom, img_shape):
    """Calculates the Intersection over Union (IoU) between the contour and the fitted ellipse."""
    # Create mask for the contour
    mask_cnt = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask_cnt, [contour], -1, 255, -1)
    
    # Create mask for the fitted ellipse
    mask_elp = np.zeros(img_shape, dtype=np.uint8)
    cv2.ellipse(mask_elp, ellipse_geom, 255, -1)
    
    # Calculate Intersection and Union
    intersection = cv2.bitwise_and(mask_cnt, mask_elp)
    union = cv2.bitwise_or(mask_cnt, mask_elp)
    
    cnt_inter = np.sum(intersection == 255)
    cnt_union = np.sum(union == 255)
    
    if cnt_union == 0:
        return 0.0
    return cnt_inter / cnt_union

def main():
    print("[+] Starting Quantitative Ellipse-Fitting Error Analysis...")
    BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
    test_dir = os.path.join(BASE_DIR, "Data", "Augmented", "test")
    eval_dir = os.path.join(BASE_DIR, "Evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    classes_to_test = ["Gunshot", "Impact_Spatter", "Passive_Drip"]
    results = []

    for cls_name in classes_to_test:
        cls_folder = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_folder):
            print(f"[-] Test folder for {cls_name} not found.")
            continue
            
        img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Analyzing {len(img_files)} images in class {cls_name}...")
        
        # We sample up to 100 images per class for robust statistics
        np.random.seed(42)
        if len(img_files) > 100:
            img_files = list(np.random.choice(img_files, 100, replace=False))
            
        for img_file in img_files:
            img_path = os.path.join(cls_folder, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold to isolate bloodstains (blood is dark, background is white)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200 and len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (w, h), angle = ellipse
                    
                    minor = min(w, h)
                    major = max(w, h)
                    
                    if major > 0:
                        ratio = minor / major
                        # Area of fitted ellipse: pi * a * b
                        # w and h are diameters, so semi-axes are w/2 and h/2
                        ellipse_area = np.pi * (minor / 2.0) * (major / 2.0)
                        
                        # Calculate error metrics
                        area_diff = abs(area - ellipse_area)
                        area_error_pct = (area_diff / area) * 100.0
                        
                        iou = calculate_iou(cnt, ellipse, gray.shape)
                        
                        theta_rad = np.arcsin(ratio)
                        theta_deg = theta_rad * 180.0 / np.pi
                        
                        results.append({
                            'Class': cls_name,
                            'Filename': img_file,
                            'Contour_Area': area,
                            'Ellipse_Area': ellipse_area,
                            'Area_Error_Pct': area_error_pct,
                            'IoU': iou,
                            'Aspect_Ratio': ratio,
                            'Calculated_Angle': theta_deg
                        })

    # Group and output stats
    df = pd.DataFrame(results)
    output_csv = os.path.join(eval_dir, "ellipse_fitting_error_analysis.csv")
    df.to_csv(output_csv, index=False)
    print(f"[+] Saved raw metrics to {output_csv}")
    
    summary_lines = []
    summary_lines.append("=================================================================")
    summary_lines.append("  BPA TASK 1 — QUANTITATIVE ELLIPSE-FITTING ERROR ANALYSIS")
    summary_lines.append("=================================================================\n")
    
    summary_lines.append("This report evaluates the shape deviation of preprocessed bloodstain contours")
    summary_lines.append("from perfect fitted ellipses using two metrics:")
    summary_lines.append("1. Intersection over Union (IoU): measures shape overlap (1.0 = perfect ellipse).")
    summary_lines.append("2. Area Error (%): measures the percentage difference between contour and ellipse areas.")
    summary_lines.append("")
    
    grouped = df.groupby('Class').agg(
        sample_count=('Filename', 'count'),
        mean_iou=('IoU', 'mean'),
        std_iou=('IoU', 'std'),
        mean_area_error=('Area_Error_Pct', 'mean'),
        std_area_error=('Area_Error_Pct', 'std'),
        mean_angle=('Calculated_Angle', 'mean')
    ).reset_index()
    
    for _, row in grouped.iterrows():
        summary_lines.append(f"Class: {row['Class']}")
        summary_lines.append(f"  Analyzed Droplets  : {row['sample_count']}")
        summary_lines.append(f"  Mean IoU           : {row['mean_iou']:.4f} (±{row['std_iou']:.4f})")
        summary_lines.append(f"  Mean Area Error (%) : {row['mean_area_error']:.2f}% (±{row['std_area_error']:.2f}%)")
        summary_lines.append(f"  Mean Estimated Angle: {row['mean_angle']:.2f}°")
        summary_lines.append("")
        
    overall_iou = df['IoU'].mean()
    overall_area_err = df['Area_Error_Pct'].mean()
    summary_lines.append("-" * 65)
    summary_lines.append(f"OVERALL SUMMARY (All Classes):")
    summary_lines.append(f"  Total Droplets Fitted: {len(df)}")
    summary_lines.append(f"  Overall Mean IoU     : {overall_iou:.4f}")
    summary_lines.append(f"  Overall Area Error   : {overall_area_err:.2f}%")
    summary_lines.append("=================================================================")
    
    report_text = "\n".join(summary_lines)
    print(report_text)
    
    report_path = os.path.join(eval_dir, "ellipse_fitting_error_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"[+] Saved text report to {report_path}")

if __name__ == "__main__":
    main()
