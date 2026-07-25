# Task 1 Classification: Development History & Optimization Journal

This document tracks the incremental design decisions, challenges, and optimization phases followed during the development of the **Task 1 Bloodstain Pattern Classification** models.

---

## 📍 Phase 1: The Initial Baseline Run (5-Class Setup)
* **Objective:** Train standard multi-task backbones (EfficientNet-B0 & ResNet-50) on 5 bloodstain pattern classes: Gunshot, Impact Spatter, Passive Drip, Transfer/Wipe, and Cough Spatter.
* **Findings:**
  * **Overall Accuracy:** 86.89% (EfficientNet-B0) / 88.20% (ResNet-50).
  * **The Passive Drip Bottleneck:** Both models performed poorly on **Passive Drip** (Recall of only **13.33%** for EfficientNet and **20.00%** for ResNet-50). Most passive drips were misclassified as **Transfer/Wipe** smears.
  * **Subject-Level Data Leakage:** A leakage audit showed that all **Cough Spatter** frames originated from a single video source (`15Aa2`), making a leak-free train/test partition impossible for that class.

---

## 📍 Phase 2: Leakage Mitigation & First Class-Weighting Attempt
* **Objective:** Address data leakage constraints and class-imbalance recall issues.
* **Key Steps & Decisions:**
  1. **Excluding Cough Spatter:** Completely removed Cough Spatter from the active training and testing sets, transitioning to a clean **4-class setup** (Gunshot, Impact, Passive Drip, Transfer/Wipe) with 100% independent parent splits.
  2. **First Loss Weighting Attempt (Ratio-Based):** To push the network to prioritize the minority classes (Passive Drip and Transfer/Wipe), we implemented a validation-frequency inverse ratio class weight in the Cross-Entropy loss.
* **Unexpected Consequences:**
  * The ratio weighting created an extreme **24.5x difference** in gradient penalties.
  * This warped the models' decision boundaries. Because of the massive penalty for misclassifying a minority class droplet, the models began aggressively over-predicting **Impact Spatter** for standard gunshot droplets. 
  * Gunshot/Impact precision collapsed (Impact Spatter precision fell to **39.94%**), rendering the models forensically unusable.

---

## 📍 Phase 3: Dataset Migration & Loss Optimization
* **Objective:** Solve the root cause of Passive Drip visual overfitting and correct the decision boundary warping.
* **Key Steps & Decisions:**
  1. **Source of the Passive Drip Issue:** Identified that the original dataset only had **3 unique video recordings** of Passive Drips. When backgrounds were masked, the model visually overfit to the exact handle contours of the tools used in those videos.
  2. **Borealis Dataset Integration:** Moved the 346 old passive images to `Passive_Drip_outdated` and integrated **400 new scanned drip patterns** from the public Borealis Data repository (doi:10.5683/SP3/YSTDGI) spanning drop heights of 25cm to 100cm.
  3. **Preprocessing Skin Mask Fix:** Resolved a bug in `preprocess_passive.py` where a video-based YCrCb skin mask (used to subtract human hands) was subtracting red blood spots from the scanned paper background. Disabling this mask enabled **100% successful cleaning** of the 400 new scans.
  4. **Unbiased Splits & Warping Augmentations:** Created a 320 train / 80 test split at the source-scan level. Augmented the training split to 1,000 images using Albumentations shape-warping (`ElasticTransform`, `GridDistortion`, and `CoarseDropout`) to prevent contour shape memorization.
  5. **Smooth Loss Weighting:** Replaced the extreme 24x weights with a smooth **square-root inverse-frequency loss weighting**, capping the penalty ratio to 3.46x.

---

## 📍 Phase 4: Final Evaluation Results (Success Metrics)
* **Objective:** Verify and compile the results of the updated models.
* **Key Outcomes:**
  * **Overall Test Accuracy (EfficientNet-B0):** Skyrocketed to **92.95%** on the raw test set and **95.83%** on the balanced holdout subset.
  * **Passive Drip Performance:** **Recall reached 100.00%** (F1-score: **98.77%**), proving that increasing source diversity and shape warping successfully resolved the overfitting.
  * **Transfer/Wipe Performance:** Achieved a near-perfect F1-score of **99.16%**.
  * **Forensic Admissibility Documentation:** Compiled all dynamic comparison tables and side-by-side confusion matrix/ROC-AUC plots into the final styled Microsoft Word document: `Task_1_Evaluation_Report.docx`.
