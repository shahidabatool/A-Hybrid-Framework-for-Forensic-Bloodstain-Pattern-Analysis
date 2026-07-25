# BPA Data Sourcing & Preprocessing Documentation
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Task 1: Pattern and Force Mechanism Classification Data Engine**

This document provides a comprehensive description of the dataset sources, frame extraction parameters, background removal color-space thresholds, and forensic-safe data augmentation strategies implemented in the preprocessing pipeline.

---

## 1. Multi-Source Data Collection Strategy
To construct a scientifically rigorous dataset, we consolidated high-resolution, peer-reviewed experimental data from five primary repositories:

| Dataset Name / Repository | Sub-folder Name | Primary References & Citations | Sourced Pattern Class | Sourced Force Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **BPA Scans for Teaching & Research** | `1_BPA_Scans_Teaching_Kaggle` | Liu et al. (2020), Attinger et al. (2018/2019) | **Gunshot** | Medium/High Velocity |
| **Bloodstain Pattern Analysis (BPA) Dataset** | `Raw_dataset/Passive_drip_extracted` | Borealis Data, doi:10.5683/SP3/YSTDGI | **Passive Drip** (New replacement dataset) | Passive / Gravitational |
| **Blood Stain Pattern Video Dataset (BSPDS)** | `4_Blood_Stain_Pattern_Videos_Mendeley` | Mendeley Data (BSPDS) | **Cough Spatter**, **Transfer/Wipe** | Cough: Med/High Velocity<br>Transfer: Low Velocity |
| **High-Resolution Impact Spatter Scans** | `5_High_Res_Impact_Spatter_Zenodo` | Zenodo Record (10909428) | **Impact Spatter** | Medium/High Velocity |
| **Supplementary HemoSpat Materials** | `SupplementaryMaterials` | HemoSpat Reference Materials | **Impact Spatter** | Medium/High Velocity |

### Sourced Dataset Characteristics
1. **Kaggle scans (Liu & Attinger):** Massive, ultra-high-resolution raw scans (~982 MB, 16 scans) of physical gunshot backspatters.
2. **Borealis Passive Drips:** 400 unique high-resolution scans of passive drops falling on paper target surfaces from four heights (25 cm, 50 cm, 75 cm, 100 cm).
3. **Mendeley Videos (BSPDS):** Comprises high-speed `.avi` videos showing physical experiments of coughed blood and surfaces being wiped.
4. **Zenodo Scans:** Highly detailed `.tiff` and `.png` scans (~725 MB, 19 scans) capturing spatter droplet geometries on paper cards.
5. **HemoSpat Materials:** Macro close-up images (~50 MB, 279 images) of impact droplets used in area-of-origin calibration.

---

## 2. Passive Drip Dataset Replacement Details
During our model development, an audit revealed that using frames extracted from Mendeley videos for the **Passive Drip** class caused the model to overfit. The Mendeley dataset only contained 3 unique source videos for passive drips, yielding very low visual diversity. Consequently, the model struggled to generalize and misclassified unseen Passive Drips as *Transfer/Wipe* smears.

To resolve this bottleneck, we replaced the video frames with a static scanned dataset:
* **Source:** Borealis Data Repository ([Borealis Link](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/YSTDGI))
* **DOI:** `doi:10.5683/SP3/YSTDGI`
* **Visual Diversity Increase:** Expanded passive drip representation by **over 100x** through 400 unique scanned drip patterns.
* **Extraction Details:**
  - 100 scans from 25 cm drop height (`paper25cm`)
  - 100 scans from 50 cm drop height (`paper50cm`)
  - 100 scans from 75 cm drop height (`paper75cm`)
  - 100 scans from 100 cm drop height (`paper100cm`)
  - **Saved location:** `Raw_dataset/Passive_drip_extracted/`
  - **Processed Organized location:** `Task_1_Classification/Data/Organized/Passive_Drip/`

---

## 3. Preprocessing & Extraction Pipeline

The raw datasets were processed inside the `Task_1_Classification/Code/preprocessing/` folder using `organize_and_extract.py` and `validate_cleaning.py`.

```
Raw Sourced Datasets & Videos
    ├── organize_and_extract.py ──> Frame Extraction & Static Mapping
    └── Preprocessing Pipeline
         ├── Dual-Masking (HSV Red Isolation + YCrCb Skin Tone Subtraction)
         ├── Contour Area Filtering (> 50px) to remove sensor noise
         ├── ROI Bounding Box Selection with 30px Padding
         ├── Pure White Background Replacements (RGB = 255, 255, 255)
         └── Forensic-Safe Augmentations (Albumentations, capped rotations, flips DISABLED)
```

### Video Frame Extraction (`organize_and_extract.py`)
Dynamic videos are extracted using an OpenCV-based frame sampler:
* **Sampling Rate:** The script dynamically calculates the sampling rate based on the video length to extract exactly $150$ non-contiguous, high-fidelity frames. This avoids temporal redundancy (identical consecutive frames).
- **Forensic Mapping:** Sourced videos are parsed by filename prefixes:
  - Prefix `15` (e.g., `Coughed blood.avi`) $\rightarrow$ **Cough_Spatter**
  - Prefix `10` (e.g., `Wiped with paper towel.avi`) $\rightarrow$ **Transfer_Wipe**

### Dual-Masking Background & Skin Subtraction
To eliminate shortcut learning (learning cardboard texture, tile backgrounds, or hands and tool outlines rather than droplet morphology), we developed a dual-masking OpenCV engine:

1. **Blood Color Isolation (HSV Space):**
   Blood displays specific hue-saturation-value profiles. We applied a double-range red threshold mask:
   $$\text{Mask}_1: H \in [0, 15], S \in [40, 255], V \in [20, 255]$$
   $$\text{Mask}_2: H \in [160, 180], S \in [40, 255], V \in [20, 255]$$
   $$\text{Blood Mask} = \text{Mask}_1 \cup \text{Mask}_2$$

2. **Skin Color Subtraction (YCrCb Space):**
   To filter out researchers' hands or bloodied fingers appearing in dynamic videos, we convert the color space to YCrCb:
   $$\text{Skin Mask}: Y \in [0, 255], Cr \in [133, 173], Cb \in [77, 127]$$
   $$\text{Final Blood Mask} = \text{Blood Mask} \cap \neg(\text{Skin Mask})$$

3. **ROI Bounding & Masking:**
   - Contours are extracted from the `Final Mask`. Any contour $\le 50$ pixels is discarded as sensor noise.
   - Bounding boxes are computed around the remaining bloodstains. A **30-pixel padding** is added to prevent clipping fine drop edges (satellite spatters/spines).
   - All non-blood pixels are replaced with a solid **pure white background ($RGB = 255, 255, 255$)**.

---

## 4. Mathematical Preprocessing Thresholds

Below are the exact values locked into the preprocessing script (`validate_cleaning.py`):

| Parameter | Bound / Value | Justification / Description |
| :--- | :--- | :--- |
| **HSV Lower Red 1** | `[0, 40, 20]` | Captures bright-to-medium red bloodstains in HSV color space. |
| **HSV Upper Red 1** | `[15, 255, 255]` | Upper bound for primary red hues. |
| **HSV Lower Red 2** | `[160, 40, 20]` | Captures dark red / dried bloodstains near the upper HSV boundary. |
| **HSV Upper Red 2** | `[180, 255, 255]` | Upper bound for secondary red hues. |
| **YCrCb Lower Skin** | `[0, 133, 77]` | Defines the lower bound of human skin color in YCrCb space. |
| **YCrCb Upper Skin** | `[255, 173, 127]` | Defines the upper bound of skin color to subtract fingers/hands. |
| **Contour Size Filter** | `> 50 pixels` | Discards camera sensor noise, compression artifacts, and dust specks. |
| **Crop Padding** | `30 pixels` | Preserves micro-splatters and spines at droplet edges. |
| **Variance Threshold** | `> 10` | Used during cropping to discard blank background tiles. |

---

## 5. Forensic-Safe Data Augmentation Strategy

Original datasets in forensic science are heavily unbalanced. An audit of our preprocessed dataset showed a severe class imbalance (Transfer: 508 images, Passive: 477 images, Impact: 298 images, Cough: 150 images, Gunshot: 147 images). 

### Why Standard Augmentation Fails in Forensics
Standard vision pipelines frequently use **horizontal and vertical flips**. In bloodstain analysis, this is highly dangerous:
- **The Physics of Travel:** The direction of travel of a blood drop (indicated by its tail, spines, and satellite spatters) is crucial for calculating the area of origin.
- **Flips Destroy Evidence:** Mirroring an image horizontally reverses the implied vector of travel. If a model trains on flipped spatters, it learns incorrect geometric-angle mappings, completely invalidating the physics-based calculations.

### Custom Augmentation Pipeline Settings
Using the **Albumentations** library, we implemented a custom pipeline that preserves physical constraints while introducing necessary visual variance:

| Parameter / Transform | Settings | Forensic Justification |
| :--- | :--- | :--- |
| **Crop Size** | `1024 x 1024` pixels | Retains fine details of satellite spatters and spines. |
| **Max Dim Resize** | `2048` pixels | Maximum dimension to resize smaller images before cropping. |
| **Rotation Limits** | `[-15°, +15°]` | Re-simulates minor target plane tilting or camera placement variations while avoiding geometry reversal. |
| **Scale Limit** | `[-10%, +10%]` | Simulates minor distance-to-target variations. |
| **Shift Limit** | `[-5%, +5%]` | Simulates minor shifts in framing / centering. |
| **Brightness & Contrast** | Limit `±20%` (p=0.5) | Simulates real-world crime scene lighting variations (shadows, flashlights). |
| **Gaussian Noise** | Variance limit `[10, 50]` (p=0.3) | Simulates camera sensor noise, especially in low-light forensic captures. |
| **Perspective Distortion** | Scale `[0.02, 0.05]` (p=0.3) | Simulates minor perspective distortions from off-angle shooting. |
| **Horizontal/Vertical Flips** | **DISABLED** | **CRITICAL**: Directionality, drip tail angles, and wipe vectors dictate the physical meaning of bloodstains. Flips are deactivated to prevent geometric inversion. |
| **Test Set Augmentations** | **DISABLED** | The holdout test set remains raw and unaugmented to ensure realistic model evaluation. |

---

## 6. Dataset Quality Assurance (QA) Audit

Our data engine completed successfully. The final training and testing splits contain perfectly balanced classes:

### Training Set (5,000 Total Images)
* **Cough_Spatter:** 1,000 images ($20.00\%$) — *Balanced*
* **Gunshot:** 1,000 images ($20.00\%$) — *Balanced*
* **Impact_Spatter:** 1,000 images ($20.00\%$) — *Balanced*
* **Passive_Drip:** 1,000 images ($20.00\%$) — *Balanced*
* **Transfer_Wipe:** 1,000 images ($20.00\%$) — *Balanced*

### Testing Set (2,500 Total Images)
* **Cough_Spatter:** 500 images ($20.00\%$) — *Balanced*
* **Gunshot:** 500 images ($20.00\%$) — *Balanced*
* **Impact_Spatter:** 500 images ($20.00\%$) — *Balanced*
* **Passive_Drip:** 500 images ($20.00\%$) — *Balanced*
* **Transfer_Wipe:** 500 images ($20.00\%$) — *Balanced*
