# BPA Task 2 (TSD) Data Sourcing & Preprocessing
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Task 2: Time Since Deposition (TSD) Estimation Data Engine**

This document details the sourcing, chemical-physical justification, temporal class mapping, and image preprocessing pipeline implemented for estimating bloodstain age.

---

## 1. Data Sourcing & Forensic Justification

### Sourced Benchmark Dataset
Our time-since-deposition training is anchored by a massive peer-reviewed benchmark dataset:
* **Dataset Name:** BloodNet-Benchmark (hosted on Figshare)
* **Scale:** Approximately **50,000 time-lapsed photographs** of individual static rabbit blood drops.
* **Metadata:** Accompanied by a master metadata manifest `bloodstain_information.csv` containing ground-truth deposition timestamps and experimental conditions (temperature, humidity).
* **Workspace Location:** `Task_2_TSD/Text/BloodNet_50k_Images/` (with train, validation, and test splits).

### Forensic & Chemical Justification
When blood is deposited outside the body, it undergoes a complex, time-dependent aging process driven by the oxidation of hemoglobin:
1. **Fresh Blood:** Contains primarily **oxyhemoglobin ($HbO_2$)**, giving it a bright red color.
2. **Intermediate States:** Oxygen dissociation and oxidation convert oxyhemoglobin into **methemoglobin ($met-Hb$)**, causing the stain to transition from bright red to dark brown.
3. **Aged States:** Over weeks, methemoglobin further degrades into **hemichrome ($HC$)**, rendering the stain a dull brown or greyish-brown color.

By capturing these distinct spectral and color-space shifts over time, convolutional neural networks can learn to non-destructively map the visual profile of a bloodstain directly to its physical age (Time Since Deposition).

---

## 2. Temporal Class Mapping Strategy

The raw Figshare dataset contains bloodstains photographed across a continuous timeframe up to 28 days. To align with practical forensic scenarios (e.g., classifying whether a stain was deposited *within the last 24 hours*, *within the first two weeks*, or is *several weeks old*), we mapped the **5 time-interval subfolders** into **3 key forensic intervals**:

| Raw Subfolder Splits | Time Since Deposition | Forensic Class Label | Target Integer | Justification |
| :--- | :--- | :--- | :---: | :--- |
| `1d_train`, `1d_val`, `1d_test` | **1 Day** (24 Hours) | **Fresh** | `0` | Represents the immediate window critical for establishing recent alibis. |
| `7d_train` / `val` / `test`<br>`14d_train` / `val` / `test` | **7 to 14 Days** (1–2 Weeks) | **Intermediate** | `1` | Captures the transition period of hemoglobin oxidation into methemoglobin. |
| `21d_train` / `val` / `test`<br>`28d_train` / `val` / `test` | **21 to 28 Days** (3–4 Weeks) | **Aged** | `2` | Represents fully oxidized hemichrome states. |

---

## 3. Image Preprocessing & Split Statistics

### Preprocessing and Normalization Pipeline
To prepare the high-resolution crop photos of the blood drops for deep learning backbones, we implement a PyTorch-based transform pipeline inside `Task_2_TSD/Code/models/TSDDataset`:

1. **Spatial Resizing:** All crop images are resized to a uniform dimension of **$128 \times 128$ pixels** ($128 \times 128 \times 3$) to maintain spatial detail while optimizing computational throughput.
2. **Tensor Conversion:** Images are normalized from $[0, 255]$ pixel values to floating-point tensors in the range $[0.0, 1.0]$.
3. **Standard Normalization:** Standard ImageNet mean and standard deviation normalization is applied to align features with pre-trained transfer-learning backbones:
   $$\mu_{\text{RGB}} = [0.485, 0.456, 0.406], \quad \sigma_{\text{RGB}} = [0.229, 0.224, 0.225]$$

### Dataset Split Distributions
The split structure parsed during training is as follows:

* **Raw Test Split Support (21,984 Total Images):**
  - **Fresh (1d):** 5,227 images ($23.78\%$)
  - **Intermediate (7d/14d):** 6,618 images ($30.10\%$)
  - **Aged (21d/28d):** 10,139 images ($46.12\%$)
  - *Note:* The raw test split reflects a significant class imbalance skewed towards aged stains, which is typical of longitudinal time-series collection.

* **Balanced Validation Split Support (300 Total Images):**
  - **Fresh:** 100 images ($33.33\%$)
  - **Intermediate:** 100 images ($33.33\%$)
  - **Aged:** 100 images ($33.33\%$)
  - *Note:* A perfectly balanced validation holdout of 300 images was constructed to compute metrics unaffected by class overrepresentation, providing an unbiased assessment of classification accuracy.
