# Google Colab Master Notebook Execution Guide
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Notebook:** `BPA_Master_Colab_Notebook.ipynb`

This document provides complete instructions on how to set up, configure, and execute the master Google Colab notebook (`BPA_Master_Colab_Notebook.ipynb`) for both model training and pre-trained inference.

---

## 🚀 Why Use Google Colab?
Deep learning architectures (specifically ResNet-50, ConvNeXt-Tiny, and multi-task networks) require heavy tensor computations. Google Colab provides a free **T4 GPU acceleration framework**, which speeds up model training and evaluation by over **20x** compared to standard local CPU execution.

---

## 📂 Phase 1: Opening the Notebook & Setting T4 GPU Runtime
1. Open Google Drive, create a new folder, and upload [BPA_Master_Colab_Notebook.ipynb](file:///Users/shahidabatool/Desktop/MRP/BPA_Master_Colab_Notebook.ipynb).
2. Right-click the notebook file and select **Open with** $\rightarrow$ **Google Colaboratory**.
3. Set the hardware accelerator:
   - Go to **Runtime** in the top menu $\rightarrow$ **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** $\rightarrow$ Click **Save**.
4. Run the first two cells under **Phase 1** to verify GPU allocation and install required dependencies (`timm`, `albumentations`, `python-docx`, etc.).

---

## 🎯 Phase 2: Navigation Options (Supports Anyone)
The notebook's directory paths are **fully dynamic** (resolving paths relative to Python's current working directory using `os.getcwd()`). This ensures the code works for anyone, regardless of whether they clone the repo locally or mount their personal Google Drive. 

In **Step A.1**, you can choose between two methods to set your working directory:

### Method 1: Clone Directly to Colab VM (Recommended for reviewers & external users)
This method clones the code repository directly from GitHub into Colab's high-speed local storage:
```python
!git clone https://github.com/shahidabatool/A-Hybrid-Framework-for-Forensic-Bloodstain-Pattern-Analysis.git
%cd A-Hybrid-Framework-for-Forensic-Bloodstain-Pattern-Analysis
```

### Method 2: Mount Personal Google Drive (Recommended for persistent development & review)
If you want to run the code using the files on your Google Drive:
1. Open the [Shared Project Archive on Google Drive](https://drive.google.com/drive/folders/1Qg8CjBzJ2wPJIfGgGEfSezEaNndCxuGy?usp=sharing).
2. Click the dropdown next to the `MRP` folder name and select **Add shortcut to Drive** $\rightarrow$ select **My Drive**.
3. Run the mount cell in Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
# Navigate directly to the shortcut folder:
%cd /content/drive/MyDrive/MRP
```

---

## 🔍 Step 3: Execution Modes

Depending on your goal, you can run the rest of the notebook in one of two modes:

### Option A: Rapid Inference Mode (Using Saved Checkpoints)
*Use this option if you want to immediately classify and estimate the age of bloodstains using the pre-trained weights included in this repository without training from scratch.*

1. Run **Step A.2** to import architectures. The notebook will automatically resolve the syspath relative to your chosen directory in Step A.1:
   - Loads Task 1 EfficientNet-B0 weights from `Task_1_Classification/Models/model1_efficientnet_b0.pth`.
   - Loads Task 2 ResNet-50 CBAM weights from `Task_2_TSD/Models/best_tsd_model_resnet50.pth`.
2. Run **Step A.3** and input your custom image path into `predict_bloodstain(image_path)` to get a unified forensic analysis report.

---

### Option B: Scratch Training Mode (Using Drive Datasets)
*Use this option if you want to re-train the models from scratch on the full training sets.*

1. **Upload Datasets to Google Drive**: 
   Compress your local data directories into zip files and upload them to the root of your Google Drive:
   - `Task_1_Classification_Colab.zip` (containing Task 1 `Code/`, `Data/`, and `Models/`).
   - `Task_2_TSD_Colab.zip` (containing Task 2 `Code/`, `Models/`, and `Text/BloodNet_50k_Images/`).
2. **Local Extraction (Step B.1)**: Run the cell to extract these zips from Drive directly onto Colab's fast local VM disk `/content/Task_1` and `/content/Task_2` (to prevent Drive I/O bottlenecks).
3. **Execute Pipeline**:
   - Run the data-leakage checks.
   - Run Task 1 model training (EfficientNet, ResNet, ConvNeXt) and evaluations.
   - Run Task 2 TSD model training and evaluations.
   - Sync final weights (`.pth`) and Word reports back to your persistent Google Drive.
