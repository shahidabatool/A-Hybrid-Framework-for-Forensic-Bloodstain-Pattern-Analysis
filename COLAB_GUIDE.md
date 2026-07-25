# Google Colab Master Notebook Execution Guide
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Notebook:** `BPA_Master_Colab_Notebook.ipynb`

This document provides complete instructions on how to set up, configure, and execute the master Google Colab notebook (`BPA_Master_Colab_Notebook.ipynb`) for both model training and pre-trained inference.

---

## 🚀 Why Use Google Colab?
Deep learning architectures (specifically ResNet-50, ConvNeXt-Tiny, and multi-task networks) require heavy tensor computations. Google Colab provides a free **T4 GPU acceleration framework**, which speeds up model training and evaluation by over **20x** compared to standard local CPU execution.

---

## 📂 Google Drive Prerequisites
Before opening the notebook, you must package and upload your raw datasets to your personal Google Drive so the Colab virtual machine (VM) can mount and extract them.

1. **Compress your folders** into two separate zip archives:
   - **`Task_1_Classification_Colab.zip`**: Contains the Task 1 `Code/`, `Data/`, and `Models/` folders.
   - **`Task_2_TSD_Colab.zip`**: Contains the Task 2 `Code/`, `Models/`, and `Text/BloodNet_50k_Images/` folders.
2. **Upload both zip files** directly to the root of your Google Drive (`MyDrive/`):
   - Path on Drive: `/MyDrive/Task_1_Classification_Colab.zip`
   - Path on Drive: `/MyDrive/Task_2_TSD_Colab.zip`

---

## ⚙️ Step 1: Open Notebook & Set T4 GPU Runtime
1. Upload [BPA_Master_Colab_Notebook.ipynb](file:///Users/shahidabatool/Desktop/MRP/BPA_Master_Colab_Notebook.ipynb) to your Google Drive.
2. Double-click it to open it in **Google Colab**.
3. Set the hardware accelerator:
   - Go to **Runtime** in the top menu $\rightarrow$ **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** $\rightarrow$ Click **Save**.

---

## 🔍 Step 2: Notebook Phases Breakdown

The notebook is divided into four chronological execution phases. Depending on your goal, you can run the notebook in two distinct modes: **Rapid Inference Mode** or **Scratch Training Mode**.

### 1. Environment Setup (Cells 1–3)
- **Mount Google Drive**: Run the mounting cell. This will prompt you to authorize Google Colab to read files from your Drive account.
- **GPU Check**: Verifies that Colab has allocated a T4 graphics card.
- **Install Requirements**: Installs critical libraries (`timm`, `albumentations`, `scikit-learn`, `python-docx`) onto the local VM.

---

### 2. Option A: Pre-Trained Model Inference Mode (Cells 4–6)
*Use this option if you want to run quick predictions on custom images using the saved model checkpoints in your repository without training from scratch.*

- **Path Setup**: Mounts Drive and changes directory (`os.chdir`) to your active repository root folder on Drive.
- **Import Architectures**: Dynamically appends model paths to python syspath and imports `BPAMultiTaskEfficientNet` and `bloodnet50`.
- **Load Weights**: Loads pre-trained weights from `Task_1_Classification/Models/` and `Task_2_TSD/Models/`.
- **Run Unified Inference**: Evaluates any custom bloodstain image path, printing out a consolidated forensic report containing:
  - Predicted Pattern Class (Task 1)
  - Predicted Force Mechanism (Task 1)
  - Predicted Drying Age / TSD (Task 2)

---

### 3. Option B: Scratch Training & Evaluation Mode (Cells 7–13)
*Use this option if you want to re-train the models from scratch on the full training sets.*

- **Local Extraction (Crucial Step)**: Extracts your Drive zip files directly to the local VM disk `/content/Task_1` and `/content/Task_2`. 
  - *Forensic Tip:* Training on Colab's local VM disk is drastically faster than training directly from Google Drive, as it avoids slow network read limits.
- **Run Leakage Audit**: Runs `audit_data_leakage.py` to verify that there are no overlapping parent-video sources between your training and validation splits.
- **Task 1 Classifier Training**: Trains EfficientNet-B0, ResNet-50, and ConvNeXt-Tiny classifiers using the smooth loss weighting script.
- **Task 1 Model Evaluation**: Runs inference on the test splits, generating local confusion matrices and ROC curves.
- **Task 2 TSD Training**: Fine-tunes the ResNet-CBAM temporal models and trains the comparison EfficientNet backbones.
- **Task 2 Evaluation**: Evaluates TSD age models on raw test splits ($21,984$ images) and balanced splits ($300$ images).
- **Compile & Sync Reports**: Runs Python-Word docx report compilers and copies all newly trained checkpoints (`.pth`) and figures from `/content/` back to your persistent Google Drive models folder.

---

### 4. Phase 4: Local Dashboard Execution (Cell 14)
- Provides standard command-line reminders to copy the final weights back to your local machine and start the Streamlit interactive dashboard locally:
  ```bash
  streamlit run dashboard.py
  ```
