# MRP Submission Replication & Execution Guide
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Project:** Forensic Bloodstain Pattern Analysis (BPA) & Time Since Deposition (TSD) Estimation

This document provides instructions on how to replicate the environment and execute the core application (the Streamlit dashboard) and model training/evaluation pipelines.

---

## 📂 Codebase Directory Structure

- `dashboard.py`: The main Streamlit web interface integrating the multi-task classifier, physics engine, and explainability overlays.
- `Task_1_Classification/Code/`: Core preprocessing, augmentation, training (`train_model.py`, `train_efficientnet.py`), and evaluation scripts.
- `Task_1_Classification/Models/`: Python-trained neural network parameters (EfficientNet-B0 weights, curves, and history logs).
- `Task_2_TSD/Code/`: Temporal estimation scripts (`bloodnet.py`, `train_tsd_resnet50.py`, etc.).
- `Task_2_TSD/Models/`: Temporal estimation neural network parameters (ResNet-50 CBAM weights, training logs).
- `assets/`: Image resources and icons loaded dynamically by the Streamlit application.
- `Task_1_Classification/MD/` and `Task_2_TSD/MD/`: Technical guides, preprocessing parameters, training commands, and comparative experimental results.
- `requirements.txt`: Python package dependency specification file.

---

## ⚡ Installation & Execution

### 1. Environment Setup
To ensure reproducibility, install Python 3.10+ and set up a new virtual environment:

```bash
# Create a virtual environment
python -m venv mrp_env

# Activate the virtual environment
# On macOS/Linux:
source mrp_env/bin/activate
# On Windows:
mrp_env\Scripts\activate
```

### 2. Install Package Dependencies
Install all required libraries specified in the unified list:

```bash
pip install -r requirements.txt
```

### 3. Launch the Forensic BPA & TSD Dashboard
Run the dashboard server to launch the web-based visual analytics interface:

```bash
streamlit run dashboard.py
```
This will open the dashboard in your default browser at `http://localhost:8501`.

---

## ⚠️ Important Dataset Submission Notice

Due to the size of the datasets (over 20 GB of raw and processed high-resolution bloodstain scan images), they have been excluded from the direct portal upload package. 

### Data Directories Description:
- `Raw_dataset/`: Original raw image folders.
- `Task_1_Classification/Data/`: Preprocessed, masked, and augmented splits.
- `Task_1_Classification/Text/`: Sourced subfolders from Mendeley, Zenodo, and Kaggle.
- `Task_2_TSD/Text/BloodNet_50k_Images/`: Sourced rabbit bloodstain datasets.

### 🔗 Complete Project Drive Archive:
To access the full workspace—including all raw/augmented datasets, intermediate processing checkpoints, and pre-trained `.pth` weights—please download them directly from the shared Drive folder:
👉 **[Access the Full Project Archive on Google Drive](https://drive.google.com/drive/folders/1Qg8CjBzJ2wPJIfGgGEfSezEaNndCxuGy?usp=sharing)**

To reproduce training, data preprocessing, or local inference, simply copy/download the folders from the Drive archive and merge them into the respective locations in your local cloned directory.
