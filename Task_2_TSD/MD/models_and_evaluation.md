# BPA Task 2 (TSD) Models & Evaluation Comparative Study
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Task 2: Time Since Deposition (TSD) Estimation Models**

This document details the architectures, training routines, local execution steps, and comparative experimental results of the four deep learning models evaluated on the Task 2 TSD dataset.

---

## 1. Description of Model Architectures

Four architectures were implemented and benchmarked against the temporal classification task:

1. **Custom BloodNet Baseline:** A lightweight, custom CNN architecture (~2.1M parameters) specifically designed for temporal bloodstain classification, consisting of stacked convolution layers, max pooling, and fully connected linear layers.
2. **Fine-Tuned ResNet-50:** A deep 50-layer residual convolutional neural network (~25.6M parameters) using bottleneck skip connections, starting from ImageNet weights and fully unfrozen for end-to-end temporal gradient learning.
3. **EfficientNet-B0:** A compound-scaled network (~5.3M parameters) leveraging depthwise separable MBConv blocks and channel-wise squeeze-and-excitation attention.
4. **ConvNeXt-Tiny:** A modernized all-convolutional network (~28.6M parameters) mimicking Vision Transformer designs (grouped convolutions, inverted bottlenecks, large $7\times 7$ kernels).

---

## 2. Training and Evaluation Routines

### Training Setup (Google Colab T4 GPU)
Due to the large dataset size (50,000 images), the models were trained using transfer learning on Google Colab:
1. Load dataset splits and metadata.
2. Freeze backbone layers, train task classifier heads with Adam optimizer ($LR=0.001$, Epochs: 5).
3. Unfreeze all parameters and run end-to-end training at a low learning rate ($LR=0.0001$, Epochs: 10).
4. Save model weights (`.pth`) and history stats inside `Task_2_TSD/Models/`.

### Local Evaluation Command
To evaluate the trained weights on your local Mac:
```bash
cd /Users/shahidabatool/Desktop/MRP
./Task_1_Classification/bpa_venv/bin/python Task_2_TSD/Code/evaluation/evaluate_tsd.py
```
This runs inference across all models, generating precision/recall reports, ROC curves, and confusion matrix heatmaps under `Task_2_TSD/Evaluation/`.

---

## 3. Comparative Experimental Results

All four models were evaluated on the raw test split ($21,984$ images) and a balanced holdout split ($300$ images):

### A. Overall Performance Summary
| Model Architecture | Raw Test Accuracy | Raw F1-Score (Macro) | Balanced Test Accuracy | Balanced F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **BloodNet Baseline** | 45.26% | 21.60% | 34.00% | 19.82% |
| **EfficientNet-B0** | 91.61% | 89.97% | 91.00% | 90.87% |
| **ConvNeXt-Tiny** | 92.41% | 91.34% | 91.67% | 91.54% |
| **Fine-Tuned ResNet-50** | **97.46%** | **97.35%** | **98.00%** | **98.01%** |

### B. Class-by-Class Comparative Performance (Raw Test Split)
The detailed precision, recall, and F1 metrics on the $21,984$ raw test images highlight the performance of each backbone:

#### 1. Fine-Tuned ResNet-50 (Accuracy: 97.46%)
* *Best Performing Model*
* **Fresh:** Precision: **99.90%** | Recall: **95.89%** | F1: **97.85%**
* **Intermediate:** Precision: **95.21%** | Recall: **96.66%** | F1: **95.93%**
* **Aged:** Precision: **97.75%** | Recall: **98.80%** | F1: **98.27%**

#### 2. ConvNeXt-Tiny (Accuracy: 92.41%)
* **Fresh:** Precision: **99.65%** | Recall: **81.06%** | F1: **89.40%**
* **Intermediate:** Precision: **85.83%** | Recall: **90.24%** | F1: **87.98%**
* **Aged:** Precision: **93.81%** | Recall: **99.68%** | F1: **96.66%**

#### 3. EfficientNet-B0 (Accuracy: 91.61%)
* **Fresh:** Precision: **99.97%** | Recall: **74.25%** | F1: **85.21%**
* **Intermediate:** Precision: **82.07%** | Recall: **93.44%** | F1: **87.39%**
* **Aged:** Precision: **95.34%** | Recall: **99.37%** | F1: **97.31%**

#### 4. Custom BloodNet Baseline (Accuracy: 45.26%)
* *Worst Performing Model*
* **Fresh:** Precision: **0.00%** | Recall: **0.00%** | F1: **0.00%**
* **Intermediate:** Precision: **02.85%** | Recall: **00.48%** | F1: **00.83%**
* **Aged:** Precision: **47.54%** | Recall: **97.82%** | F1: **63.99%**

### C. Fine-Tuned ResNet-50 Balanced Holdout Performance (Accuracy: 98.00%)
On the balanced split (100 images per class), ResNet-50 achieved near-perfect scores:
* **Fresh:** Precision: **100.00%** | Recall: **96.00%** | F1: **97.96%**
* **Intermediate:** Precision: **95.19%** | Recall: **99.00%** | F1: **97.06%**
* **Aged:** Precision: **99.00%** | Recall: **99.00%** | F1: **99.00%**

---

## 4. Comparative Analysis & Key Scientific Insights

1. **Baseline Failure Due to Overfitting:**
   The custom BloodNet baseline suffered a massive failure, classifying nearly all stains as **Aged** (yielding $0\%$ recall on Fresh and $<1\%$ recall on Intermediate). This was caused by the severe class imbalance in the training data (Aged samples represent nearly $50\%$ of the dataset), which the simple custom CNN layout failed to adapt to, defaulting to the majority class guess.
2. **Superiority of Deep Transfer Learning:**
   Fine-tuned ResNet-50 successfully resolved the class imbalance bottleneck, achieving **97.46% raw accuracy** and **98.00% balanced accuracy**. The pre-trained ImageNet filters provided a rich baseline for features like texture, edge details, and fine color changes. Fully unfreezing the network allowed these filters to fine-tune to the subtle shifts of hemoglobin oxidation (red to brown to grey), enabling high accuracy.
3. **ResNet-50 vs. ConvNeXt and EfficientNet:**
   ResNet-50 outperformed the modern ConvNeXt-Tiny by **+5.05%** and EfficientNet-B0 by **+5.85%** on raw test accuracy. ResNet's classical residual blocks and skip connections preserve local pixel color representations better than the heavily grouped convolutions and attention layers of ConvNeXt and EfficientNet, which are more suited to extracting macro objects rather than micro color-space oxidation states.
