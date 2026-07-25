# BPA Models & Evaluation Comparative Study
**Toronto Metropolitan University — Major Research Project (MRP)**  
**Task 1: Pattern and Force Mechanism Classification Models**

This document details the theoretical foundations, implementation architectures, training configurations, and comparative results of the two deep learning models evaluated on Task 1.

---

## 1. Theoretical Foundations of Architectures

### Model 1: Multi-Task EfficientNet-B0
EfficientNet scaling introduced by Google in 2019 scales depth (layers), width (channels), and resolution (image size) uniformly using a fixed compound coefficient:
$$\text{Depth (d)} = \alpha^\phi, \quad \text{Width (w)} = \beta^\phi, \quad \text{Resolution (r)} = \gamma^\phi$$
$$\text{subject to } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \quad \text{and} \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$
* **Mobile Inverted Bottleneck (MBConv):** Employs depthwise separable convolutions to drastically reduce parameter calculations, linear bottlenecks, inverted residuals, and squeeze-and-excitation optimization (an attention mechanism that adaptively recalibrates channel responses).
* **Parameters:** ~5.3 Million parameters.

### Model 2: Multi-Task ResNet-50
ResNet (Residual Network) solved the vanishing gradient problem in very deep networks by introducing **residual connections (skip connections)**:
$$H(x) = F(x) + x$$
Where $F(x)$ represents the stacked non-linear layers, and $x$ is the identity mapping bypassed directly to the addition block. This allows gradients to flow directly back through skip connections during backpropagation.
* **Architecture:** Composed of 50 layers using bottleneck residual blocks.
* **Parameters:** ~25.6 Million parameters (5x larger than EfficientNet-B0).

---

## 2. Multi-Task Learning Framework

Our classification engine uses a **Multi-Task Learning (MTL) architecture**. The network splits into two task-specific classifier heads branching from a single shared convolutional backbone:

```
                  Input Image (224x224x3)
                            │
                            ▼
                Shared Convolutional Backbone
             (EfficientNet-B0 or ResNet-50)
                            │
                            ▼
              Shared Spatial Feature Vector (X)
                            ├───► Pattern Head (Linear) ───► Pattern (5 Classes)
                            └───► Mechanism Head (Linear) ──► Force (3 Classes)
```

### Heads and Output Dimensions
* **Pattern Head:** Maps shared spatial features ($1,280$ for EfficientNet, $2,048$ for ResNet) to **5 classes**: *Cough Spatter, Gunshot, Impact Spatter, Passive Drip, and Transfer/Wipe*.
* **Force Mechanism Head:** Maps shared features to **3 classes**: *Passive, Low Velocity, and Medium/High Velocity*.
* **Combined Loss Function:** Optimizes both tasks simultaneously by summing Cross-Entropy losses:
  $$L_{\text{total}} = L_{\text{pattern}} + L_{\text{mechanism}}$$

### Two-Stage Fine-Tuning Schedule
1. **Stage 1 (Feature Extraction Warm-up):**
   - Shared backbone is frozen (`requires_grad = False`).
   - Only the task-specific linear heads are trained.
   - **Optimizer:** Adam ($LR = 0.001$), Duration: 5 Epochs.
2. **Stage 2 (Full Model Fine-Tuning):**
   - The entire network is unfrozen.
   - **Optimizer:** Adam ($LR = 0.0001$) to prevent corrupting pre-trained features.
   - **Duration:** 10 Epochs.

---

## 3. Training & Local Evaluation Instructions

### A. Training on Google Colab T4 GPU
Because deep residual networks are computationally intensive, train them on Colab's free T4 GPU:
1. Open a new notebook on [Google Colab](https://colab.research.google.com) and set runtime: `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ **T4 GPU** $\rightarrow$ Save.
2. Mount Google Drive and install requirements:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !pip install timm albumentations -q
   ```
3. Run the training script:
   - For EfficientNet: `!python3 "/content/drive/MyDrive/Task_1_Classification/Code/training/train_efficientnet.py"`
   - For ResNet-50: `!python3 "/content/drive/MyDrive/Task_1_Classification/Code/training/train_resnet50.py"`

Once training is complete, the weights (`.pth`), epoch history (`.json`), and loss/accuracy plots (`.png`) are saved inside `Task_1_Classification/Models/`.

### B. Local Model Evaluation
To compile thesis metrics on your local Mac:
1. Open Terminal and run the evaluation script:
   ```bash
   cd /Users/shahidabatool/Desktop/MRP
   ./Task_1_Classification/bpa_venv/bin/python Task_1_Classification/Code/evaluation/evaluate_efficientnet.py
   ./Task_1_Classification/bpa_venv/bin/python Task_1_Classification/Code/evaluation/evaluate_resnet50.py
   ```
This generates precision/recall text reports, confusion matrices, and ROC curves under `Task_1_Classification/Evaluation/`.

---

## 4. Comparative Experimental Results

Both models were evaluated on **2,470 completely unseen, preprocessed validation images**:

### A. Overall Performance Summary
| Model Architecture | Pattern Accuracy | Pattern F1-Score (Macro) | Mechanism Accuracy | Mechanism F1-Score (Macro) | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | **97.81%** | **97.78%** | **99.60%** | **99.55%** | **~5.3 Million** |
| **ResNet-50** | 97.04% | 97.00% | 99.15% | 99.04% | ~25.6 Million |

### B. Pattern Classification Class-by-Class Metrics
#### 1. EfficientNet-B0 (Macro F1-Score: 97.78%)
| Pattern Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cough Spatter** | 0.9597 | 1.0000 | **0.9794** | 500 |
| **Gunshot** | 0.9699 | 0.9680 | **0.9690** | 500 |
| **Impact Spatter** | 0.9821 | 0.9319 | **0.9563** | 470 |
| **Passive Drip** | 0.9940 | 0.9960 | **0.9950** | 500 |
| **Transfer/Wipe** | 0.9861 | 0.9920 | **0.9890** | 500 |

#### 2. ResNet-50 (Macro F1-Score: 97.00%)
| Pattern Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cough Spatter** | 0.9311 | 1.0000 | **0.9643** | 500 |
| **Gunshot** | 0.9663 | 0.9760 | **0.9711** | 500 |
| **Impact Spatter** | 0.9641 | 0.9149 | **0.9389** | 470 |
| **Passive Drip** | 0.9959 | 0.9760 | **0.9859** | 500 |
| **Transfer/Wipe** | 0.9980 | 0.9820 | **0.9899** | 500 |

### C. Force Mechanism Classification Metrics
#### 1. EfficientNet-B0 (Macro F1-Score: 99.55%)
| Force Mechanism | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Passive** | 0.9940 | 0.9940 | **0.9940** | 500 |
| **Low Velocity** | 1.0000 | 0.9920 | **0.9960** | 500 |
| **Medium/High Velocity** | 0.9953 | 0.9980 | **0.9966** | 1,470 |

#### 2. ResNet-50 (Macro F1-Score: 99.04%)
| Force Mechanism | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Passive** | 0.9959 | 0.9800 | **0.9879** | 500 |
| **Low Velocity** | 0.9980 | 0.9820 | **0.9899** | 500 |
| **Medium/High Velocity** | 0.9879 | 0.9986 | **0.9932** | 1,470 |

---

## 5. Comparative Analysis & Key Scientific Insights

1. **The Efficiency Paradox:**
   EfficientNet-B0 achieved an accuracy of **97.81% (+0.77% over ResNet-50)** and a mechanism accuracy of **99.60% (+0.45% over ResNet-50)** despite having **5x fewer parameters** (~5.3M vs. ~25.6M). This indicates that simply adding depth or model scale does not yield superior results for forensic droplet geometry tasks.
2. **Impact of Squeeze-and-Excitation (SE) Attention:**
   EfficientNet's edge is driven by its uniform **Compound Scaling** and **SE attention blocks**. The SE blocks dynamically weight feature channels, allowing the backbone to ignore remaining micro-texture variations (e.g., paper cards vs. plastic) and focus specifically on droplet geometry borders (satellite splatters, spines, scallops).
3. **Robustness Against Ambient Noise:**
   Both networks obtained extremely high force mechanism accuracies (above 99.1%), indicating that background-masked feature extraction successfully uncovers the underlying kinetic energy signatures of droplets.
