import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Render plots headlessly (CLI & Colab-safe)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Environment Auto-Detection ---
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab environment.")
except ImportError:
    IN_COLAB = False
    print("Running in Local environment.")

if IN_COLAB:
    # NOTE: Do NOT call drive.mount() here — this script is run as a subprocess
    # (!python ...) from a Colab notebook cell, which means it runs outside the
    # IPython kernel and drive.mount() will crash with AttributeError.
    # The notebook cell mounts Drive BEFORE calling this script.
    
    # Try a list of potential paths where the 'Task_1_Classification' folder might reside
    possible_dirs = [
        "/content/drive/MyDrive/Task_1_Classification",
        "/content/drive/MyDrive/MRP/Task_1_Classification",
        "/content/drive/My Drive/Task_1_Classification",
        "/content/drive/My Drive/MRP/Task_1_Classification"
    ]
    
    BASE_DIR = None
    for path in possible_dirs:
        if os.path.exists(path):
            BASE_DIR = path
            print(f"Auto-detected workspace directory in Google Drive: {BASE_DIR}")
            break
            
    if BASE_DIR is None:
        BASE_DIR = "/content/drive/MyDrive/Task_1_Classification"
        print(f"Warning: 'Task_1_Classification' folder not found. Falling back to: {BASE_DIR}")
else:
    # Local Mac Path
    BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"

LOCAL_DATA_DIRS = [
    "/content/Task_1_Classification/Data/Augmented",
    "/content/Data/Augmented",
    "/content/Augmented"
]

DATA_DIR = None
if IN_COLAB:
    for path in LOCAL_DATA_DIRS:
        if os.path.exists(path):
            DATA_DIR = path
            print(f"[+] Detected high-speed local data directory at: {DATA_DIR}")
            break

if DATA_DIR is None:
    DATA_DIR = os.path.join(BASE_DIR, "Data", "Augmented")
    print(f"[*] Using data directory: {DATA_DIR}")

TEST_DIR = os.path.join(DATA_DIR, "test")

# Search for the model weights locally first, fallback to Google Drive
LOCAL_MODEL_PATH = "/content/Models/model3_convnext.pth"
DRIVE_MODEL_PATH = os.path.join(BASE_DIR, "Models", "model3_convnext.pth")

if IN_COLAB and os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
    print(f"[+] Found model checkpoint locally at: {MODEL_PATH}")
else:
    MODEL_PATH = DRIVE_MODEL_PATH
    print(f"[*] Loading model checkpoint from Drive: {MODEL_PATH}")

# Evaluation output paths (save directly to Google Drive)
EVAL_DIR = os.path.join(BASE_DIR, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Device configuration (CUDA for Colab, MPS for Mac, CPU fallback)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device for evaluation: {device}")

PATTERN_CLASSES = ["Gunshot", "Impact Spatter", "Passive Drip", "Transfer/Wipe"]
MECHANISM_CLASSES = ["Passive", "Low Velocity", "Medium/High Velocity"]

# --- BPA Multi-Task Model Definition ---
class BPAMultiTaskConvNeXt(nn.Module):
    def __init__(self, num_pattern=4, num_mechanism=3):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        
        self.pattern_head = nn.Linear(num_features, num_pattern)
        self.mechanism_head = nn.Linear(num_features, num_mechanism)

    def forward(self, x):
        features = self.backbone(x)
        return {
            'pattern': self.pattern_head(features),
            'mechanism': self.mechanism_head(features)
        }

# --- BPA Dataset Class ---
class BPADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Mapping for Multi-Task labels (matches training class mapping exactly)
        # Cough_Spatter is excluded from the dataset entirely.
        self.class_map = {
            "Gunshot":        (0, 2),
            "Impact_Spatter": (1, 2),
            "Passive_Drip":   (2, 0),
            "Transfer_Wipe":  (3, 1)
        }
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
            
        for cls_name, (p_idx, m_idx) in self.class_map.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append((p_idx, m_idx))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        p_label, m_label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, {
            'pattern': torch.tensor(p_label, dtype=torch.long),
            'mechanism': torch.tensor(m_label, dtype=torch.long)
        }

# --- Plotting Helpers ---
def plot_confusion_matrix(cm, classes, title, save_path, cmap="Blues"):
    """Plots an annotated confusion matrix using Seaborn."""
    plt.figure(figsize=(8, 6))
    
    # Calculate percentage annotation values
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = [f"{val}\n({perc:.1f}%)" for val, perc in zip(cm.flatten(), cm_perc.flatten())]
    labels = np.asarray(labels).reshape(cm.shape[0], cm.shape[1])
    
    sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 11, "weight": "bold"})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Class label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to: {save_path}")

def plot_multiclass_roc(y_true, y_probs, classes, title, save_path):
    """Plots multi-class one-vs-rest ROC curves with macro-average details."""
    plt.figure(figsize=(9, 7))
    
    n_classes = len(classes)
    
    # Plot curve for each individual class
    for i in range(n_classes):
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_prob_class = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.4f})')
        
    # Plot macro-average curve
    all_fpr = np.unique(np.concatenate([roc_curve((np.array(y_true) == c).astype(int), y_probs[:, c])[0] for c in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve((np.array(y_true) == c).astype(int), y_probs[:, c])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    plt.plot(all_fpr, mean_tpr, label=f'Macro-Average (AUC = {macro_auc:.4f})',
             color='black', linestyle=':', lw=3)
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC-AUC Curve saved to: {save_path}")

# --- Core Evaluation Pipeline ---
def evaluate_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading test dataset...")
    try:
        test_dataset = BPADataset(TEST_DIR, transform=transform)
    except Exception as e:
        print(f"[!] Error loading test directory: {e}")
        return
        
    print(f"Found {len(test_dataset)} test images.")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # --- Prepare Balanced Test Subset (up to 30 samples per class) ---
    import random
    from collections import defaultdict
    from torch.utils.data import Subset
    
    random.seed(42)
    class_indices = defaultdict(list)
    for idx, (p_idx, _) in enumerate(test_dataset.labels):
        class_indices[p_idx].append(idx)
        
    balanced_indices = []
    for p_idx, indices in class_indices.items():
        sample_size = min(30, len(indices))
        balanced_indices.extend(random.sample(indices, sample_size))
        
    balanced_dataset = Subset(test_dataset, balanced_indices)
    balanced_loader = DataLoader(balanced_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"[+] Constructed balanced test subset of size: {len(balanced_indices)}")
    
    print(f"Loading trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model file not found at {MODEL_PATH}.")
        print("Please train the model first and place the weights here.")
        return
        
    model = BPAMultiTaskConvNeXt(num_pattern=4, num_mechanism=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Helper to run inference
    def run_eval(loader):
        all_true_p, all_pred_p, all_probs_p = [], [], []
        all_true_m, all_pred_m, all_probs_m = [], [], []
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                p_targets = targets['pattern'].numpy()
                m_targets = targets['mechanism'].numpy()
                outputs = model(images)
                
                p_probs = softmax(outputs['pattern']).cpu().numpy()
                p_preds = np.argmax(p_probs, axis=1)
                
                m_probs = softmax(outputs['mechanism']).cpu().numpy()
                m_preds = np.argmax(m_probs, axis=1)
                
                all_true_p.extend(p_targets)
                all_pred_p.extend(p_preds)
                all_probs_p.append(p_probs)
                
                all_true_m.extend(m_targets)
                all_pred_m.extend(m_preds)
                all_probs_m.append(m_probs)
                
        return (np.array(all_true_p), np.array(all_pred_p), np.vstack(all_probs_p),
                np.array(all_true_m), np.array(all_pred_m), np.vstack(all_probs_m))

    # --- Pass A: Imbalanced Evaluation ---
    print("\nRunning inference on RAW (Imbalanced) test dataset...")
    true_p_raw, pred_p_raw, probs_p_raw, true_m_raw, pred_m_raw, probs_m_raw = run_eval(test_loader)
    
    report_p_raw = classification_report(true_p_raw, pred_p_raw, target_names=PATTERN_CLASSES, digits=4, zero_division=0)
    report_m_raw = classification_report(true_m_raw, pred_m_raw, target_names=MECHANISM_CLASSES, digits=4, zero_division=0)
    
    print("\n" + "="*60)
    print("  RAW TEST PATTERN CLASSIFICATION REPORT  (4-class)")
    print("="*60)
    print(report_p_raw)
    
    # Save raw report
    raw_report_path = os.path.join(EVAL_DIR, "convnext_classification_report.txt")
    with open(raw_report_path, "w") as f:
        f.write("=================================================================\n")
        f.write("  BPA MULTI-TASK CONVNEXT — RAW TEST REPORT\n")
        f.write("=================================================================\n\n")
        f.write("--- Pattern Classification (4 Classes) ---\n")
        f.write(report_p_raw + "\n\n")
        f.write("--- Force Mechanism Classification (3 Classes) ---\n")
        f.write(report_m_raw + "\n")
    print(f"Raw test report saved to: {raw_report_path}")
    
    # --- Pass B: Balanced Evaluation ---
    print("\nRunning inference on BALANCED test dataset...")
    true_p_bal, pred_p_bal, probs_p_bal, true_m_bal, pred_m_bal, probs_m_bal = run_eval(balanced_loader)
    
    report_p_bal = classification_report(true_p_bal, pred_p_bal, target_names=PATTERN_CLASSES, digits=4, zero_division=0)
    report_m_bal = classification_report(true_m_bal, pred_m_bal, target_names=MECHANISM_CLASSES, digits=4, zero_division=0)
    
    print("\n" + "="*60)
    print("  BALANCED TEST PATTERN CLASSIFICATION REPORT  (4-class)")
    print("="*60)
    print(report_p_bal)
    
    # Save balanced report
    bal_report_path = os.path.join(EVAL_DIR, "convnext_balanced_classification_report.txt")
    with open(bal_report_path, "w") as f:
        f.write("=================================================================\n")
        f.write("  BPA MULTI-TASK CONVNEXT — BALANCED TEST REPORT\n")
        f.write("=================================================================\n\n")
        f.write("--- Pattern Classification (4 Classes) ---\n")
        f.write(report_p_bal + "\n\n")
        f.write("--- Force Mechanism Classification (3 Classes) ---\n")
        f.write(report_m_bal + "\n")
    print(f"Balanced test report saved to: {bal_report_path}")
    
    # Generate Confusion Matrices & ROC Curves for Raw Test Set
    cm_p = confusion_matrix(true_p_raw, pred_p_raw)
    cm_m = confusion_matrix(true_m_raw, pred_m_raw)
    
    plot_confusion_matrix(cm_p, PATTERN_CLASSES,
                          "ConvNeXt: Pattern Confusion Matrix (Raw)",
                          os.path.join(EVAL_DIR, "convnext_pattern_confusion_matrix.png"),
                          cmap="Blues")
                          
    plot_confusion_matrix(cm_m, MECHANISM_CLASSES,
                          "ConvNeXt: Mechanism Confusion Matrix (Raw)",
                          os.path.join(EVAL_DIR, "convnext_mechanism_confusion_matrix.png"),
                          cmap="Greens")
                          
    plot_multiclass_roc(true_p_raw, probs_p_raw, PATTERN_CLASSES,
                        "ConvNeXt: Pattern ROC Curves (Raw)",
                        os.path.join(EVAL_DIR, "convnext_pattern_roc_auc.png"))
                        
    plot_multiclass_roc(true_m_raw, probs_m_raw, MECHANISM_CLASSES,
                        "ConvNeXt: Mechanism ROC Curves (Raw)",
                        os.path.join(EVAL_DIR, "convnext_mechanism_roc_auc.png"))
                        
    print("\n[+] ConvNeXt evaluation complete!")
    print(f"    All reports and plots saved inside: {EVAL_DIR}")

if __name__ == "__main__":
    evaluate_model()
