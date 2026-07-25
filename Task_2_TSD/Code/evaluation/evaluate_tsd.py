import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import timm

# Import custom architectures
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
from bloodnet import bloodnet50

class TSDDataset(Dataset):
    """Custom PyTorch Dataset for Task 2 TSD to map 5 aging intervals to 3 forensic classes."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Subdirectories representing the 5 intervals
        self.class_mapping = {
            '1d_test': 0, '1d_train': 0, '1d_val': 0,          # Fresh
            '7d_test': 1, '7d_train': 1, '7d_val': 1,          # Intermediate
            '14d_test': 1, '14d_train': 1, '14d_val': 1,       # Intermediate
            '21d_test': 2, '21d_train': 2, '21d_val': 2,       # Aged
            '28d_test': 2, '28d_train': 2, '28d_val': 2        # Aged
        }
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in self.class_mapping:
                target_class = self.class_mapping[folder_name]
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(folder_path, file_name),
                            target_class
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def extract_balanced_subset(dataset, samples_per_class=100):
    """Extracts a balanced subset from the dataset to compute unbiased metrics."""
    np.random.seed(42)
    indices_by_class = {0: [], 1: [], 2: []}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(idx)
        
    balanced_indices = []
    for label, idxs in indices_by_class.items():
        if len(idxs) < samples_per_class:
            print(f"[!] Warning: Class {label} only has {len(idxs)} samples. Selecting all.")
            balanced_indices.extend(idxs)
        else:
            selected = np.random.choice(idxs, samples_per_class, replace=False)
            balanced_indices.extend(selected)
            
    # Create a subset dataset
    subset = torch.utils.data.Subset(dataset, balanced_indices)
    return subset

def plot_confusion_matrix(y_true, y_pred, classes, output_path, title):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                cbar=False, annot_kws={"size": 14})
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_auc(y_true, y_probs, classes, output_path, title):
    plt.figure(figsize=(7, 6))
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=lw,
                 label=f'{classes[i]} (AUC = {roc_auc:.4f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def evaluate_model(model, data_loader, device, is_baseline=False):
    """Runs inference and returns true labels, predicted labels, and probabilities."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            outputs = model(images)
            
            if is_baseline:
                # Baseline model outputs 5 logits. Convert to probabilities via Softmax
                probs_5 = torch.softmax(outputs, dim=1).cpu().numpy()
                # Map 5 classes to 3 classes:
                # Class 0 (Fresh): prob[0]
                # Class 1 (Intermediate): prob[1] + prob[2]
                # Class 2 (Aged): prob[3] + prob[4]
                probs_3 = np.zeros((probs_5.shape[0], 3))
                probs_3[:, 0] = probs_5[:, 0]
                probs_3[:, 1] = probs_5[:, 1] + probs_5[:, 2]
                probs_3[:, 2] = probs_5[:, 2] + probs_5[:, 3] + probs_5[:, 4]  # Note: sum remaining to make 1.0
                
                # Normalize probabilities to sum to 1.0
                probs_sum = np.sum(probs_3, axis=1, keepdims=True)
                probs_3 = probs_3 / (probs_sum + 1e-10)
                
                preds = np.argmax(probs_3, axis=1)
                all_probs.extend(probs_3)
                all_preds.extend(preds)
            else:
                probs_3 = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs_3, axis=1)
                all_probs.extend(probs_3)
                all_preds.extend(preds)
                
            all_targets.extend(labels.numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 TSD Models")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to holdout test directory (e.g. data/outside_test)")
    parser.add_argument("--baseline_weights", type=str, required=True, help="Path to baseline bloodnet50_new.pth weights")
    parser.add_argument("--resnet50_weights", type=str, required=True, help="Path to our fine-tuned ResNet-50 weights")
    parser.add_argument("--efficientnet_weights", type=str, required=True, help="Path to our trained EfficientNet-B0 weights")
    parser.add_argument("--convnext_weights", type=str, default=None, help="Path to our trained ConvNeXt weights")
    parser.add_argument("--output_dir", type=str, default="./Evaluation", help="Directory to save evaluation reports and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Evaluating on device: {device}")

    # Image Transforms
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    print("Loading test dataset...")
    full_test_dataset = TSDDataset(args.test_dir, transform=val_transform)
    balanced_test_dataset = extract_balanced_subset(full_test_dataset, samples_per_class=100)

    # Dataloaders
    batch_size = 128
    full_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    balanced_loader = DataLoader(balanced_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = ["Fresh", "Intermediate", "Aged"]

    convnext_weights_path = args.convnext_weights if args.convnext_weights is not None else os.path.join(os.path.dirname(args.resnet50_weights), "best_tsd_model_convnext.pth")

    # List of evaluations to run:
    # (model_name, model_instantiation_func, weights_path, is_baseline)
    eval_configs = [
        ("bloodnet_baseline", lambda: bloodnet50(num_classes=5), args.baseline_weights, True, "BloodNet Baseline"),
        ("resnet50_tsd", lambda: bloodnet50(num_classes=3), args.resnet50_weights, False, "Fine-Tuned ResNet-50"),
        ("efficientnet_tsd", lambda: models.efficientnet_b0(weights=None), args.efficientnet_weights, False, "Trained EfficientNet-B0"),
        ("convnext_tsd", lambda: timm.create_model('convnext_tiny', pretrained=False, num_classes=3), convnext_weights_path, False, "Trained ConvNeXt-Tiny")
    ]

    for name, model_fn, weights_path, is_baseline, display_name in eval_configs:
        print(f"\n==================================================")
        print(f"  Evaluating {display_name}...")
        print(f"==================================================")
        
        # Instantiate and load model
        model = model_fn()
        if name == "efficientnet_tsd":
            # Modify classifier before loading state dict
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 3)
            
        if not os.path.exists(weights_path):
            print(f"[!] Warning: Weights not found at {weights_path}. Skipping evaluation.")
            continue
            
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(device)

        # Pass 1: Raw Test Set
        print(f"Running Raw Test Set evaluation...")
        y_true, y_pred, y_probs = evaluate_model(model, full_loader, device, is_baseline=is_baseline)
        
        # Save reports
        report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        report_path = os.path.join(args.output_dir, f"{name}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"=================================================\n")
            f.write(f"  TSD EVALUATION REPORT — {display_name.upper()} (RAW TEST)\n")
            f.write(f"=================================================\n\n")
            f.write(report_str)
        print(f"    [+] Saved raw report to {report_path}")
        
        # Save plots
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            os.path.join(args.output_dir, f"{name}_confusion_matrix.png"),
            f"{display_name} Confusion Matrix (Raw Test)"
        )
        plot_roc_auc(
            y_true, y_probs, class_names,
            os.path.join(args.output_dir, f"{name}_roc_auc.png"),
            f"{display_name} ROC Curves (Raw Test)"
        )

        # Pass 2: Balanced Holdout Set
        print(f"Running Balanced Holdout Set evaluation...")
        y_true_bal, y_pred_bal, y_probs_bal = evaluate_model(model, balanced_loader, device, is_baseline=is_baseline)
        
        report_bal_str = classification_report(y_true_bal, y_pred_bal, target_names=class_names, digits=4)
        report_bal_path = os.path.join(args.output_dir, f"{name}_balanced_classification_report.txt")
        with open(report_bal_path, "w") as f:
            f.write(f"=================================================\n")
            f.write(f"  TSD EVALUATION REPORT — {display_name.upper()} (BALANCED TEST)\n")
            f.write(f"=================================================\n\n")
            f.write(report_bal_str)
        print(f"    [+] Saved balanced report to {report_bal_path}")

    print("\n[+] TSD Evaluation complete! All reports and plots saved to:", args.output_dir)

if __name__ == "__main__":
    main()
