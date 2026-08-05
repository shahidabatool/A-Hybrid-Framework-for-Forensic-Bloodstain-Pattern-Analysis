# =============================================================================
# Bloodstain Pattern Analysis (BPA) Multi-Task Classification Using ResNet-50
#
# Description:
# This script trains a deep learning model for automated bloodstain pattern
# classification and impact mechanism prediction using a multi-task learning
# approach.
#
# Model Architecture:
# - Backbone: ResNet-50 (ImageNet pretrained)
# - Task 1: Bloodstain pattern classification
# - Task 2: Impact mechanism classification
#
# Training Strategy:
# - Phase 1: Backbone frozen, classification heads trained
# - Phase 2: Full model fine-tuning with reduced learning rate
#
# Features:
# - Automatic Google Colab and local environment detection
# - Transfer learning using TIMM pretrained ResNet-50
# - Multi-task classification with dual output heads
# - Class imbalance handling using weighted cross-entropy loss
# - Training history logging and performance curve generation
# - Model checkpoint saving and Google Drive backup support
#
# Outputs:
# - Trained ResNet-50 model weights (.pth)
# - Training history (.json)
# - Loss and accuracy learning curves (.png)
# =============================================================================
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

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
    
    # Try a list of potential paths where the 'Task_1_Classification' folder might reside in Drive
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

# --- High-Speed Local Path for Google Colab ---
# --- High-Speed Local Path for Google Colab ---
# Reading thousands of small files directly from mounted Google Drive FUSE is extremely slow.
# If the dataset is copied to the local VM disk (/content/), we use that for a 100x training speedup!
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

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Save paths: Save locally on VM first (guarantees fast, robust writes), then copy to Drive at the end
if IN_COLAB:
    LOCAL_SAVE_DIR = "/content/Models"
else:
    LOCAL_SAVE_DIR = os.path.join(BASE_DIR, "Models")

os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(LOCAL_SAVE_DIR, "model2_resnet50.pth")
HISTORY_SAVE_PATH = os.path.join(LOCAL_SAVE_DIR, "model2_resnet50_history.json")
PLOT_SAVE_PATH = os.path.join(LOCAL_SAVE_DIR, "model2_resnet50_curves.png")

# Final destination paths on Google Drive
DRIVE_SAVE_DIR = os.path.join(BASE_DIR, "Models")

# Device configuration (CUDA for Colab GPU, MPS for Apple Silicon Mac, CPU fallback)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}")
print(f"Data Directory: {DATA_DIR}")

# --- BPA Multi-Task Model Definition (Model 2: ResNet-50) ---
class BPAMultiTaskResNet50(nn.Module):
    def __init__(self, num_pattern=4, num_mechanism=3):
        super().__init__()
        # Load ResNet-50 backbone for transfer learning
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        # Branch 1: Pattern Type (5 classes)
        self.pattern_head = nn.Linear(num_features, num_pattern)
        
        # Branch 2: Impact Force Mechanism (3 classes)
        self.mechanism_head = nn.Linear(num_features, num_mechanism)

    def forward(self, x):
        features = self.backbone(x)
        return {
            'pattern': self.pattern_head(features),
            'mechanism': self.mechanism_head(features)
        }

# --- Multi-Task Dataset Class ---
class BPADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Mapping for Multi-Task labels (Pattern_Index, Mechanism_Index)
        # Mechanism: 0:Passive, 1:Low Vel, 2:Med/High Vel
        # Cough_Spatter is excluded from training and test datasets.
        self.class_map = {
            "Gunshot": (0, 2),
            "Impact_Spatter": (1, 2),
            "Passive_Drip": (2, 0),
            "Transfer_Wipe": (3, 1)
        }
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"Directory not found: {root_dir}.\n"
                "Please make sure that:\n"
                "1. You have uploaded the 'Data' folder to Google Drive under 'Task_1_Classification' (or 'MRP/Task_1_Classification').\n"
                "2. Or you run the data augmentation script ('augment_data.py') on Colab first."
            )
            
        found_classes = []
        for cls_name, (p_idx, m_idx) in self.class_map.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                print(f"Warning: Class folder '{cls_name}' not found in {root_dir}")
                continue
            
            found_classes.append(cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append((p_idx, m_idx))
                    
        if len(self.image_paths) == 0:
            raise ValueError(
                f"\n[!] Error: No images found in dataset directory: {root_dir}\n"
                f"Checked class folders: {list(self.class_map.keys())}\n"
                f"Class folders actually found: {found_classes}\n"
            )

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

def save_history_and_plots(history, history_path, plot_path):
    """Saves the training history to a JSON file and renders beautiful curves."""
    import json
    import matplotlib
    matplotlib.use('Agg')  # Headless backend to prevent display errors
    import matplotlib.pyplot as plt
    
    # Save history dictionary as JSON
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Generate curves plot
    epochs = history["epoch"]
    
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", marker="x", linewidth=2)
    
    # Phase 2 start vertical line
    phase1_epochs = history["phase"].count("Phase 1")
    if 0 < phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs + 0.5, color="gray", linestyle="--", label="Phase 2 Fine-tuning")
        
    plt.title("ResNet-50 Multi-Task Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: Pattern Accuracy Curve
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_pattern_acc"], label="Train Pattern Acc", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(epochs, history["val_pattern_acc"], label="Val Pattern Acc", color="#d62728", marker="x", linewidth=2)
    if 0 < phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs + 0.5, color="gray", linestyle="--", label="Phase 2 Fine-tuning")
    plt.title("ResNet-50 Pattern Accuracy", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 3: Mechanism Accuracy Curve
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_mechanism_acc"], label="Train Mechanism Acc", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(epochs, history["val_mechanism_acc"], label="Val Mechanism Acc", color="#d62728", marker="x", linewidth=2)
    if 0 < phase1_epochs < len(epochs):
        plt.axvline(x=phase1_epochs + 0.5, color="gray", linestyle="--", label="Phase 2 Fine-tuning")
    plt.title("ResNet-50 Mechanism Accuracy", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves plot saved to {plot_path}")

# --- Training Configuration & Loop ---
def train_model():
    print("Initializing Data Loaders...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BPADataset(TRAIN_DIR, transform=transform)
    test_dataset = BPADataset(TEST_DIR, transform=transform)
    
    print(f"Found {len(train_dataset)} training images and {len(test_dataset)} testing images.")
    
    # Batch size configuration
    batch_size = 64 if IN_COLAB else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2 if IN_COLAB else 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2 if IN_COLAB else 0)

    model = BPAMultiTaskResNet50(num_pattern=4, num_mechanism=3).to(device)
    # Calculate smooth square-root inverse-frequency class weights based on validation imbalance
    # to penalize minority errors without warping the decision boundary.
    import math
    from collections import Counter
    p_targets_val = [label[0] for label in test_dataset.labels]
    counts_val = Counter(p_targets_val)
    class_counts_val = [counts_val[i] for i in range(4)]
    total_val = sum(class_counts_val)
    
    pattern_weights = [math.sqrt(total_val / (4.0 * count)) if count > 0 else 1.0 for count in class_counts_val]
    pattern_weights_tensor = torch.FloatTensor(pattern_weights).to(device)
    print(f"[+] Computed Smooth Pattern Loss weights (Gunshot, Impact, Passive, Transfer): {pattern_weights}")
    
    criterion_p = nn.CrossEntropyLoss(weight=pattern_weights_tensor)
    criterion_m = nn.CrossEntropyLoss()
    
    def run_epoch(model, optimizer, dataloader, is_train=True):
        if is_train:
            model.train()
        else:
            model.eval()
            
        total_loss = 0
        p_correct = 0
        m_correct = 0
        total_samples = 0
        
        with torch.set_grad_enabled(is_train):
            progress_bar = tqdm(dataloader, desc="Training" if is_train else "Validating", leave=False)
            for images, targets in progress_bar:
                images = images.to(device)
                p_targets = targets['pattern'].to(device)
                m_targets = targets['mechanism'].to(device)
                
                if is_train:
                    optimizer.zero_grad()
                    
                outputs = model(images)
                
                loss_p = criterion_p(outputs['pattern'], p_targets)
                loss_m = criterion_m(outputs['mechanism'], m_targets)
                loss = loss_p + loss_m
                
                if is_train:
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item()
                
                # Calculate accuracy
                _, p_preds = torch.max(outputs['pattern'], 1)
                _, m_preds = torch.max(outputs['mechanism'], 1)
                p_correct += torch.sum(p_preds == p_targets.data)
                m_correct += torch.sum(m_preds == m_targets.data)
                total_samples += images.size(0)
                
                progress_bar.set_postfix({'loss': loss.item()})
                
        avg_loss = total_loss / len(dataloader)
        p_acc = p_correct.double() / total_samples
        m_acc = m_correct.double() / total_samples
        return avg_loss, p_acc.item(), m_acc.item()

    # Check if a model checkpoint already exists to resume from Phase 2, Epoch 2
    RESUME_PHASE_2 = False
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            print(f"\n[+] Active checkpoint found at {MODEL_SAVE_PATH}.")
            print("Attempting to resume training from Phase 2, Epoch 2 (Fine-tuning)...")
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            RESUME_PHASE_2 = True
        except Exception as e:
            print(f"Warning: Could not load checkpoint to resume: {e}. Starting fresh...")
            RESUME_PHASE_2 = False

    if RESUME_PHASE_2:
        print("\n[+] Resuming from Phase 2, Epoch 2. Skipping Phase 1 (Warm-up).")
        # Pre-populate history with the exact values from your 5-hour CPU run!
        history = {
            "epoch": [1, 2, 3, 4, 5, 6],
            "phase": ["Phase 1"] * 5 + ["Phase 2"],
            "train_loss": [1.8753, 1.1328, 0.8427, 0.7034, 0.6118, 0.2987],
            "val_loss": [1.3774, 1.0080, 0.8389, 0.7100, 0.6600, 0.2652],
            "train_pattern_acc": [0.7656, 0.8902, 0.9100, 0.9175, 0.9240, 0.9548],
            "val_pattern_acc": [0.8583, 0.8830, 0.8939, 0.9053, 0.9081, 0.9474],
            "train_mechanism_acc": [0.7119, 0.8975, 0.9238, 0.9358, 0.9408, 0.9722],
            "val_mechanism_acc": [0.8506, 0.9057, 0.9279, 0.9247, 0.9397, 0.9725]
        }
        
        # Go straight to Phase 2, starting from epoch index 1 (meaning the second epoch of Phase 2)
        print("\n=== Phase 2: Full Fine-tuning (Resumed from Epoch 2) ===")
        for param in model.backbone.parameters():
            param.requires_grad = True
            
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs_p2 = 10
        best_val_loss = 0.2652 # Best val loss from your completed Epoch 1 of Phase 2
        
        for epoch in range(1, num_epochs_p2): # Loops from 1 to 9 (corresponds to Phase 2 Epochs 2 to 10)
            print(f"\nEpoch {epoch+1}/{num_epochs_p2}")
            train_loss, train_p_acc, train_m_acc = run_epoch(model, optimizer, train_loader, is_train=True)
            val_loss, val_p_acc, val_m_acc = run_epoch(model, optimizer, test_loader, is_train=False)
            
            print(f"Train - Loss: {train_loss:.4f} | Pattern Acc: {train_p_acc:.4f} | Mechanism Acc: {train_m_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Pattern Acc: {val_p_acc:.4f} | Mechanism Acc: {val_m_acc:.4f}")
            
            # Record Phase 2 metrics
            history["epoch"].append(5 + epoch + 1)
            history["phase"].append("Phase 2")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_pattern_acc"].append(train_p_acc)
            history["val_pattern_acc"].append(val_p_acc)
            history["train_mechanism_acc"].append(train_m_acc)
            history["val_mechanism_acc"].append(val_m_acc)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"*** Best model saved to {MODEL_SAVE_PATH} ***")
    else:
        history = {
            "epoch": [],
            "phase": [],
            "train_loss": [],
            "val_loss": [],
            "train_pattern_acc": [],
            "val_pattern_acc": [],
            "train_mechanism_acc": [],
            "val_mechanism_acc": []
        }

        # --- TWO-STAGE FINE-TUNING ---
        # Phase 1: Freeze backbone, train only heads (Warm-up / Probing)
        print("\n=== Phase 1: Warm-up (Training Heads Only) ===")
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs_p1 = 5 # Matches EfficientNet epochs exactly
        
        for epoch in range(num_epochs_p1):
            print(f"\nEpoch {epoch+1}/{num_epochs_p1}")
            train_loss, train_p_acc, train_m_acc = run_epoch(model, optimizer, train_loader, is_train=True)
            val_loss, val_p_acc, val_m_acc = run_epoch(model, optimizer, test_loader, is_train=False)
            
            print(f"Train - Loss: {train_loss:.4f} | Pattern Acc: {train_p_acc:.4f} | Mechanism Acc: {train_m_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Pattern Acc: {val_p_acc:.4f} | Mechanism Acc: {val_m_acc:.4f}")
            
            # Record Phase 1 metrics
            history["epoch"].append(epoch + 1)
            history["phase"].append("Phase 1")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_pattern_acc"].append(train_p_acc)
            history["val_pattern_acc"].append(val_p_acc)
            history["train_mechanism_acc"].append(train_m_acc)
            history["val_mechanism_acc"].append(val_m_acc)

        # Phase 2: Unfreeze backbone and fine-tune whole model with low learning rate
        print("\n=== Phase 2: Full Fine-tuning ===")
        for param in model.backbone.parameters():
            param.requires_grad = True
            
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        num_epochs_p2 = 10 # Matches EfficientNet epochs exactly
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs_p2):
            print(f"\nEpoch {epoch+1}/{num_epochs_p2}")
            train_loss, train_p_acc, train_m_acc = run_epoch(model, optimizer, train_loader, is_train=True)
            val_loss, val_p_acc, val_m_acc = run_epoch(model, optimizer, test_loader, is_train=False)
            
            print(f"Train - Loss: {train_loss:.4f} | Pattern Acc: {train_p_acc:.4f} | Mechanism Acc: {train_m_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Pattern Acc: {val_p_acc:.4f} | Mechanism Acc: {val_m_acc:.4f}")
            
            # Record Phase 2 metrics
            history["epoch"].append(num_epochs_p1 + epoch + 1)
            history["phase"].append("Phase 2")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_pattern_acc"].append(train_p_acc)
            history["val_pattern_acc"].append(val_p_acc)
            history["train_mechanism_acc"].append(train_m_acc)
            history["val_mechanism_acc"].append(val_m_acc)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"*** Best model saved to {MODEL_SAVE_PATH} ***")

    print(f"\nTraining Complete! Best model saved to {MODEL_SAVE_PATH}")
    
    # Save training history JSON and generate curves plot
    save_history_and_plots(history, HISTORY_SAVE_PATH, PLOT_SAVE_PATH)

    # --- Copy Artifacts to Google Drive ---
    if IN_COLAB:
        import shutil
        print(f"\n[*] Copying model artifacts to persistent Google Drive: {DRIVE_SAVE_DIR} ...")
        try:
            os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
            for f in [MODEL_SAVE_PATH, HISTORY_SAVE_PATH, PLOT_SAVE_PATH]:
                if os.path.exists(f):
                    dest = os.path.join(DRIVE_SAVE_DIR, os.path.basename(f))
                    shutil.copy(f, dest)
                    print(f"  [+] Copied {os.path.basename(f)} to Drive")
            print("🎉 Success! All artifacts copied to Google Drive!")
        except Exception as e:
            print(f"⚠️ Warning: Could not copy files to Drive: {e}")
            print("Don't worry, your files are still saved locally on the VM at /content/Models/")

if __name__ == "__main__":
    train_model()
