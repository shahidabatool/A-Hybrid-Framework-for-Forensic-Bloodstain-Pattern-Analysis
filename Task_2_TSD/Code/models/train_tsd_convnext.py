import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

class TSDDataset(Dataset):
    """Custom PyTorch Dataset for Task 2 TSD to map 5 aging intervals to 3 forensic classes."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Subdirectories representing the 5 intervals
        self.class_mapping = {
            '1d_train': 0, '1d_val': 0, '1d_test': 0,          # Fresh
            '7d_train': 1, '7d_val': 1, '7d_test': 1,          # Intermediate
            '14d_train': 1, '14d_val': 1, '14d_test': 1,       # Intermediate
            '21d_train': 2, '21d_val': 2, '21d_test': 2,       # Aged
            '28d_train': 2, '28d_val': 2, '28d_test': 2        # Aged
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
                        
        print(f"[+] Loaded {len(self.samples)} images from {data_dir}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        dist = {0: 0, 1: 0, 2: 0}
        for _, label in self.samples:
            dist[label] += 1
        print(f"    Class distribution -> Fresh: {dist[0]}, Intermediate: {dist[1]}, Aged: {dist[2]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def main():
    parser = argparse.ArgumentParser(description="Train Task 2 TSD ConvNeXt Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to split data directory (e.g. data/train1)")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation split data directory (e.g. data/test)")
    parser.add_argument("--output_dir", type=str, default="./Models", help="Directory to save checkpoint models")
    parser.add_argument("--epochs", type=str, default="5", help="Number of training epochs")
    parser.add_argument("--batch_size", type=str, default="64", help="Training batch size")
    parser.add_argument("--lr", type=str, default="1e-4", help="Learning rate")
    args = parser.parse_args()

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # Image Transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets & Dataloaders
    print("Loading training dataset...")
    train_dataset = TSDDataset(args.data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print("Loading validation dataset...")
    val_dataset = TSDDataset(args.val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load ConvNeXt-Tiny pre-trained model
    print("Initializing ConvNeXt-Tiny with ImageNet weights...")
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=3)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("Starting Training Loop...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += images.size(0)
            
        epoch_train_loss = running_loss / total_preds
        epoch_train_acc = (correct_preds.double() / total_preds).item()
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_preds = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                
                running_val_loss += val_loss.item() * val_images.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                correct_val_preds += torch.sum(val_preds == val_labels.data)
                total_val_preds += val_images.size(0)
                
        epoch_val_loss = running_val_loss / total_val_preds
        epoch_val_acc = (correct_val_preds.double() / total_val_preds).item()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) -> "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
              
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        # Checkpoint saving
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_path = os.path.join(args.output_dir, "best_tsd_model_convnext.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"    [+] Saved new best model checkpoint to {best_model_path} (Val Acc: {epoch_val_acc*100:.2f}%)")

    # Save final curves
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, "tsd_convnext_training_history.csv"), index=False)
    print("Training Complete!")

if __name__ == "__main__":
    main()
