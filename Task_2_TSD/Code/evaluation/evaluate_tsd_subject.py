import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import timm

# Import custom BloodNet definition
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
from bloodnet import bloodnet50

class TSDSubjectDataset(Dataset):
    """Custom PyTorch Dataset for Task 2 TSD that returns image, label, and rabbit_id."""
    def __init__(self, data_dir, name_to_rabbit, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        self.class_mapping = {
            '1d_test': 0, '1d_train': 0, '1d_val': 0,
            '7d_test': 1, '7d_train': 1, '7d_val': 1,
            '14d_test': 1, '14d_train': 1, '14d_val': 1,
            '21d_test': 2, '21d_train': 2, '21d_val': 2,
            '28d_test': 2, '28d_train': 2, '28d_val': 2
        }
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in self.class_mapping:
                target_class = self.class_mapping[folder_name]
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        base = os.path.splitext(file_name)[0].lower()
                        rabbit_id = name_to_rabbit.get(base, 'UNKNOWN')
                        self.samples.append({
                            'path': os.path.join(folder_path, file_name),
                            'label': target_class,
                            'rabbit_id': rabbit_id
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, sample['label'], sample['rabbit_id']

def wilson_ci(correct, total, confidence=0.95):
    """Calculates Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    z = 1.96  # 95% confidence level
    denominator = 1 + z**2 / total
    centre_adj_p = p + z**2 / (2 * total)
    adjusted_variance = z * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    lower = (centre_adj_p - adjusted_variance) / denominator
    upper = (centre_adj_p + adjusted_variance) / denominator
    return max(0.0, lower), min(1.0, upper)

def evaluate_model_by_subject(model, data_loader, device, is_baseline=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, labels, rabbit_ids in data_loader:
            images = images.to(device)
            outputs = model(images)
            
            if is_baseline:
                probs_5 = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_3 = np.zeros((probs_5.shape[0], 3))
                probs_3[:, 0] = probs_5[:, 0]
                probs_3[:, 1] = probs_5[:, 1] + probs_5[:, 2]
                probs_3[:, 2] = probs_5[:, 2] + probs_5[:, 3] + probs_5[:, 4]
                # Normalize
                probs_sum = np.sum(probs_3, axis=1, keepdims=True)
                probs_3 = probs_3 / (probs_sum + 1e-10)
                preds = np.argmax(probs_3, axis=1)
            else:
                probs_3 = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs_3, axis=1)
                
            labels_np = labels.numpy()
            for pred, label, rabbit_id in zip(preds, labels_np, rabbit_ids):
                results.append({
                    'rabbit_id': rabbit_id,
                    'correct': int(pred == label)
                })
                
    return pd.DataFrame(results)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[+] Using device: {device}")
    
    # Load metadata CSV to map names to rabbits
    csv_path = "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Text/BloodNet_50k_Images/bloodstain_information.csv"
    print(f"Loading metadata CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Name_lower'] = df['Name'].str.lower()
    name_to_rabbit = dict(zip(df['Name_lower'], df['rabbit_ID']))
    
    # Test directory path
    test_dir = "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Text/BloodNet_50k_Images/data/outside_test"
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading subject test dataset...")
    test_dataset = TSDSubjectDataset(test_dir, name_to_rabbit, transform=val_transform)
    print(f"Loaded {len(test_dataset)} test samples.")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Load all models
    models_to_eval = [
        ("bloodnet_baseline", lambda: bloodnet50(num_classes=5), "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Text/BloodNet_50k_Images/bloodnet50_new.pth", True, "BloodNet Baseline"),
        ("efficientnet_tsd", lambda: models.efficientnet_b0(weights=None), "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Models/best_tsd_model_efficientnet.pth", False, "Trained EfficientNet-B0"),
        ("convnext_tsd", lambda: timm.create_model('convnext_tiny', pretrained=False, num_classes=3), "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Models/best_tsd_model_convnext.pth", False, "Trained ConvNeXt-Tiny"),
        ("resnet50_tsd", lambda: bloodnet50(num_classes=3), "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Models/best_tsd_model_resnet50.pth", False, "Fine-Tuned ResNet-50 (CBAM)"),
    ]
    
    # Keep track of results for all models
    all_subject_metrics = []
    
    for name, model_fn, weights_path, is_baseline, display_name in models_to_eval:
        print(f"\nEvaluating model: {display_name}...")
        model = model_fn()
        if name == "efficientnet_tsd":
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 3)
            
        if not os.path.exists(weights_path):
            print(f"[!] Weights not found at {weights_path}, skipping.")
            continue
            
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model = model.to(device)
        
        # Evaluate model predictions grouped by rabbit ID
        eval_df = evaluate_model_by_subject(model, test_loader, device, is_baseline=is_baseline)
        
        # Group by subject and calculate accuracies
        grouped = eval_df.groupby('rabbit_id').agg(
            correct=('correct', 'sum'),
            total=('correct', 'count')
        ).reset_index()
        
        # Calculate overall stats too
        overall_correct = eval_df['correct'].sum()
        overall_total = len(eval_df)
        
        # Append overall stats
        row_overall = {
            'Model': display_name,
            'Subject': 'Overall Test Set',
            'Correct': overall_correct,
            'Total': overall_total,
            'Accuracy': overall_correct / overall_total
        }
        low, high = wilson_ci(overall_correct, overall_total)
        row_overall['95% CI'] = f"[{low*100:.2f}%, {high*100:.2f}%]"
        all_subject_metrics.append(row_overall)
        
        # Append per-subject stats
        for _, row in grouped.iterrows():
            sub = row['rabbit_id']
            corr = row['correct']
            tot = row['total']
            acc = corr / tot
            low, high = wilson_ci(corr, tot)
            all_subject_metrics.append({
                'Model': display_name,
                'Subject': sub,
                'Correct': corr,
                'Total': tot,
                'Accuracy': acc,
                '95% CI': f"[{low*100:.2f}%, {high*100:.2f}%]"
            })
            
    # Save output to CSV and show it
    results_df = pd.DataFrame(all_subject_metrics)
    output_csv = "/Users/shahidabatool/Desktop/MRP/Task_2_TSD/Evaluation/tsd_subject_performance.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n[+] Saved subject performance metrics to {output_csv}")
    
    # Format the table for the report
    print("\nSubject-Level Performance Table:")
    for model_name in results_df['Model'].unique():
        print(f"\nModel: {model_name}")
        model_rows = results_df[results_df['Model'] == model_name]
        for _, row in model_rows.iterrows():
            print(f"  {row['Subject']}: Accuracy = {row['Accuracy']*100:.2f}%  95% CI = {row['95% CI']} (N={row['Total']})")

if __name__ == "__main__":
    main()
