# =============================================================================
# Project: CNN-Based Bloodstain Pattern Classification with Impact and Temporal
#          Parameter Estimation
#
# Description:
# This script performs model interpretability and forensic validation for the
# trained EfficientNet-B0 multi-task bloodstain pattern classification model.
#
# Methodology:
# - Deep learning architecture: EfficientNet-B0
# - Multi-task classification:
#       1. Bloodstain Pattern Classification
#       2. Bloodstain Mechanism Classification
#
# Analysis Techniques:
# - Grad-CAM visualization for model decision interpretation.
# - Ellipse fitting for bloodstain morphology analysis.
# - Impact angle estimation using stain geometric properties.
#
# Outputs:
# - Grad-CAM heatmap visualization.
# - Ellipse fitting validation visualization.
#
# =============================================================================
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import timm

# Define model structure to load weights
class BPAMultiTaskEfficientNet(nn.Module):
    def __init__(self, num_pattern=4, num_mechanism=3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        self.pattern_head = nn.Linear(num_features, num_pattern)
        self.mechanism_head = nn.Linear(num_features, num_mechanism)

    def forward(self, x):
        features = self.backbone(x)
        return {
            'pattern': self.pattern_head(features),
            'mechanism': self.mechanism_head(features)
        }

# Grad-CAM classes to compute heatmaps
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        outputs = self.model(x)
        logits = outputs['pattern']
        score = logits[0, class_idx]
        score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of feature map channels
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to retain positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def generate_gradcam_plot():
    print("[+] Generating Grad-CAM heatmaps...")
    BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
    weights_path = os.path.join(BASE_DIR, "Models", "model1_efficientnet_b0.pth")
    test_dir = os.path.join(BASE_DIR, "Data", "Augmented", "test")
    eval_dir = os.path.join(BASE_DIR, "Evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cpu")
    model = BPAMultiTaskEfficientNet(num_pattern=4, num_mechanism=3)
    
    if not os.path.exists(weights_path):
        print(f"[-] Error: Weights not found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Target the last convolutional block of EfficientNet backbone
    target_layer = model.backbone.conv_head
    grad_cam = GradCAM(model, target_layer)

    # Class mappings
    classes = ["Gunshot", "Impact_Spatter", "Passive_Drip", "Transfer_Wipe"]
    
    fig, axes = plt.subplots(len(classes), 2, figsize=(8, 12))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i, cls_name in enumerate(classes):
        cls_folder = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_folder):
            print(f"[-] Test folder for {cls_name} not found.")
            continue
            
        img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            print(f"[-] No images found in {cls_folder}")
            continue
            
        # Select first image
        img_path = os.path.join(cls_folder, img_files[0])
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess for model
        input_tensor = transform(img).unsqueeze(0)
        
        # Run Grad-CAM
        cam = grad_cam(input_tensor, i)
        
        # Load image for overlay in OpenCV
        raw_img = cv2.imread(img_path)
        raw_img = cv2.resize(raw_img, (224, 224))
        
        # Resize heatmap and apply color map
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Plot
        axes[i, 0].imshow(raw_img_rgb)
        axes[i, 0].set_title(f"Original: {cls_name.replace('_', ' ')}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Grad-CAM Heatmap (Model Focus)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(eval_dir, "gradcam_efficientnet.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    grad_cam.remove_hooks()
    print(f"[+] Saved Grad-CAM validation plot to: {output_path}")

def generate_ellipse_fitting_plot():
    print("[+] Running physical validation (ellipse-fitting/angle calculation)...")
    BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
    test_dir = os.path.join(BASE_DIR, "Data", "Augmented", "test")
    eval_dir = os.path.join(BASE_DIR, "Evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    classes_to_test = ["Gunshot", "Impact_Spatter", "Passive_Drip"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, cls_name in enumerate(classes_to_test):
        cls_folder = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_folder):
            print(f"[-] Test folder for {cls_name} not found.")
            continue
            
        img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            continue
            
        img_path = os.path.join(cls_folder, img_files[0])
        img = cv2.imread(img_path)
        
        # Convert to grayscale and threshold to isolate droplets
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Background is white (255), blood is dark/red. Let's invert and threshold
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_img = img.copy()
        
        # Sort contours by area and pick the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        fit_count = 0
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 200: # Sufficient size for reliable ellipse fitting
                if len(cnt) >= 5: # cv2.fitEllipse requires at least 5 points
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (w, h), angle = ellipse
                    
                    # Ensure width is the minor axis, height is the major axis
                    minor = min(w, h)
                    major = max(w, h)
                    
                    if major > 0:
                        ratio = minor / major
                        # Calculate impact angle: theta = arcsin(w/l)
                        theta_rad = np.arcsin(ratio)
                        theta_deg = theta_rad * 180.0 / np.pi
                        
                        # Draw the fitted ellipse in green
                        cv2.ellipse(annotated_img, ellipse, (0, 255, 0), 2)
                        
                        # Draw center point
                        cv2.circle(annotated_img, (int(x), int(y)), 3, (0, 0, 255), -1)
                        
                        # Label the calculated angle
                        label_text = f"W:{minor:.1f}px, L:{major:.1f}px | Angle: {theta_deg:.1f}deg"
                        # Print parameters on image close to the droplet
                        cv2.putText(annotated_img, f"{theta_deg:.1f} deg", (int(x) - 40, int(y) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                        
                        fit_count += 1
                        if fit_count >= 3: # Limit annotations to keep visualization clean
                            break
                            
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(annotated_img_rgb)
        axes[idx].set_title(f"{cls_name.replace('_', ' ')} (Physics Fit)")
        axes[idx].axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(eval_dir, "ellipse_fitting_validation.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved ellipse fitting validation plot to: {output_path}")

def main():
    generate_gradcam_plot()
    generate_ellipse_fitting_plot()

if __name__ == "__main__":
    main()
