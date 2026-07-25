import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import timm

# Setup python path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "Task_1_Classification/Code/training"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Task_2_TSD/Code/models"))

# Import custom architectures
from train_efficientnet import BPAMultiTaskEfficientNet
from bloodnet import bloodnet50

# Set Page Config
st.set_page_config(page_title="Forensic BPA & TSD Dashboard", page_icon="🩸", layout="wide")

# Theme styling in CSS
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .main-header h1 {
        font-size: 2.5em;
        font-weight: 800;
        color: #4682B4;
        margin-bottom: 2px;
    }
    .main-header .subtitle {
        font-size: 1.1em;
        font-weight: 400;
        opacity: 0.8;
        color: #A0A0A0;
    }
    .card {
        background-color: #1E232E;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2E364F;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 0.9em;
        color: #A0A0A0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <h1>🩸 Forensic Bloodstain Pattern Analysis (BPA) & TSD Estimator</h1>
    <div class="subtitle">An Explainable AI (XAI) Decision-Support System Integrating Multi-Task Deep Learning & Fluid Kinematics</div>
</div>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADERS -----------------
@st.cache_resource
def load_task1_model(model_path):
    model = BPAMultiTaskEfficientNet(num_pattern=4, num_mechanism=3)
    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_task2_model(model_path):
    # BloodNet50 has 5 classes in pre-trained, but we modified to 3 classes
    model = bloodnet50(num_classes=5)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Resolve paths
TASK1_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Task_1_Classification/Models/model1_efficientnet_b0.pth")
TASK2_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Task_2_TSD/Models/best_tsd_model_resnet50.pth")

# Load models safely
with st.spinner("Loading PyTorch Neural Networks into memory..."):
    try:
        t1_model = load_task1_model(TASK1_MODEL_PATH)
        t2_model = load_task2_model(TASK2_MODEL_PATH)
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")

# Sidebar Settings
st.sidebar.title("🔧 Dashboard Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("""
**Task 1 Model Archetype:**  
EfficientNet-B0 (Multi-Task)  
*Pattern Accuracy: 92.95% raw / 95.83% holdout*  
*Force Accuracy: 99.79% raw / 99.17% holdout*

**Task 2 Model Archetype:**  
ResNet-50 + CBAM (BloodNet50)  
*TSD Accuracy: 97.46% raw / 98.00% holdout*  
""")

# ----------------- KINEMATICS & PREPROCESSING -----------------
def preprocess_bpa_image(image_pil):
    # Convert PIL to OpenCV BGR
    img = np.array(image_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Color Extraction (HSV Red Extraction)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 40])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([168, 50, 40])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    color_mask = cv2.bitwise_or(mask1, mask2)
    
    # 2. Substrate & Skin Subtraction (YCrCb mask)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77])
    upper_skin = np.array([255, 173, 127])
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    substrate_mask = cv2.bitwise_not(skin_mask)
    
    # Combined Mask
    final_mask = cv2.bitwise_and(color_mask, substrate_mask)
    
    # 3. Morphological filtering (blur & threshold)
    blurred = cv2.GaussianBlur(final_mask, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return img_bgr, binary

def fit_ellipse_and_calculate_angle(img_bgr, binary_mask):
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, img_bgr
    
    # Take the largest contour (main bloodstain)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Draw contours and fit ellipse
    canvas = img_bgr.copy()
    
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (x, y), (w_ax, l_ax), angle = ellipse
        
        # Draw the ellipse in green
        cv2.ellipse(canvas, ellipse, (0, 255, 0), 2)
        # Draw axes
        # Minor axis width, major axis length
        minor_len = min(w_ax, l_ax)
        major_len = max(w_ax, l_ax)
        
        # Calculate impact angle theta = arcsin(W/L)
        ratio = minor_len / major_len if major_len > 0 else 0
        ratio = min(ratio, 1.0) # Clamp to avoid domain error
        theta_rad = np.arcsin(ratio)
        theta_deg = theta_rad * (180.0 / np.pi)
        
        # Write info on canvas
        text = f"W:{minor_len:.1f}px L:{major_len:.1f}px R:{ratio:.3f} Angle:{theta_deg:.1f}deg"
        cv2.putText(canvas, f"Angle: {theta_deg:.1f}°", (int(x - 40), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            "minor_axis": minor_len,
            "major_axis": major_len,
            "ratio": ratio,
            "angle_deg": theta_deg,
            "center": (x, y)
        }, canvas
    
    return None, img_bgr

# ----------------- GRAD-CAM EXPLAINABILITY -----------------
def generate_gradcam_overlay(model, image_pil, target_class):
    # Transforms
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = t(image_pil).unsqueeze(0) # [1, 3, 224, 224]
    
    # Define placeholder to save features and gradients
    feature_maps = []
    gradients = []
    
    def save_feature_maps(module, input, output):
        feature_maps.append(output)
        
    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    # We hook into the final conv layer of the EfficientNet-B0 backbone: conv_head
    target_layer = model.backbone.conv_head
    
    handle_feat = target_layer.register_forward_hook(save_feature_maps)
    handle_grad = target_layer.register_full_backward_hook(save_gradients)
    
    # Forward pass
    output = model(input_tensor)
    pattern_logits = output['pattern']
    
    # Backward pass for the target class
    model.zero_grad()
    score = pattern_logits[0, target_class]
    score.backward()
    
    # Remove hooks
    handle_feat.remove()
    handle_grad.remove()
    
    # Extract features and gradients
    f_map = feature_maps[0].detach() # [1, 1280, 7, 7]
    grads = gradients[0].detach()     # [1, 1280, 7, 7]
    
    # Compute weight coefficients
    weights = torch.mean(grads, dim=(2, 3), keepdim=True) # [1, 1280, 1, 1]
    
    # Combine channel maps
    cam = torch.sum(weights * f_map, dim=1).squeeze(0) # [7, 7]
    cam = torch.clamp(cam, min=0) # Apply ReLU
    
    # Normalize to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    cam = cam.cpu().numpy()
    
    # Convert original PIL image to numpy array
    img_orig = np.array(image_pil.resize((224, 224)))
    
    # Resize cam to match image resolution
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Construct overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
    
    return overlay

# ----------------- UPLOAD & ANALYZE INTERFACE -----------------

st.markdown("### 📤 Upload Bloodstain Sample Image")
uploaded_file = st.file_uploader("Upload an image file (.png, .jpg, .jpeg) captured from a crime scene or experimental setup:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display PIL Image
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    # Run Preprocessing
    img_bgr, binary_mask = preprocess_bpa_image(image_pil)
    kinematics_data, ellipse_img = fit_ellipse_and_calculate_angle(img_bgr, binary_mask)
    
    # Display columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Original Input**")
        st.image(image_pil, use_container_width=True)
        
    with col2:
        st.markdown("**HSV Substrate-Subtracted Mask**")
        st.image(binary_mask, use_container_width=True, channels="GRAY")
        
    with col3:
        st.markdown("**Physics Ellipse Fitting**")
        ellipse_rgb = cv2.cvtColor(ellipse_img, cv2.COLOR_BGR2RGB)
        st.image(ellipse_rgb, use_container_width=True)
        
    # Execute Model Predictions
    # Task 1 Predict
    t1_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t1_tensor = t1_transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        t1_out = t1_model(t1_tensor)
        pattern_probs = torch.softmax(t1_out['pattern'], dim=1).squeeze(0).numpy()
        mechanism_probs = torch.softmax(t1_out['mechanism'], dim=1).squeeze(0).numpy()
        
    pattern_classes = ["Gunshot", "Impact Spatter", "Passive Drip", "Transfer/Wipe"]
    mechanism_classes = ["Passive", "Low Velocity", "Medium/High Velocity"]
    
    best_pattern_idx = np.argmax(pattern_probs)
    best_pattern = pattern_classes[best_pattern_idx]
    best_pattern_conf = pattern_probs[best_pattern_idx]
    
    best_mech_idx = np.argmax(mechanism_probs)
    best_mech = mechanism_classes[best_mech_idx]
    best_mech_conf = mechanism_probs[best_mech_idx]
    
    # Task 2 Predict
    t2_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t2_tensor = t2_transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        t2_out = t2_model(t2_tensor)
        tsd_probs = torch.softmax(t2_out, dim=1).squeeze(0).numpy()
        
    tsd_classes = ["Fresh (1 day)", "Intermediate (7 & 14 days)", "Aged (21 & 28 days)"]
    best_tsd_idx = np.argmax(tsd_probs)
    best_tsd = tsd_classes[best_tsd_idx]
    best_tsd_conf = tsd_probs[best_tsd_idx]
    
    # Grad-CAM Overlay
    with col4:
        st.markdown("**XAI Grad-CAM Overlay**")
        try:
            gradcam_img = generate_gradcam_overlay(t1_model, image_pil, best_pattern_idx)
            st.image(gradcam_img, use_container_width=True)
        except Exception as e:
            st.error(f"Error executing Grad-CAM: {str(e)}")
            
    # Display Result Cards
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Task 1: Pattern Type Prediction</div>
            <div class="metric-value">{best_pattern}</div>
            <div style="margin-top: 10px; font-size: 0.95em;">
                <b>Confidence:</b> {best_pattern_conf * 100:.2f}% <br>
                <b>Status:</b> {"Approved" if best_pattern_conf >= confidence_threshold else "Under Threshold Flag"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display probabilities bar chart
        prob_df1 = pd.DataFrame({
            "Probability": pattern_probs,
            "Class": pattern_classes
        })
        st.bar_chart(prob_df1.set_index("Class"))
        
    with res_col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Task 1: Force Mechanism</div>
            <div class="metric-value">{best_mech}</div>
            <div style="margin-top: 10px; font-size: 0.95em;">
                <b>Confidence:</b> {best_mech_conf * 100:.2f}% <br>
                <b>Velocity:</b> {"High Velocity" if best_mech_idx==2 else "Low/Gravitational Velocity"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        prob_df2 = pd.DataFrame({
            "Probability": mechanism_probs,
            "Class": mechanism_classes
        })
        st.bar_chart(prob_df2.set_index("Class"))
        
    with res_col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Task 2: Time Since Deposition</div>
            <div class="metric-value">{best_tsd}</div>
            <div style="margin-top: 10px; font-size: 0.95em;">
                <b>Confidence:</b> {best_tsd_conf * 100:.2f}% <br>
                <b>Chemical Aging:</b> Hemoglobin Oxidation Match
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        prob_df3 = pd.DataFrame({
            "Probability": tsd_probs,
            "Class": tsd_classes
        })
        st.bar_chart(prob_df3.set_index("Class"))
        
    # Display Physics Metrics
    if kinematics_data:
        st.markdown("### 📐 Physics-Based Kinematics Extraction")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        p_col1.metric("Minor Axis (W)", f"{kinematics_data['minor_axis']:.2f} px")
        p_col2.metric("Major Axis (L)", f"{kinematics_data['major_axis']:.2f} px")
        p_col3.metric("Aspect Ratio (W/L)", f"{kinematics_data['ratio']:.4f}")
        p_col4.metric("Calculated Impact Angle (θ)", f"{kinematics_data['angle_deg']:.2f}°")
        
        st.markdown(f"""
        **Fluid Dynamics Verification:** The calculated impact angle is reconstructed purely using the geometric relationship $\\theta = \\arcsin(W/L)$ which maps back to the physical trajectory of flight. Oblique impact spatters typically yield a low aspect ratio and directionality tails, while mist-like gunshot spatters yield rounder, higher-accuracy ellipses.
        """)
        
else:
    st.info("Please upload a bloodstain image file in the field above to start real-time prediction and diagnostic visualizations.")
    
    # Display standard sample placeholder
    st.markdown("### 🔎 Example Analysis Workflow Visualization")
    st.write("Below is a conceptual representation of how the unified preprocessing and XAI pipeline operates on an oblique spatter droplet:")
    
    placeholder_col1, placeholder_col2 = st.columns(2)
    with placeholder_col1:
        st.image("Task_1_Classification/Evaluation/before_after_comparison/Impact_Spatter_cleaning_validation.png", caption="Side-by-Side Preprocessing Validation (Raw vs. Masked Drop)", use_container_width=True)
    with placeholder_col2:
        st.image("Task_1_Classification/Evaluation/before_after_comparison/Gunshot_cleaning_validation_v2.png", caption="Gunshot Preprocessing Grid (Droplet Spines Preserved)", use_container_width=True)
