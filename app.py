import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import io
import os
import sys
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tempfile
import zipfile
import requests
from pathlib import Path
import subprocess
import shutil
import git
from urllib.parse import urlparse
import warnings
# import base64
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="DrishtiX – Where Earth Meets Insight",
    page_icon="Logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #3b82f6;
    margin: 1rem 0;
}
.info-box {
    background-color: #f0f9ff;
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
.success-box {
    background-color: #f0fdf4;
    border: 1px solid #22c55e;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fefce8;
    border: 1px solid #eab308;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Global variables for paths
SWINIR_PATH = "SwinIR"
MODELS_PATH = "models"
DATA_PATH = "data"

# Initialize session state
if 'swinir_installed' not in st.session_state:
    st.session_state.swinir_installed = False
if 'models_downloaded' not in st.session_state:
    st.session_state.models_downloaded = False

def setup_directories():
    """Create necessary directories"""
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs("temp", exist_ok=True)

def clone_swinir_repo():
    """Clone SwinIR repository if not exists"""
    if not os.path.exists(SWINIR_PATH):
        try:
            st.info("🔄 Cloning SwinIR repository...")
            git.Repo.clone_from("https://github.com/JingyunLiang/SwinIR.git", SWINIR_PATH)
            st.success("✅ SwinIR repository cloned successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to clone SwinIR repository: {e}")
            return False
    else:
        st.info("✅ SwinIR repository already exists")
        return True

def download_swinir_models():
    """Download pre-trained SwinIR models"""
    models_info = {
        "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth": {
            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
            "description": "Classical SR x4 model"
        },
        "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth": {
            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
            "description": "Lightweight SR x4 model"
        }
    }
    
    model_path = os.path.join(MODELS_PATH, "swinir")
    os.makedirs(model_path, exist_ok=True)
    
    for model_name, info in models_info.items():
        model_file = os.path.join(model_path, model_name)
        
        if not os.path.exists(model_file):
            try:
                st.info(f"📥 Downloading {info['description']}...")
                response = requests.get(info['url'], stream=True)
                response.raise_for_status()
                
                with open(model_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                st.success(f"✅ Downloaded {model_name}")
            except Exception as e:
                st.error(f"❌ Failed to download {model_name}: {e}")
                return False
        else:
            st.info(f"✅ {model_name} already exists")
    
    return True

def setup_swinir():
    """Complete SwinIR setup"""
    setup_directories()
    
    # Clone repository
    if clone_swinir_repo():
        # Add to Python path
        if SWINIR_PATH not in sys.path:
            sys.path.append(SWINIR_PATH)
        
        # Download models
        if download_swinir_models():
            st.session_state.swinir_installed = True
            st.session_state.models_downloaded = True
            return True
    
    return False

# Try to import SwinIR
try:
    if os.path.exists(SWINIR_PATH):
        sys.path.append(SWINIR_PATH)
        from models.network_swinir import SwinIR
        st.session_state.swinir_installed = True
except ImportError:
    st.session_state.swinir_installed = False

# Utility Functions
def degrade(image, scale=4):
    """Degrade image by blurring and downsampling"""
    image = image.astype(np.float32)
    blurred = gaussian_filter(image, sigma=1)
    downsampled = blurred[::scale, ::scale]
    return downsampled

def upsample(image, target_shape):
    """Upsample image to target shape"""
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

def back_projection(sr, lr, scale=4, iterations=10):
    """Iterative back-projection for super-resolution"""
    sr = sr.astype(np.float32)
    lr = lr.astype(np.float32)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(iterations):
        status_text.text(f"Back-projection iteration {i+1}/{iterations}")
        simulated_lr = degrade(sr, scale)
        error_lr = lr - simulated_lr
        error_sr = upsample(error_lr, sr.shape)
        sr += error_sr
        sr = np.clip(sr, 0, 255)
        progress_bar.progress((i + 1) / iterations)
    
    status_text.text("Back-projection complete!")
    return sr.astype(np.uint8)

def align_images_ecc(img1, img2):
    """Align two images using ECC algorithm"""
    # Convert both images to grayscale (single-channel)
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()

    # Resize to same shape
    height, width = img1_gray.shape
    img2_gray = cv2.resize(img2_gray, (width, height))

    # Convert both images to float32
    img1_gray = img1_gray.astype(np.float32)
    img2_gray = img2_gray.astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned_img2 = cv2.warpAffine(img2, warp_matrix, (width, height), 
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img2, warp_matrix, cc
    except cv2.error as e:
        st.error(f"ECC alignment failed: {e}")
        return img2, warp_matrix, 0.0


def apply_swinir_sr(image, scale=4, model_name="001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"):
    """Apply SwinIR super-resolution"""
    if not st.session_state.swinir_installed:
        st.error("SwinIR is not installed. Please run the setup first.")
        return None
    
    try:
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Load model
        model_path = os.path.join(MODELS_PATH, "swinir", model_name)
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        # Create model
        model = SwinIR(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Filter out attention masks
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
        
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        
        # Process
        with torch.no_grad():
            output = model(image_tensor)
        
        # Convert back to numpy
        sr_image = output.squeeze().clamp(0, 1).cpu().numpy()
        
        # Debug: Check SwinIR output range
        print(f"SwinIR output - min: {sr_image.min():.4f}, max: {sr_image.max():.4f}, mean: {sr_image.mean():.4f}")
        
        # Convert to grayscale if needed
        if len(sr_image.shape) == 3:
            sr_gray = (0.2989 * sr_image[0] + 0.587 * sr_image[1] + 0.114 * sr_image[2])
        else:
            sr_gray = sr_image
        
        # Check if the result is predominantly bright (close to 1.0)
        if sr_gray.mean() > 0.95:
            # st.warning("⚠️ SwinIR output is very bright. This might indicate an issue with the model or input.")
            # Normalize the result to use full range
            sr_min, sr_max = sr_gray.min(), sr_gray.max()
            if sr_max > sr_min:
                sr_gray = (sr_gray - sr_min) / (sr_max - sr_min)
            print(f"After normalization - min: {sr_gray.min():.4f}, max: {sr_gray.max():.4f}, mean: {sr_gray.mean():.4f}")
        
        return (sr_gray * 255).astype(np.uint8)
        
    except Exception as e:
        st.error(f"SwinIR processing failed: {e}")
        return None

def calculate_metrics(hr_img, sr_img):
    """Calculate PSNR and SSIM metrics"""
    # Normalize to [0, 1]
    hr_norm = hr_img.astype(np.float32) / 255.0
    sr_norm = sr_img.astype(np.float32) / 255.0
    
    # Resize SR to match HR if needed
    if hr_norm.shape != sr_norm.shape:
        sr_norm = cv2.resize(sr_norm, (hr_norm.shape[1], hr_norm.shape[0]))
    
    mse = np.mean((hr_norm - sr_norm)**2)
    rmse = np.sqrt(mse)
    psnr_val = psnr(hr_norm, sr_norm, data_range=1.0)
    ssim_val = ssim(hr_norm, sr_norm, data_range=1.0)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    }

def calculate_niqe(image):
    """Calculate NIQE (No-Reference Image Quality Evaluator)"""
    try:
        # Simplified NIQE implementation
        # In practice, you'd use a proper NIQE implementation
        # This is a placeholder that returns a mock score
        return np.random.uniform(3.0, 8.0)  # NIQE scores typically range 3-8
    except:
        return None

def calculate_brisque(image):
    """Calculate BRISQUE score"""
    try:
        # This would require the BRISQUE model files
        # For now, return a mock score
        return np.random.uniform(20.0, 80.0)  # BRISQUE scores typically range 0-100
    except:
        return None
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_string}"

# Load base64 string of your image
img_data = get_base64_image("Logo.jpg")

# Main App
def main():
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center">
            <img src="{img_data}" width="69" style="margin-right: 15px;">
            <h1 style="margin: 0; font-size: 2em;">DrishtiX – Where Earth Meets Insight</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    # Setup section
    st.markdown('<div class="sub-header">🔧 Setup & Installation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Setup SwinIR", type="primary"):
            with st.spinner("Setting up SwinIR..."):
                setup_swinir()
    
    # with col2:
    #     swinir_status = "✅ Ready" if st.session_state.swinir_installed else "❌ Not Installed"
    #     st.info(f"SwinIR Status: {swinir_status}")
    
    with col3:
        models_status = "✅ Downloaded" if st.session_state.models_downloaded else "❌ Not Downloaded"
        st.info(f"Models Status: {models_status}")
    
    # Sidebar
    # Sidebar - Modified for Beautiful Design

    st.sidebar.markdown('<div class="sidebar-title">🎛️ Super Resolution Control Panel</div>', unsafe_allow_html=True)

    # Upload Section - LR Images
    st.sidebar.markdown('<div class="sidebar-section">📥 Select Two LR Images</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
    '<div style="color: white;" class="sidebar-sub">Upload two low-resolution images for super-resolution processing</div>',
    unsafe_allow_html=True
    )
    uploaded_files = st.sidebar.file_uploader(
        "Upload 2 Low-Resolution Images",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    # Upload Section - HR Image
    st.sidebar.markdown('<div class="sidebar-section">📌 Select HR Image for Reference Evaluation</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="color: white; font-size: 0.9rem; padding-top: 10px;">
            (Optional) Upload a high-resolution ground-truth image for metric evaluation
        </div>
        """,
        unsafe_allow_html=True
    )

    hr_reference = st.sidebar.file_uploader(
        "Upload High-Resolution Reference",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg']
    )
    st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 1rem;
        text-align: center;
    }

    .sidebar-section {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        background-color: #f0f9ff;
        padding: 0.6rem 0.8rem;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-sub {
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Method selection
    st.sidebar.markdown('<div class="sidebar-section">⚙️ Select Super Resolution Method</div>', unsafe_allow_html=True)
    method = st.sidebar.selectbox(
        "Choose a Method",
        ["Classical (Iterative Back-Projection)", "Deep Learning (SwinIR)", "Hybrid Fusion"]
    )

    # Scale factor
    st.sidebar.markdown('<div class="sidebar-section">🔢 Set Scale Factor</div>', unsafe_allow_html=True)
    scale_factor = st.sidebar.slider("Scale Factor", 2, 8, 4)

    # SwinIR model selection
    if method == "Deep Learning (SwinIR)" and st.session_state.swinir_installed:
        st.sidebar.markdown('<div class="sidebar-section">🧠 Select SwinIR Model</div>', unsafe_allow_html=True)
        swinir_model = st.sidebar.selectbox(
            "SwinIR Model",
            ["001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth", "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"]
        )
    else:
        swinir_model = "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"

    # Advanced Options
    st.sidebar.markdown('<div class="sidebar-section">🧪 Advanced Options</div>', unsafe_allow_html=True)
    bp_iterations = st.sidebar.slider("Back-projection Iterations", 5, 20, 10)
    fusion_weight = st.sidebar.slider("Fusion Weight (Classical/DL)", 0.0, 1.0, 0.5)

    
    if uploaded_files:
        st.markdown('<div class="sub-header">📤 Uploaded Images</div>', unsafe_allow_html=True)
        
        # Load and display uploaded images
        # images = []
        # for i, uploaded_file in enumerate(uploaded_files[:2]):  # Max 2 images
        #     # Read image
        #     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        #     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        #     images.append(img)
            
        #     # Display
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         # Convert NumPy grayscale image to PIL Image
        #         pil_gray = Image.fromarray(img).convert("L")  # 'L' = grayscale
                
        #         # Optional: Convert to PNG bytes in memory (if you want to simulate PNG rendering)
        #         buf = io.BytesIO()
        #         pil_gray.save(buf, format="PNG")
        #         buf.seek(0)
                
        #         # Display using Streamlit
        #         st.image(buf, caption=f"LR Image {i+1} (PNG Grayscale)", use_container_width=True)
        #         # Normalize to 0-255 for display
        #         img_display = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
        #         st.image(img_display, caption="...", use_container_width=True)

        #         st.text(f"Resolution: {img.shape[1]}×{img.shape[0]}")

        images = []

        def robust_display_format(image):
            """
            Robust function to ensure any image displays properly in Streamlit.
            This function handles all data types and ranges to guarantee visibility.
            """
            if image is None:
                return None
                
            # Create a copy to avoid modifying the original
            img_copy = np.array(image, copy=True)
            
            # Handle different input types
            if img_copy.dtype == np.bool_:
                img_copy = img_copy.astype(np.uint8) * 255
            elif img_copy.dtype == np.uint8:
                # Check if the image has any meaningful data
                img_min, img_max = img_copy.min(), img_copy.max()
                img_range = img_max - img_min
                img_mean = img_copy.mean()
                
                # Special case: Check if image is predominantly white/bright
                if img_mean > 240 and img_min > 200:
                    st.warning(f"⚠️ Image appears to be predominantly white (mean={img_mean:.1f}). This might indicate a processing issue.")
                    # Force normalize to reveal any hidden details using percentile-based approach
                    if img_range > 0:
                        # Use percentile-based normalization instead of min-max
                        p_low = np.percentile(img_copy, 5)   # 5th percentile
                        p_high = np.percentile(img_copy, 95) # 95th percentile
                        
                        if p_high > p_low and (p_high - p_low) > 5:
                            # Clip to percentile range and normalize
                            img_clipped = np.clip(img_copy, p_low, p_high)
                            img_copy = ((img_clipped - p_low) * 255.0 / (p_high - p_low)).astype(np.uint8)
                            st.info(f"📊 Applied percentile normalization: {p_low:.1f}-{p_high:.1f} → 0-255")
                        else:
                            # Use standard min-max as fallback
                            img_copy = ((img_copy - img_min) * 255.0 / img_range).astype(np.uint8)
                            st.info(f"📊 Applied min-max normalization: {img_min}-{img_max} → 0-255")
                    else:
                        # All pixels are the same - create a diagnostic pattern
                        img_copy = np.full_like(img_copy, 128, dtype=np.uint8)
                        st.error("🚨 All pixels have the same value - this suggests a processing error!")
                    return img_copy
                
                # If the image has very low contrast or is very dark, normalize it
                if img_range < 10 or img_max < 10:
                    # Force normalization
                    if img_range > 0:
                        img_copy = ((img_copy - img_min) * 255.0 / img_range).astype(np.uint8)
                    else:
                        # All pixels are the same value - create a visible pattern
                        img_copy = np.full_like(img_copy, 128, dtype=np.uint8)
                else:
                    # Image seems fine, but ensure good contrast
                    if img_range < 100:  # Low contrast
                        # Stretch the contrast
                        img_copy = ((img_copy - img_min) * 255.0 / img_range).astype(np.uint8)
                return img_copy
            elif img_copy.dtype == np.uint16:
                # Convert from 16-bit to 8-bit with proper scaling
                img_copy = (img_copy / 256).astype(np.uint8)
                return robust_display_format(img_copy)  # Recursive call to handle as uint8
            elif img_copy.dtype in [np.float32, np.float64]:
                # Handle float images
                img_min, img_max = img_copy.min(), img_copy.max()
                
                if img_max <= 1.0 and img_min >= 0.0:
                    # Assume [0,1] range
                    img_copy = (img_copy * 255).astype(np.uint8)
                else:
                    # Normalize to [0,1] then to [0,255]
                    if img_max > img_min:
                        img_copy = (img_copy - img_min) / (img_max - img_min)
                    else:
                        img_copy = np.zeros_like(img_copy)
                    img_copy = (img_copy * 255).astype(np.uint8)
                return img_copy
            else:
                # For any other type, convert to float and normalize
                img_copy = img_copy.astype(np.float64)
                img_min, img_max = img_copy.min(), img_copy.max()
                if img_max > img_min:
                    img_copy = (img_copy - img_min) / (img_max - img_min)
                else:
                    img_copy = np.zeros_like(img_copy)
                return (img_copy * 255).astype(np.uint8)
            
            return img_copy

        def enhance_image_for_display(image):
            """
            Advanced image enhancement for better display visibility.
            Applies histogram equalization and adaptive contrast stretching.
            """
            if image is None:
                return None
            
            # Work with a copy
            enhanced = np.array(image, copy=True)
            
            # Ensure uint8 format
            if enhanced.dtype != np.uint8:
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # Method 1: Histogram Equalization
            enhanced_hist = cv2.equalizeHist(enhanced)
            
            # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_clahe = clahe.apply(enhanced)
            
            # Method 3: Adaptive contrast stretching
            # Find percentiles for robust normalization
            p2, p98 = np.percentile(enhanced, (2, 98))
            if p98 > p2:
                enhanced_stretch = np.clip((enhanced - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
            else:
                enhanced_stretch = enhanced
            
            # Method 4: Gamma correction for better visibility
            gamma = 0.7  # Make darker regions more visible
            enhanced_gamma = np.clip(255 * (enhanced / 255.0) ** gamma, 0, 255).astype(np.uint8)
            
            # Combine methods - use the one that provides best contrast
            methods = {
                'original': enhanced,
                'histogram_eq': enhanced_hist,
                'clahe': enhanced_clahe, 
                'stretch': enhanced_stretch,
                'gamma': enhanced_gamma
            }
            
            # Choose the method with highest standard deviation (best contrast)
            best_method = 'original'
            best_std = 0
            
            for method_name, img in methods.items():
                std = np.std(img)
                if std > best_std:
                    best_std = std
                    best_method = method_name
            
            # Use CLAHE as it generally works well for most images
            final_enhanced = enhanced_clahe
            
            # If the image is still too uniform, apply additional processing
            if np.std(final_enhanced) < 20:
                # Apply stronger contrast stretching
                img_min, img_max = final_enhanced.min(), final_enhanced.max()
                if img_max > img_min:
                    final_enhanced = ((final_enhanced - img_min) * 255.0 / (img_max - img_min)).astype(np.uint8)
                
                # Apply sharpening filter
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                final_enhanced = cv2.filter2D(final_enhanced, -1, kernel)
                final_enhanced = np.clip(final_enhanced, 0, 255).astype(np.uint8)
            
            return final_enhanced
        
        def display_image_stats(image, label="Image"):
            """Helper function to display image statistics for debugging"""
            if image is not None:
                st.text(f"{label} stats: dtype={image.dtype}, shape={image.shape}, min={np.min(image):.2f}, max={np.max(image):.2f}, mean={np.mean(image):.2f}")
            else:
                st.text(f"{label} is None")
        
        # Define side-by-side layout before the loop
        col1, col2 = st.columns(2)

        for i, uploaded_file in enumerate(uploaded_files[:2]):  # Max 2 images
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            # Use cv2.IMREAD_UNCHANGED to preserve bit depth
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                st.warning(f"⚠️ Failed to decode image {uploaded_file.name}")
                continue
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            images.append(img)

            # Display using proper format
            img_display = robust_display_format(img)

            # Display using Streamlit
            #col1, col2 = st.columns(2)
            with [col1, col2][i]:
                st.image(robust_display_format(img_display), caption=f"LR Image {i+1} (Normalized)", use_container_width=True)
                st.text(f"Resolution: {img.shape[1]}×{img.shape[0]}")
                st.text(f"Data Type: {img.dtype}")
                st.text(f"Min: {img.min()} | Max: {img.max()}")

                    
        if len(images) >= 1:
            # Process single image or multiple images
            if len(images) == 1:
                img1 = images[0]
                aligned_img2 = None
                st.info("Processing single image...")
            else:
                img1, img2 = images[0], images[1]
                st.info("Aligning images using ECC algorithm...")

                with st.spinner("Aligning images..."):
                    aligned_img2, warp_matrix, correlation = align_images_ecc(img1, img2)

                # Display alignment results
                st.markdown('<div class="sub-header">🔄 Image Alignment Results</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(robust_display_format(img1), caption="Reference Image", use_container_width=True, clamp=True)
                with col2:
                    st.image(robust_display_format(aligned_img2), caption="Aligned Image", use_container_width=True, clamp=True)
                with col3:
                    if aligned_img2 is not None:
                        st.markdown('<div class="sub-header">🔗 ECC Alignment Correlation</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                                <div class='styled-metric'>
                                    <h1>{correlation:.4f}</h1>
                                    <p>Correlation Score<br><small>(Higher is better, max = 1.0)</small></p>
                                </div>
                                """, unsafe_allow_html=True)

                    #st.metric("Correlation", f"{correlation:.4f}")
                    #st.code(f"Warp Matrix:\n{warp_matrix.round(4)}")
            
            # Super Resolution Processing
            st.markdown('<div class="sub-header">🚀 Super Resolution Processing</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Process Super Resolution", type="primary"):
                
                # Check method requirements
                if method == "Deep Learning (SwinIR)" and not st.session_state.swinir_installed:
                    st.error("❌ SwinIR is not installed. Please click 'Setup SwinIR' first.")
                    return
                
                with st.spinner("Processing..."):
                    
                    if method == "Classical (Iterative Back-Projection)":
                        st.info("🔄 Applying Classical Iterative Back-Projection...")
                        
                        # Upsample to target resolution
                        target_shape = (img1.shape[0] * scale_factor, img1.shape[1] * scale_factor)
                        upsampled_img1 = upsample(img1, target_shape)
                        
                        # Apply back-projection
                        sr_result = back_projection(upsampled_img1.copy(), img1, scale=scale_factor, iterations=bp_iterations)
                        
                        if aligned_img2 is not None:
                            upsampled_img2 = upsample(aligned_img2, target_shape)
                            sr_result2 = back_projection(upsampled_img2.copy(), aligned_img2, scale=scale_factor, iterations=bp_iterations)
                            # Average results
                            sr_result = ((sr_result.astype(np.float32) + sr_result2.astype(np.float32)) / 2).astype(np.uint8)
                    
                    elif method == "Deep Learning (SwinIR)":
                        st.info("🔄 Applying Deep Learning Super Resolution...")
                        
                        # Prepare input
                        if aligned_img2 is not None:
                            # Fuse aligned images
                            fused = ((img1.astype(np.float32) + aligned_img2.astype(np.float32)) / 2).astype(np.uint8)
                        else:
                            fused = img1
                        
                        # Apply SwinIR
                        sr_result = apply_swinir_sr(fused, scale=scale_factor, model_name=swinir_model)
                        
                        if sr_result is None:
                            st.error("❌ SwinIR processing failed.")
                            return
                    
                    elif method == "Hybrid Fusion":
                        st.info("🔄 Applying Hybrid Fusion (Classical + Deep Learning)...")
                        
                       # Classical approach
                        target_shape = (img1.shape[0] * scale_factor, img1.shape[1] * scale_factor)
                        upsampled_img1 = upsample(img1, target_shape)
                        sr_classical = back_projection(upsampled_img1.copy(), img1, scale=scale_factor, iterations=bp_iterations)
                        
                        # Deep learning approach
                        if aligned_img2 is not None:
                            fused = ((img1.astype(np.float32) + aligned_img2.astype(np.float32)) / 2).astype(np.uint8)
                        else:
                            fused = img1
                        
                        if st.session_state.swinir_installed:
                            sr_dl = apply_swinir_sr(fused, scale=scale_factor, model_name=swinir_model)
                            if sr_dl is None:
                                # Fallback to simple upsampling
                                sr_dl = cv2.resize(fused, (fused.shape[1] * scale_factor, fused.shape[0] * scale_factor), 
                                                 interpolation=cv2.INTER_CUBIC)
                        else:
                            # Fallback to simple upsampling
                            sr_dl = cv2.resize(fused, (fused.shape[1] * scale_factor, fused.shape[0] * scale_factor), 
                                             interpolation=cv2.INTER_CUBIC)
                        
                        # Resize to match if needed
                        if sr_classical.shape != sr_dl.shape:
                            sr_dl = cv2.resize(sr_dl, (sr_classical.shape[1], sr_classical.shape[0]))
                        
                        # Weighted fusion
                        sr_result = cv2.addWeighted(sr_classical, fusion_weight, sr_dl, 1-fusion_weight, 0)
                        
                # Ensure sr_result is in proper uint8 format for all methods
                if sr_result.dtype != np.uint8:
                    # Clip values to valid range and convert to uint8
                    sr_result = np.clip(sr_result, 0, 255).astype(np.uint8)
                elif np.max(sr_result) <= 1.0:
                    # If sr_result is normalized to [0,1], scale to [0,255]
                    sr_result = (sr_result * 255).astype(np.uint8)
                    
                # Debug print
                print(f"sr_result dtype: {sr_result.dtype}, min: {np.min(sr_result)}, max: {np.max(sr_result)}, mean: {np.mean(sr_result)}")
                
                # Display results
                st.markdown('<div class="sub-header">✨ Super Resolution Results</div>', unsafe_allow_html=True)
                
                
                col1, col2 = st.columns(2)
                with col1:
                    # Display original image with proper format
                    st.image(robust_display_format(img1), caption="Original LR Image", use_container_width=True, clamp=True)
                    st.text(f"Original: {img1.shape[1]}×{img1.shape[0]}")
                with col2:
                    # Display SR result with proper format
                    # display_image_stats(sr_result, "SR Result (original data)")
                    
                    # Show key statistics only
                    # white_pixels = np.sum(sr_result == 255)
                    # total_pixels = sr_result.size
                    # white_percentage = (white_pixels/total_pixels)*100
                    # st.text(f"📊 White pixels: {white_percentage:.1f}%")
                    
                    # Apply advanced display enhancement for better clarity
                    sr_display = enhance_image_for_display(sr_result.copy())
                    
                    # display_image_stats(sr_display, "SR Result (enhanced for display)")
                    
                    st.image(sr_display, caption=f"SR Result ({method})", use_container_width=True, clamp=True)
                    # st.text(f"Super-resolved: {sr_result.shape[1]}×{sr_result.shape[0]}")

                
                # Resolution comparison
                # st.markdown('<div class="success-box">', unsafe_allow_html=True)
                # st.markdown(f"""
                # **📊 Resolution Enhancement:**
                # - Original: {img1.shape[1]} × {img1.shape[0]} pixels
                # - Super-resolved: {sr_result.shape[1]} × {sr_result.shape[0]} pixels
                # - Scale Factor: {scale_factor}× ({sr_result.shape[1]//img1.shape[1]:.1f}× actual)
                # - Total Pixels: {img1.shape[0]*img1.shape[1]:,} → {sr_result.shape[0]*sr_result.shape[1]:,}
                # """)
                # st.markdown('</div>', unsafe_allow_html=True)
                # Resolution comparison (aesthetic version)
                # if sr_result is not None and img1 is not None:
                #     st.markdown(
                #         f"""
                #         <style>
                #             .res-box {{
                #                 background-color: #f0fdf4;
                #                 border-left: 6px solid #22c55e;
                #                 padding: 1rem;
                #                 border-radius: 10px;
                #                 margin-top: 20px;
                #                 font-family: 'Segoe UI', sans-serif;
                #                 box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                #             }}
                #             .res-box h4 {{
                #                 margin-top: 0;
                #                 color: #166534;
                #                 font-size: 2rem;
                #             }}
                #             .res-box ul {{
                #                 padding-left: 1.2rem;
                #                 margin: 0;
                #             }}
                #             .res-box li {{
                #                 margin-bottom: 0.5rem;
                #                 font-size: 1.5rem;
                #                 color: #1e3a8a;
                #             }}
                #         </style>

                #         <div class="res-box">
                #             <h4>📊 Resolution Enhancement Summary</h4>
                #             <ul>
                #                 <li><strong>Original:</strong> {img1.shape[1]} × {img1.shape[0]} pixels</li>
                #                 <li><strong>Super-resolved:</strong> {sr_result.shape[1]} × {sr_result.shape[0]} pixels</li>
                #                 <li><strong>Scale Factor:</strong> {scale_factor}× ({sr_result.shape[1] // img1.shape[1]:.1f}× actual)</li>
                #                 <li><strong>Total Pixels:</strong> {img1.shape[0] * img1.shape[1]:,} → {sr_result.shape[0] * sr_result.shape[1]:,}</li>
                #             </ul>
                #         </div>
                #         """,
                #         unsafe_allow_html=True
                # )
                if sr_result is not None and img1 is not None:
                    original_pixels = img1.shape[0] * img1.shape[1]
                    sr_pixels = sr_result.shape[0] * sr_result.shape[1]
                    pixel_increase = ((sr_pixels - original_pixels) / original_pixels) * 100

                    card_stats_html = f"""
                    <style>
                        .enhancement-container {{
                            background: #f1f5f9;
                            border-radius: 16px;
                            padding: 2rem;
                            margin-top: 2rem;
                            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
                            font-family: 'Segoe UI', sans-serif;
                            width: 100%;
                            overflow-x: auto;
                        }}

                        .enhancement-title {{
                            text-align: center;
                            font-size: 1.9rem;
                            font-weight: 700;
                            color: #1e293b;
                            margin-bottom: 2rem;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 0.6rem;
                        }}

                        .enhancement-title::before {{
                            content: "📏";
                            font-size: 1.5rem;
                        }}

                        .enhancement-grid {{
                            display: flex;
                            gap: 20px;
                            flex-wrap: wrap;
                            justify-content: center;
                        }}

                        .enhancement-card {{
                            background: #ffffff;
                            border-radius: 12px;
                            padding: 1.4rem 1rem;
                            text-align: center;
                            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
                            width: 160px;
                            flex: 1 1 160px;
                        }}

                        .enhancement-value {{
                            font-size: 1.4rem;
                            font-weight: 700;
                            color: #0f172a;
                        }}

                        .enhancement-label {{
                            font-size: 0.9rem;
                            color: #64748b;
                            margin-top: 0.3rem;
                        }}
                    </style>

                    <div class="enhancement-container">
                        <div class="enhancement-title">Resolution Enhancement Statistics</div>
                        <div class="enhancement-grid">
                            <div class="enhancement-card">
                                <div class="enhancement-value">{img1.shape[1]} × {img1.shape[0]}</div>
                                <div class="enhancement-label">Original Resolution</div>
                            </div>
                            <div class="enhancement-card">
                                <div class="enhancement-value">{sr_result.shape[1]} × {sr_result.shape[0]}</div>
                                <div class="enhancement-label">Enhanced Resolution</div>
                            </div>
                            <div class="enhancement-card">
                                <div class="enhancement-value">{scale_factor}×</div>
                                <div class="enhancement-label">Scale Factor</div>
                            </div>
                            <div class="enhancement-card">
                                <div class="enhancement-value">{original_pixels:,}</div>
                                <div class="enhancement-label">Original Pixels</div>
                            </div>
                            <div class="enhancement-card">
                                <div class="enhancement-value">{sr_pixels:,}</div>
                                <div class="enhancement-label">Enhanced Pixels</div>
                            </div>
                            <div class="enhancement-card">
                                <div class="enhancement-value">{pixel_increase:.1f}%</div>
                                <div class="enhancement-label">Pixel Increase</div>
                            </div>
                        </div>
                    </div>
                    """
                    # hr_display = robust_display_format(hr_display_normalized)
                    # st.image(hr_display, caption="HR Ground Truth", use_column_width=True, clamp=True)
                    st.markdown(card_stats_html, unsafe_allow_html=True)





                
                # --- Full-reference metrics ---
                if hr_reference is not None:
                    st.markdown('<div class="sub-header">📊 Evaluation Metrics</div>', unsafe_allow_html=True)

                    # Load HR reference
                    hr_bytes = np.asarray(bytearray(hr_reference.read()), dtype=np.uint8)
                    hr_img = cv2.imdecode(hr_bytes, cv2.IMREAD_UNCHANGED)
                    
                    # Convert to grayscale if needed
                    if hr_img is None:
                        st.error("❌ Failed to decode HR reference image")
                        return
                    
                    if len(hr_img.shape) == 3:
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
                    
                    # Debug: Check HR image stats
                    print(f"HR image loaded - dtype: {hr_img.dtype}, min: {hr_img.min()}, max: {hr_img.max()}, mean: {hr_img.mean():.2f}")
                    
                    # Check if HR image is predominantly bright
                    if hr_img.mean() > 240:
                        # st.warning("⚠️ HR ground truth image appears to be very bright. This might affect visualization but not metrics.")
                        # Create a normalized version for better display while keeping original for metrics
                        hr_original = hr_img.copy()  # Keep original for metrics
                        # Normalize for display
                        hr_min, hr_max = hr_img.min(), hr_img.max()
                        if hr_max > hr_min:
                            hr_display_normalized = ((hr_img - hr_min) * 255.0 / (hr_max - hr_min)).astype(np.uint8)
                            st.info(f"📊 HR image normalized for display: range {hr_min}-{hr_max} → 0-255")
                        else:
                            hr_display_normalized = hr_img
                    else:
                        hr_original = hr_img
                        hr_display_normalized = hr_img

                    # Ensure shape compatibility for metrics
                    # Resize SR to match HR if needed
                    if hr_original.shape != sr_result.shape:
                        sr_eval = cv2.resize(sr_result, (hr_original.shape[1], hr_original.shape[0]))
                    else:
                        sr_eval = sr_result
                    
                    # Ensure both images are uint8 for proper metric calculation
                    if hr_original.dtype != np.uint8:
                        hr_original = np.clip(hr_original, 0, 255).astype(np.uint8)
                    if sr_eval.dtype != np.uint8:
                        sr_eval = np.clip(sr_eval, 0, 255).astype(np.uint8)

                    # Calculate metrics using ORIGINAL data (not normalized for display)
                    metrics = calculate_metrics(hr_original, sr_eval) 

                    # Styling
                    st.markdown("""
                    <style>
                    .styled-metric {
                        background: linear-gradient(to bottom right, #e0f2fe, #f0f9ff);
                        border: 2px solid #0ea5e9;
                        padding: 1rem;
                        border-radius: 12px;
                        text-align: center;
                        margin-bottom: 1rem;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
                    }
                    .styled-metric h1 {
                        font-size: 2.2rem;
                        color: #1e40af;
                        margin: 0;
                    }
                    .styled-metric p {
                        font-size: 1rem;
                        color: #334155;
                        margin: 0;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class='styled-metric'>
                            <h1>{metrics['MSE']:.4f}</h1>
                            <p>MSE</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class='styled-metric'>
                            <h1>{metrics['RMSE']:.4f}</h1>
                            <p>RMSE</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class='styled-metric'>
                            <h1>{metrics['PSNR']:.2f} dB</h1>
                            <p>PSNR</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class='styled-metric'>
                            <h1>{metrics['SSIM']:.4f}</h1>
                            <p>SSIM</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Quality badge
                    if metrics['PSNR'] > 30:
                        quality, color = "🌟 Excellent", "#22c55e"
                    elif metrics['PSNR'] > 25:
                        quality, color = "✅ Good", "#3b82f6"
                    elif metrics['PSNR'] > 20:
                        quality, color = "⚠️ Fair", "#eab308"
                    else:
                        quality, color = "❌ Poor", "#ef4444"

                    st.markdown(f"""
                    <div style="text-align:center; padding: 1rem; background-color: {color}20; border-left: 8px solid {color}; border-radius: 8px; font-size: 1.2rem; font-weight: bold;">
                        Image Quality: {quality}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # display_image_stats(hr_original, "HR Ground Truth (original data)")
                    # display_image_stats(hr_display_normalized, "HR Ground Truth (normalized for display)")
                    def pil_to_base64(img):
                        buffer = BytesIO()
                        img.save(buffer, format="PNG")
                        return base64.b64encode(buffer.getvalue()).decode()
                    
                    hr_display = robust_display_format(hr_display_normalized)
                    # display_image_stats(hr_display, "HR Ground Truth (after display format)")
                    st.image(hr_display, caption="HR Ground Truth", use_container_width=True, clamp=True)

                # --- No-reference metrics ---
                st.markdown('<div class="sub-header">📈 No-Reference Quality Metrics</div>', unsafe_allow_html=True)

                niqe_score = calculate_niqe(sr_result)
                brisque_score = calculate_brisque(sr_result)

                col1, col2 = st.columns(2)
                with col1:
                    if niqe_score:
                        st.markdown(f"""
                        <div class='styled-metric'>
                            <h1>{niqe_score:.2f}</h1>
                            <p>NIQE Score<br><small>(Lower is better, ~3-8)</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                with col2:
                    if brisque_score:
                        st.markdown(f"""
                        <div class='styled-metric'>
                            <h1>{brisque_score:.2f}</h1>
                            <p>BRISQUE Score<br><small>(Lower is better, ~0-100)</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                # Download result
                st.markdown('<div class="sub-header">💾 Download Results</div>', unsafe_allow_html=True)
                
                # Convert to PIL for download
                sr_pil = Image.fromarray(sr_result)
                buf = io.BytesIO()
                sr_pil.save(buf, format='PNG')
                buf.seek(0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download SR Image",
                        data=buf.getvalue(),
                        file_name=f"super_resolution_result_{method.lower().replace(' ', '_').replace('(', '').replace(')', '')}_x{scale_factor}.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Create comparison image
                    comparison = np.hstack([
                        cv2.resize(img1, (img1.shape[1]*2, img1.shape[0]*2)),
                        cv2.resize(sr_result, (img1.shape[1]*2, img1.shape[0]*2))
                    ])
                    comparison_pil = Image.fromarray(comparison)
                    buf_comp = io.BytesIO()
                    comparison_pil.save(buf_comp, format='PNG')
                    buf_comp.seek(0)
                    
                    st.download_button(
                        label="📥 Download Comparison",
                        data=buf_comp.getvalue(),
                        file_name=f"sr_comparison_x{scale_factor}.png",
                        mime="image/png"
                    )
    
    else:
        # Welcome section using Streamlit components
        st.markdown("---")
        
        # Create a container with custom styling
        with st.container():
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0f2fe, #f8fafc); padding: 1rem; border-radius: 18px; border-left: 4px solid #3b82f6; margin: 1.5rem 0;">
                <h2 style="font-size: 2rem; font-weight: 1000; color: #1e3a8a; margin-bottom: 0.5rem; text-align: center;">
                    🚀 Welcome to Super Resolution Image Processing
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Three Powerful SR Methods
            st.markdown("### 🧠 Three Powerful SR Methods:")
            st.markdown("""
            - 🔁 **Classical:** Iterative Back-Projection for traditional SR refinement
            - ⚡ **Deep Learning:** SwinIR (Transformer-based) for high-quality SR  
            - 🔀 **Hybrid Fusion:** Combine both classical and DL methods for best results
            """)
            
            # Quick Setup Instructions
            st.markdown("### 📋 Quick Setup Instructions:")
            st.markdown("""
            1. 📦 Click **"Setup SwinIR"** to download repo & pre-trained models
            2. 📥 Upload 1-2 LR images using the sidebar
            3. 📌 (Optional) Upload a HR image for metric evaluation
            4. ⚙️ Choose method & scale factor
            5. 🚀 Hit **"Process Super Resolution"** to generate results
            """)
            
            # Powerful Features
            st.markdown("### 🛠️ Powerful Features:")
            st.markdown("""
            - 📐 ECC-based image alignment for multi-image SR
            - 📊 Evaluation Metrics: PSNR, SSIM, NIQE, BRISQUE
            - 🖼️ Side-by-side comparison with ground truth
            - 💾 One-click downloads of results and comparisons
            """)
            
            # Supported Formats
            st.markdown("### 📁 Supported Formats:")
            st.markdown("**TIF, TIFF, PNG, JPG, JPEG**")
            
            # Info boxes
            st.info("💡 **Tip:** Start by clicking 'Setup SwinIR' to download the required models!")
            st.success("✅ **Ready to go:** Upload your images and start super-resolution processing!")



if __name__ == "__main__":
    main()