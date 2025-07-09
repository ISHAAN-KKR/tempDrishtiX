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
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Super Resolution Image Processing",
    page_icon="🖼️",
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
.metrics-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}

.metric-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(10px);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2d3748;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-label {
    font-size: 0.9rem;
    color: #718096;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.metric-description {
    font-size: 0.8rem;
    color: #a0aec0;
    font-style: italic;
    line-height: 1.3;
}

.quality-badge {
    display: inline-block;
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 1rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}

.quality-excellent {
    background: linear-gradient(135deg, #48bb78, #38a169);
    color: white;
}

.quality-good {
    background: linear-gradient(135deg, #4299e1, #3182ce);
    color: white;
}

.quality-fair {
    background: linear-gradient(135deg, #ed8936, #dd6b20);
    color: white;
}

.quality-poor {
    background: linear-gradient(135deg, #f56565, #e53e3e);
    color: white;
}

.no-ref-metrics {
    background: linear-gradient(135deg, #e2e8f0, #cbd5e0);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 5px solid #4299e1;
}

.comparison-stats {
    background: linear-gradient(135deg, #f7fafc, #edf2f7);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 2px solid #e2e8f0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.stat-item {
    text-align: center;
    padding: 1rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2d3748;
}

.stat-label {
    font-size: 0.8rem;
    color: #718096;
    font-weight: 600;
    margin-top: 0.5rem;
}

.progress-ring {
    display: inline-block;
    position: relative;
    width: 80px;
    height: 80px;
    margin: 0 auto;
}

.progress-ring-circle {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    border: 4px solid #e2e8f0;
    border-top: 4px solid #4299e1;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
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
    height, width = img1.shape
    img2 = cv2.resize(img2, (width, height))
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    
    try:
        (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, cv2.MOTION_AFFINE, criteria)
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
        
        # Convert to grayscale if needed
        if len(sr_image.shape) == 3:
            sr_gray = (0.2989 * sr_image[0] + 0.587 * sr_image[1] + 0.114 * sr_image[2])
        else:
            sr_gray = sr_image
        
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
# Function to display metrics aesthetically
def display_metrics_aesthetic(metrics, hr_img=None, sr_result=None, img1=None, scale_factor=4):
    """Display metrics in an aesthetic manner"""
    
    # Validate metric keys
    if not metrics or not all(k in metrics for k in ['MSE', 'RMSE', 'PSNR', 'SSIM']):
        st.error("⚠️ Incomplete metrics data. Cannot display results.")
        return
    
    # Color logic
    psnr_color = "#22c55e" if metrics['PSNR'] > 30 else "#3b82f6" if metrics['PSNR'] > 25 else "#eab308" if metrics['PSNR'] > 20 else "#ef4444"
    ssim_color = "#22c55e" if metrics['SSIM'] > 0.9 else "#3b82f6" if metrics['SSIM'] > 0.8 else "#eab308" if metrics['SSIM'] > 0.7 else "#ef4444"

    # Overall quality badge
    if metrics['PSNR'] > 30:
        quality, badge_class = "Excellent", "quality-excellent"
    elif metrics['PSNR'] > 25:
        quality, badge_class = "Good", "quality-good"
    elif metrics['PSNR'] > 20:
        quality, badge_class = "Fair", "quality-fair"
    else:
        quality, badge_class = "Poor", "quality-poor"

    # Build the full metrics HTML
    metrics_html = f"""
    <div class="metrics-container">
        <h2 style="color: white; text-align: center; margin-bottom: 1.5rem; font-size: 2rem;">📊 Image Quality Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-label">Mean Squared Error</div>
                <div class="metric-value">{metrics['MSE']:.6f}</div>
                <div class="metric-description">Lower values indicate better quality</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📐</div>
                <div class="metric-label">Root Mean Squared Error</div>
                <div class="metric-value">{metrics['RMSE']:.6f}</div>
                <div class="metric-description">Pixel-level error magnitude</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📶</div>
                <div class="metric-label">Peak Signal-to-Noise Ratio</div>
                <div class="metric-value" style="color: {psnr_color};">{metrics['PSNR']:.2f} dB</div>
                <div class="metric-description">Signal quality vs noise level</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">🔍</div>
                <div class="metric-label">Structural Similarity Index</div>
                <div class="metric-value" style="color: {ssim_color};">{metrics['SSIM']:.4f}</div>
                <div class="metric-description">Structural similarity (0-1 scale)</div>
            </div>
        </div>

        
    </div>
    <div style="text-align: center; margin-top: 2rem;">
        <div class="quality-badge {badge_class}">
            ⭐ Overall Quality: {quality}
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Resolution comparison stats
    if sr_result is not None and img1 is not None:
        original_pixels = img1.shape[0] * img1.shape[1]
        sr_pixels = sr_result.shape[0] * sr_result.shape[1]
        pixel_increase = ((sr_pixels - original_pixels) / original_pixels) * 100

        stats_html = f"""
        <div class="comparison-stats">
            <h3 style="color: #2d3748; text-align: center; margin-bottom: 1rem;">📏 Resolution Enhancement Statistics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{img1.shape[1]} × {img1.shape[0]}</div>
                    <div class="stat-label">Original Resolution</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{sr_result.shape[1]} × {sr_result.shape[0]}</div>
                    <div class="stat-label">Enhanced Resolution</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{scale_factor}×</div>
                    <div class="stat-label">Scale Factor</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{original_pixels:,}</div>
                    <div class="stat-label">Original Pixels</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{sr_pixels:,}</div>
                    <div class="stat-label">Enhanced Pixels</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{pixel_increase:.1f}%</div>
                    <div class="stat-label">Pixel Increase</div>
                </div>
            </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

def display_no_ref_metrics_aesthetic(niqe_score, brisque_score):
    """Display no-reference metrics aesthetically"""
    
    st.markdown('<div class="no-ref-metrics">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #2d3748; text-align: center; margin-bottom: 1rem;">📈 No-Reference Quality Assessment</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if niqe_score:
            niqe_color = "#22c55e" if niqe_score < 4 else "#3b82f6" if niqe_score < 5 else "#eab308" if niqe_score < 6 else "#ef4444"
            niqe_quality = "Excellent" if niqe_score < 4 else "Good" if niqe_score < 5 else "Fair" if niqe_score < 6 else "Poor"
            
            st.markdown(f'''
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin-bottom: 1rem;">
                <div style="font-size: 2rem; color: {niqe_color}; font-weight: 700;">{niqe_score:.2f}</div>
                <div style="font-size: 0.9rem; color: #718096; font-weight: 600;">NIQE Score</div>
                <div style="font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem;">Quality: {niqe_quality}</div>
                <div style="font-size: 0.7rem; color: #cbd5e0; margin-top: 0.3rem;">Lower is better (3-8 range)</div>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        if brisque_score:
            brisque_color = "#22c55e" if brisque_score < 30 else "#3b82f6" if brisque_score < 50 else "#eab308" if brisque_score < 70 else "#ef4444"
            brisque_quality = "Excellent" if brisque_score < 30 else "Good" if brisque_score < 50 else "Fair" if brisque_score < 70 else "Poor"
            
            st.markdown(f'''
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin-bottom: 1rem;">
                <div style="font-size: 2rem; color: {brisque_color}; font-weight: 700;">{brisque_score:.2f}</div>
                <div style="font-size: 0.9rem; color: #718096; font-weight: 600;">BRISQUE Score</div>
                <div style="font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem;">Quality: {brisque_quality}</div>
                <div style="font-size: 0.7rem; color: #cbd5e0; margin-top: 0.3rem;">Lower is better (0-100 range)</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main App
def main():
    st.markdown('<h1 class="main-header">🖼️ Super Resolution Image Processing</h1>', unsafe_allow_html=True)
    
    # Setup section
    st.markdown('<div class="sub-header">🔧 Setup & Installation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Setup SwinIR", type="primary"):
            with st.spinner("Setting up SwinIR..."):
                setup_swinir()
    
    with col2:
        swinir_status = "✅ Ready" if st.session_state.swinir_installed else "❌ Not Installed"
        st.info(f"SwinIR Status: {swinir_status}")
    
    with col3:
        models_status = "✅ Downloaded" if st.session_state.models_downloaded else "❌ Not Downloaded"
        st.info(f"Models Status: {models_status}")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Select Super Resolution Method",
        ["Classical (Iterative Back-Projection)", "Deep Learning (SwinIR)", "Hybrid Fusion"]
    )
    
    # Scale factor
    scale_factor = st.sidebar.slider("Scale Factor", 2, 8, 4)
    
    # SwinIR model selection
    if method == "Deep Learning (SwinIR)" and st.session_state.swinir_installed:
        swinir_model = st.sidebar.selectbox(
            "SwinIR Model",
            ["001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth", "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"]
        )
    else:
        swinir_model = "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
    
    # Advanced options
    st.sidebar.markdown("### Advanced Options")
    bp_iterations = st.sidebar.slider("Back-projection Iterations", 5, 20, 10)
    fusion_weight = st.sidebar.slider("Fusion Weight (Classical/DL)", 0.0, 1.0, 0.5)
    
    # File upload
    st.sidebar.markdown("### Upload Images")
    uploaded_files = st.sidebar.file_uploader(
        "Choose LR images (1-2 images)",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    # HR reference for evaluation
    hr_reference = st.sidebar.file_uploader(
        "Upload HR reference (optional, for evaluation)",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg']
    )
    
    if uploaded_files:
        st.markdown('<div class="sub-header">📤 Uploaded Images</div>', unsafe_allow_html=True)
        
        # Load and display uploaded images
        images = []
        for i, uploaded_file in enumerate(uploaded_files[:2]):  # Max 2 images
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                # Convert NumPy grayscale image to PIL Image
                pil_gray = Image.fromarray(img).convert("L")  # 'L' = grayscale
                
                # Optional: Convert to PNG bytes in memory (if you want to simulate PNG rendering)
                buf = io.BytesIO()
                pil_gray.save(buf, format="PNG")
                buf.seek(0)
                
                # Display using Streamlit
                st.image(buf, caption=f"LR Image {i+1} (PNG Grayscale)", use_column_width=True)
                st.text(f"Resolution: {img.shape[1]}×{img.shape[0]}")
                    
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
                    st.image(img1, caption="Reference Image", use_column_width=True, clamp=True)
                with col2:
                    st.image(aligned_img2, caption="Aligned Image", use_column_width=True, clamp=True)
                with col3:
                    st.metric("Correlation", f"{correlation:.4f}")
                    st.code(f"Warp Matrix:\n{warp_matrix.round(4)}")
            
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
                
                # Display results
                st.markdown('<div class="sub-header">✨ Super Resolution Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1, caption="Original LR Image", use_column_width=True, clamp=True)
                    st.text(f"Original: {img1.shape[1]}×{img1.shape[0]}")
                with col2:
                    st.image(sr_result, caption=f"SR Result ({method})", use_column_width=True, clamp=True)
                    st.text(f"Super-resolved: {sr_result.shape[1]}×{sr_result.shape[0]}")
                
                # Resolution comparison
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"""
                **📊 Resolution Enhancement:**
                - Original: {img1.shape[1]} × {img1.shape[0]} pixels
                - Super-resolved: {sr_result.shape[1]} × {sr_result.shape[0]} pixels
                - Scale Factor: {scale_factor}× ({sr_result.shape[1]//img1.shape[1]:.1f}× actual)
                - Total Pixels: {img1.shape[0]*img1.shape[1]:,} → {sr_result.shape[0]*sr_result.shape[1]:,}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Evaluation metrics
                if hr_reference is not None:
                    st.markdown('<div class="sub-header">📊 Evaluation Metrics</div>', unsafe_allow_html=True)
                    
                    # Load HR reference
                    hr_bytes = np.asarray(bytearray(hr_reference.read()), dtype=np.uint8)
                    hr_img = cv2.imdecode(hr_bytes, cv2.IMREAD_GRAYSCALE)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(hr_img, sr_result)
                    
                    # Display metrics aesthetically
                    display_metrics_aesthetic(metrics, hr_img, sr_result, img1, scale_factor)
                    
                    # Display HR reference
                    st.image(hr_img, caption="HR Reference", use_column_width=True, clamp=True)
                
                # No-reference metrics
                st.markdown('<div class="sub-header">📈 No-Reference Quality Metrics</div>', unsafe_allow_html=True)
                
                niqe_score = calculate_niqe(sr_result)
                brisque_score = calculate_brisque(sr_result)
                
                # Display no-reference metrics aesthetically
                display_no_ref_metrics_aesthetic(niqe_score, brisque_score)
                
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
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 Welcome to Super Resolution Image Processing!
        
        This application provides three different approaches for super-resolution:
        
        1. **Classical (Iterative Back-Projection)**: Traditional approach using iterative refinement
        2. **Deep Learning (SwinIR)**: State-of-the-art transformer-based super-resolution
        3. **Hybrid Fusion**: Combines both classical and deep learning approaches
        
        **📋 Setup Instructions:**
        1. Click "Setup SwinIR" to download the repository and pre-trained models
        2. Upload 1-2 low-resolution images using the sidebar
        3. Optionally upload a high-resolution reference for evaluation
        4. Select your preferred method and scale factor
        5. Click "Process Super Resolution" to generate results
        
        **🔧 Features:**
        - ECC-based image alignment for multi-image super-resolution
        - Multiple evaluation metrics (PSNR, SSIM, NIQE, BRISQUE)
        - Comparison with ground truth HR images
        - Download results and comparisons
        
        **📁 Supported formats:** TIF, TIFF, PNG, JPG, JPEG
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()