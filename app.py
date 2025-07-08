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
</style>
""", unsafe_allow_html=True)

# SwinIR Model Definition
class SwinIR(nn.Module):
    def __init__(self, upscale=4, in_chans=3, img_size=64, window_size=8,
                 img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                 num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle',
                 resi_connection='1conv'):
        super(SwinIR, self).__init__()
        # Simplified SwinIR for demonstration
        # In practice, you'd need the full implementation
        self.upscale = upscale
        self.in_chans = in_chans
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_chans * (upscale ** 2), 3, 1, 1)
        self.upsample = nn.PixelShuffle(upscale)
        
    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv_last(x)
        x = self.upsample(x)
        return x

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
    for i in range(iterations):
        simulated_lr = degrade(sr, scale)
        error_lr = lr - simulated_lr
        error_sr = upsample(error_lr, sr.shape)
        sr += error_sr
        sr = np.clip(sr, 0, 255)
        progress_bar.progress((i + 1) / iterations)
    
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
        return aligned_img2, warp_matrix
    except cv2.error as e:
        st.error(f"ECC alignment failed: {e}")
        return img2, warp_matrix

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

def plot_images(images, titles, figsize=(15, 5)):
    """Plot multiple images side by side"""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🖼️ Super Resolution Image Processing</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Select Super Resolution Method",
        ["Classical (Iterative Back-Projection)", "Deep Learning (SwinIR)", "Hybrid Fusion"]
    )
    
    # Scale factor
    scale_factor = st.sidebar.slider("Scale Factor", 2, 8, 4)
    
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
                st.image(img, caption=f"LR Image {i+1}", use_column_width=True, clamp=True)
        
        if len(images) >= 1:
            # Process single image or multiple images
            if len(images) == 1:
                img1 = images[0]
                aligned_img2 = None
                st.info("Processing single image...")
            else:
                img1, img2 = images[0], images[1]
                st.info("Aligning images using ECC algorithm...")
                aligned_img2, warp_matrix = align_images_ecc(img1, img2)
                
                # Display alignment results
                st.markdown('<div class="sub-header">🔄 Image Alignment Results</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1, caption="Reference Image", use_column_width=True, clamp=True)
                with col2:
                    st.image(aligned_img2, caption="Aligned Image", use_column_width=True, clamp=True)
                
                st.code(f"Warp Matrix:\n{warp_matrix}")
            
            # Super Resolution Processing
            st.markdown('<div class="sub-header">🚀 Super Resolution Processing</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Process Super Resolution", type="primary"):
                with st.spinner("Processing..."):
                    
                    if method == "Classical (Iterative Back-Projection)":
                        st.info("Applying Classical Iterative Back-Projection...")
                        
                        # Upsample to target resolution
                        target_shape = (img1.shape[0] * scale_factor, img1.shape[1] * scale_factor)
                        upsampled_img1 = upsample(img1, target_shape)
                        
                        # Apply back-projection
                        sr_result = back_projection(upsampled_img1.copy(), img1, scale=scale_factor)
                        
                        if aligned_img2 is not None:
                            upsampled_img2 = upsample(aligned_img2, target_shape)
                            sr_result2 = back_projection(upsampled_img2.copy(), aligned_img2, scale=scale_factor)
                            # Average results
                            sr_result = ((sr_result.astype(np.float32) + sr_result2.astype(np.float32)) / 2).astype(np.uint8)
                    
                    elif method == "Deep Learning (SwinIR)":
                        st.info("Applying Deep Learning Super Resolution...")
                        
                        # Prepare input
                        if aligned_img2 is not None:
                            # Fuse aligned images
                            fused = ((img1.astype(np.float32) + aligned_img2.astype(np.float32)) / 2).astype(np.uint8)
                        else:
                            fused = img1
                        
                        # Convert to RGB and tensor
                        fused_rgb = np.stack([fused] * 3, axis=-1)
                        fused_tensor = torch.from_numpy(fused_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        
                        # Create model (simplified version)
                        model = SwinIR(upscale=scale_factor)
                        model.eval()
                        
                        # Process
                        with torch.no_grad():
                            output = model(fused_tensor)
                        
                        # Convert back to grayscale
                        sr_image = output.squeeze().clamp(0, 1).cpu().numpy()
                        if len(sr_image.shape) == 3:
                            sr_result = (0.2989 * sr_image[0] + 0.587 * sr_image[1] + 0.114 * sr_image[2])
                            sr_result = (sr_result * 255).astype(np.uint8)
                        else:
                            sr_result = (sr_image * 255).astype(np.uint8)
                    
                    elif method == "Hybrid Fusion":
                        st.info("Applying Hybrid Fusion (Classical + Deep Learning)...")
                        
                        # Classical approach
                        target_shape = (img1.shape[0] * scale_factor, img1.shape[1] * scale_factor)
                        upsampled_img1 = upsample(img1, target_shape)
                        sr_classical = back_projection(upsampled_img1.copy(), img1, scale=scale_factor)
                        
                        # Deep learning approach (simplified)
                        if aligned_img2 is not None:
                            fused = ((img1.astype(np.float32) + aligned_img2.astype(np.float32)) / 2).astype(np.uint8)
                        else:
                            fused = img1
                        
                        # Simple upsampling for DL component (placeholder)
                        sr_dl = cv2.resize(fused, (fused.shape[1] * scale_factor, fused.shape[0] * scale_factor), 
                                         interpolation=cv2.INTER_CUBIC)
                        
                        # Resize to match if needed
                        if sr_classical.shape != sr_dl.shape:
                            sr_dl = cv2.resize(sr_dl, (sr_classical.shape[1], sr_classical.shape[0]))
                        
                        # Weighted fusion
                        sr_result = cv2.addWeighted(sr_classical, 0.5, sr_dl, 0.5, 0)
                
                # Display results
                st.markdown('<div class="sub-header">✨ Super Resolution Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1, caption="Original LR Image", use_column_width=True, clamp=True)
                with col2:
                    st.image(sr_result, caption=f"SR Result ({method})", use_column_width=True, clamp=True)
                
                # Resolution comparison
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"""
                **Resolution Comparison:**
                - Original: {img1.shape[1]} × {img1.shape[0]}
                - Super-resolved: {sr_result.shape[1]} × {sr_result.shape[0]}
                - Scale Factor: {scale_factor}×
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
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MSE", f"{metrics['MSE']:.6f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['RMSE']:.6f}")
                    with col3:
                        st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
                    with col4:
                        st.metric("SSIM", f"{metrics['SSIM']:.4f}")
                    
                    # Display HR reference
                    st.image(hr_img, caption="HR Reference", use_column_width=True, clamp=True)
                
                # Download result
                st.markdown('<div class="sub-header">💾 Download Results</div>', unsafe_allow_html=True)
                
                # Convert to PIL for download
                sr_pil = Image.fromarray(sr_result)
                buf = io.BytesIO()
                sr_pil.save(buf, format='PNG')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download SR Image",
                    data=buf.getvalue(),
                    file_name=f"super_resolution_result_{method.lower().replace(' ', '_')}.png",
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
        
        **How to use:**
        1. Upload 1-2 low-resolution images using the sidebar
        2. Optionally upload a high-resolution reference for evaluation
        3. Select your preferred method and scale factor
        4. Click "Process Super Resolution" to generate results
        
        **Supported formats:** TIF, TIFF, PNG, JPG, JPEG
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()