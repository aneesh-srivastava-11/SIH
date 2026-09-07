import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def compute_ncc(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    v1 = img1[mask].astype(np.float32)
    v2 = img2[mask].astype(np.float32)
    
    # Normalize zero mean
    v1 -= np.mean(v1)
    v2 -= np.mean(v2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))

def compute_ssim_masked(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    # Scale to 0-1 for SSIM
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    data_range = float(np.maximum(img1_f.max(), img2_f.max()) - np.minimum(img1_f.min(), img2_f.min()))
    if data_range == 0:
        return 1.0
        
    _, ssim_map = ssim(img1_f, img2_f, data_range=data_range, full=True)
    if not np.any(mask):
        return 0.0
    return float(np.mean(ssim_map[mask]))

def gradient_histogram_distance(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, bins=36) -> float:
    gx1 = cv2.Sobel(img1.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(img1.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    gx2 = cv2.Sobel(img2.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(img2.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    mag1, ang1 = cv2.cartToPolar(gx1, gy1, angleInDegrees=True)
    mag2, ang2 = cv2.cartToPolar(gx2, gy2, angleInDegrees=True)
    
    # Only keep masked regions with some gradient to avoid flat regions dominating
    thresh1 = np.percentile(mag1[mask], 50) if np.any(mask) else 0
    thresh2 = np.percentile(mag2[mask], 50) if np.any(mask) else 0
    
    valid1 = mask & (mag1 > thresh1)
    valid2 = mask & (mag2 > thresh2)
    
    if not np.any(valid1) or not np.any(valid2):
        return 0.0
        
    hist1, _ = np.histogram(ang1[valid1], bins=bins, range=(0, 360), density=True)
    hist2, _ = np.histogram(ang2[valid2], bins=bins, range=(0, 360), density=True)
    
    # L1 distance
    return float(np.sum(np.abs(hist1 - hist2)))

def validate_render_vs_real(rendered: np.ndarray, real: np.ndarray, shadow_mask: np.ndarray) -> dict:
    mask = ~shadow_mask
    return {
        "ncc": compute_ncc(rendered, real, mask),
        "ssim": compute_ssim_masked(rendered, real, mask),
        "grad_hist_l1": gradient_histogram_distance(rendered, real, mask)
    }
