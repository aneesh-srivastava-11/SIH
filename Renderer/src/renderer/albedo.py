import numpy as np
import cv2
from src.renderer.reflectance import reflectance
from dataclasses import dataclass

@dataclass
class AlbedoField:
    albedo: np.ndarray
    valid_mask: np.ndarray

def photometric_flatten(image_real: np.ndarray, normals: np.ndarray, s0: np.ndarray, v0: np.ndarray, L_table, source_shadow_mask=None, clamp=(0.02, 0.6)):
    """
    Â = I_real / R_LL(i₀,e₀,α₀), clamped. Returns (albedo, valid_mask).
    valid_mask is False where the source was shadowed (unrecoverable, F17) or clamped.
    """
    # Create a uniform albedo of 1.0 to get the pure reflectance R_LL
    unit_albedo = np.ones(image_real.shape[:2], dtype=np.float32)
    R_LL = reflectance(normals, s0, v0, unit_albedo, L_table, model="lunar_lambert")
    
    # Avoid division by zero
    valid_R = R_LL > 1e-4
    A_hat = np.zeros_like(image_real, dtype=np.float32)
    A_hat[valid_R] = image_real[valid_R] / R_LL[valid_R]
    
    valid_mask = valid_R.copy()
    if source_shadow_mask is not None:
        valid_mask = valid_mask & (~source_shadow_mask)
        
    # Clamp
    # A-TIER ASSUMPTION: Lunar normal albedo runs roughly 0.07 to 0.15. 
    # Clamp range (0.02, 0.6) catches division blowups without encoding physics.
    out_of_bounds = (A_hat < clamp[0]) | (A_hat > clamp[1])
    valid_mask = valid_mask & (~out_of_bounds)
    
    A_hat = np.clip(A_hat, clamp[0], clamp[1])
    return A_hat, valid_mask

def inpaint_albedo(albedo: np.ndarray, valid_mask: np.ndarray, method="telea") -> np.ndarray:
    """Inpaint holes using OpenCV."""
    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    mask_8u = (~valid_mask).astype(np.uint8) * 255
    inpainted = cv2.inpaint(albedo.astype(np.float32), mask_8u, inpaintRadius=3, flags=flags)
    return inpainted

def procedural_albedo(shape, rng: np.random.Generator, mean: float, scale_range: float, fbm_octaves: int, fbm_sigma: float):
    """Fractal (fBm) albedo field. Fallback path for tiles with no co-located image."""
    H, W = shape
    from scipy import ndimage
    
    # Generate simple fBm-like noise
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    for _ in range(fbm_octaves):
        # Generate white noise, blur it for low frequency
        layer = rng.normal(0, 1, size=shape).astype(np.float32)
        layer = ndimage.gaussian_filter(layer, sigma=fbm_sigma / freq)
        noise += layer * amplitude
        amplitude *= 0.5
        freq *= 2.0
        
    # Normalize noise to roughly zero mean, unit variance
    noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-6)
    
    # Scale and add mean
    A = mean + noise * scale_range
    return np.clip(A, 0.02, 0.6)

def drape(albedo: np.ndarray, valid_mask: np.ndarray, wac_dc_level=None) -> AlbedoField:
    # A-TIER ASSUMPTION: wac_dc_level is an optional DC offset from 100m mosaic
    A = albedo.copy()
    if wac_dc_level is not None:
        A += (wac_dc_level - np.mean(A[valid_mask]))
        
    return AlbedoField(albedo=A, valid_mask=valid_mask)
