import numpy as np
from scipy import ndimage

def apply_sensor(radiance: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """
    radiance: (H,W) float32, linear, unitless reflectance × albedo.
    Returns uint8 or uint16 DN.

    ORDER — do not reorder:
      1. line_jitter
      2. secondary_illum
      3. to_electrons
      4. psf_blur
      5. shot_noise
      6. read_noise
      7. gain_offset
      8. quantize
      9. to_8bit
    """
    H, W = radiance.shape
    rad = radiance.copy()
    
    # 1. line_jitter
    jitter_sigma = cfg.get("jitter_sigma", 0.0)
    if jitter_sigma > 0:
        # low-order random walk for row shifts
        steps = rng.normal(0, jitter_sigma, size=H)
        shifts = np.cumsum(steps)
        # remove mean drift
        shifts -= np.mean(shifts)
        
        yy, xx = np.indices((H, W))
        xx_shifted = xx - shifts[:, None]
        rad = ndimage.map_coordinates(rad, [yy, xx_shifted], mode='nearest', order=1)
        
    # 2. secondary_illum
    # eps fraction of the local unshadowed mean, added inside shadows
    eps = cfg.get("secondary_eps", 0.0)
    if eps > 0:
        # A-TIER ASSUMPTION: unshadowed mean is approximated by mean of pixels > 1e-4
        unshadowed = rad[rad > 1e-4]
        if len(unshadowed) > 0:
            ambient_estimate = np.mean(unshadowed)
            rad += eps * ambient_estimate
            
    # 3. to_electrons
    signal_scale = cfg.get("signal_scale", 1.0)
    e = rad * signal_scale
    
    # 4. psf_blur
    psf_sigma = cfg.get("psf_sigma", 0.0)
    if psf_sigma > 0:
        e = ndimage.gaussian_filter(e, sigma=psf_sigma)
        
    # 5. shot_noise
    # lambda=0 -> 0 electrons
    e_noisy = rng.poisson(np.clip(e, 0, None)).astype(np.float32)
    
    # 6. read_noise
    read_noise_e = cfg.get("read_noise_e", 0.0)
    if read_noise_e > 0:
        e_noisy += rng.normal(0, read_noise_e, size=(H, W))
        
    # 7. gain_offset
    gain = cfg.get("gain", 1.0)
    offset = cfg.get("offset", 0.0)
    dn = e_noisy / gain + offset
    
    # 8. quantize
    bits = cfg.get("bits", 16)
    max_val = (1 << bits) - 1
    dn_quant = np.clip(np.round(dn), 0, max_val)
    
    if bits <= 8:
        out = dn_quant.astype(np.uint8)
    else:
        out = dn_quant.astype(np.uint16)
        
    # 9. to_8bit
    if cfg.get("to_8bit", False) and bits > 8:
        # Scale to 8 bit
        out = (out / max_val * 255).astype(np.uint8)
        
    return out
