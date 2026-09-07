import numpy as np
from src.renderer.config import RenderConfig

def draw_parameters(cfg: RenderConfig, rng: np.random.Generator, polar=False) -> dict:
    """Draw a randomized parameter set for a single render."""
    rc = cfg.randomize
    
    el_range = rc.elevation_polar if polar else rc.elevation_eq
    
    return {
        "sun_az_rad": np.radians(rng.uniform(*rc.azimuth)),
        "sun_el_rad": np.radians(rng.uniform(*el_range)),
        "albedo_scale": rng.uniform(*rc.albedo_scale),
        "additive_noise_sigma": rng.uniform(*rc.additive_noise_sigma),
        "roughness_amp": rng.uniform(*rc.roughness_amp),
        "psf_sigma": rng.uniform(*rc.psf_sigma),
        "read_noise_e": rng.uniform(*rc.read_noise_e),
        "secondary_eps": rng.uniform(*rc.secondary_eps),
        "scale_ratio": rng.uniform(*rc.scale_ratio),
        "rotation_rad": np.radians(rng.uniform(*rc.rotation))
    }
