import numpy as np
from typing import List
from src.renderer.config import RenderConfig
from src.renderer.randomize import draw_parameters
from src.renderer.reflectance import sun_vector, reflectance
from src.renderer.albedo import procedural_albedo, drape, AlbedoField
from src.renderer.sensor import apply_sensor
from src.renderer.geometry.interface import TerrainTile, cast_shadow

def dummy_L_table(alpha_rad):
    # A-TIER ASSUMPTION: Linear placeholder for L_table until pho_emp_local is available.
    return 0.5

def render_tile(tile: TerrainTile, cfg: RenderConfig, rng: np.random.Generator, n_illuminations: int = 8) -> List[dict]:
    renders = []
    
    # 1. Base Albedo (cache per-tile invariant)
    H, W = tile.elevation.shape
    ac = cfg.albedo
    raw_albedo = procedural_albedo(
        (H, W), rng, 
        mean=ac.procedural_mean, 
        scale_range=ac.procedural_scale, 
        fbm_octaves=ac.fbm_octaves, 
        fbm_sigma=ac.fbm_sigma
    )
    albedo_field = drape(raw_albedo, tile.valid)
    
    v = np.array([0, 0, 1], dtype=np.float32)
    
    for _ in range(n_illuminations):
        params = draw_parameters(cfg, rng)
        
        # Sensor config overrides
        sensor_cfg = cfg.sensor.model_dump()
        sensor_cfg["psf_sigma"] = params["psf_sigma"]
        sensor_cfg["read_noise_e"] = params["read_noise_e"]
        sensor_cfg["secondary_eps"] = params["secondary_eps"]
        # Jitter is not randomized here per the param table, but could be added.
        
        # Geometry
        s = sun_vector(params["sun_az_rad"], params["sun_el_rad"])
        shadows = cast_shadow(tile, params["sun_az_rad"], params["sun_el_rad"])
        
        # Reflectance
        A = albedo_field.albedo * params["albedo_scale"]
        R = reflectance(tile.normals, s, v, A, L_table=dummy_L_table, model=cfg.reflectance.model, eps=cfg.reflectance.eps)
        
        # Direct reflectance is 0 in shadow.
        R[shadows] = 0.0
        
        # Sensor pipeline (adds ambient light into shadows)
        dn = apply_sensor(R, sensor_cfg, rng)
        
        renders.append({
            "image": dn,
            "params": params,
            "shadow_mask": shadows
        })
        
    return renders
