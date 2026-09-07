import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from src.renderer.geometry.dem_loader import load_and_crop, row_center_lats
from src.renderer.geometry.horizon_maps import ground_spacing, surface_normals, horizon_map
from src.renderer.geometry.interface import TerrainTile
from src.renderer.config import load_config
from src.renderer.pipeline import render_tile

def main():
    print("Loading real tile (Part A)...")
    min_lon, max_lon = 20.0, 20.5
    min_lat, max_lat = -10.5, -10.0
    dem_path = "SLDEM2015_512_30S_00S_000_045_FLOAT.LBL"
    
    dem, transform, dlon_deg, dlat_deg = load_and_crop(dem_path, min_lon, max_lon, min_lat, max_lat)
    dem = dem * 1000.0
    lat_deg = row_center_lats(transform, dem.shape[0])
    dx, dy = ground_spacing(lat_deg, dlon_deg, dlat_deg)
    
    normals = surface_normals(dem, dx, dy).astype(np.float32)
    
    # Precompute horizon map (simulating the cached Part A asset)
    K = 16
    print(f"Precomputing horizon map for {dem.shape} tile...")
    horizon_deg, _ = horizon_map(dem, dx, dy, k_azimuths=K, max_dist_m=3000.0, n_steps=30)
    horizon_rad = np.radians(horizon_deg).astype(np.float32)
    
    # Construct TerrainTile
    # gsd_m roughly the spacing at the center latitude
    gsd_m = dx[len(dx)//2]
    
    tile = TerrainTile(
        normals=normals,
        elevation=dem.astype(np.float32),
        horizon=horizon_rad,
        valid=np.ones_like(dem, dtype=bool), # Assumed valid
        lat0=lat_deg[0],
        lon0=min_lon,
        dlat=dlat_deg,
        dlon=dlon_deg,
        gsd_m=float(gsd_m),
        K=K,
        halo_px=0
    )
    
    print("Running Part B Pipeline...")
    cfg = load_config("configs/render.yaml")
    rng = np.random.default_rng(42)
    
    n_illums = 8
    
    t0 = time.time()
    renders = render_tile(tile, cfg, rng, n_illuminations=n_illums)
    t1 = time.time()
    
    duration = t1 - t0
    
    pairs_per_tile = 28 # 8 choose 2
    target_pairs = 20000
    tiles_needed = target_pairs / pairs_per_tile
    total_time_hours = (duration * tiles_needed) / 3600.0
    
    print(f"\n--- Throughput Results (T4.3) ---")
    print(f"Rendered 1 tile with {n_illums} illuminations in {duration:.2f} seconds.")
    print(f"This represents {pairs_per_tile} image pairs.")
    print(f"Time per pair: {duration / pairs_per_tile:.4f} seconds.")
    print(f"Extrapolated time for {target_pairs} pairs: {total_time_hours:.2f} hours.")
    print(f"---------------------------------")
    
if __name__ == "__main__":
    main()
