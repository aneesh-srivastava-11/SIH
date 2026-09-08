import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
import itertools

from src.renderer.geometry.dem_loader import load_and_crop, row_center_lats
from src.renderer.geometry.horizon_maps import ground_spacing, surface_normals, horizon_map
from src.renderer.geometry.interface import TerrainTile
from src.renderer.config import load_config
from src.renderer.pipeline import render_tile
from src.renderer.pairing import make_pair, save_pairs, Similarity

def main():
    parser = argparse.ArgumentParser(description="Build the dataset for Part B.")
    parser.add_argument("--probe", action="store_true", help="Generate 500-pair probe dataset (T8).")
    parser.add_argument("--out_dir", type=str, default="dataset_output/bulk_set", help="Output directory")
    args = parser.parse_args()
    
    if args.probe:
        args.out_dir = "dataset_output/probe_set"
        target_pairs = 500
    else:
        target_pairs = 20000
        
    print(f"Building dataset in {args.out_dir}. Target pairs: {target_pairs}")
    os.makedirs(args.out_dir, exist_ok=True)
    
    cfg = load_config("configs/render.yaml")
    rng = np.random.default_rng(42)
    
    print("Loading base DEM...")
    dem_path = "SLDEM2015_512_30S_00S_000_045_FLOAT.LBL"
    # Take a 300x300 crop
    min_lon, max_lon = 20.0, 20.5
    min_lat, max_lat = -10.5, -10.0
    dem, transform, dlon_deg, dlat_deg = load_and_crop(dem_path, min_lon, max_lon, min_lat, max_lat)
    dem = dem * 1000.0
    lat_deg = row_center_lats(transform, dem.shape[0])
    dx, dy = ground_spacing(lat_deg, dlon_deg, dlat_deg)
    
    normals = surface_normals(dem, dx, dy).astype(np.float32)
    
    K = 16
    print("Computing horizon map...")
    horizon_deg, _ = horizon_map(dem, dx, dy, k_azimuths=K, max_dist_m=3000.0, n_steps=30)
    horizon_rad = np.radians(horizon_deg).astype(np.float32)
    
    gsd_m = dx[len(dx)//2]
    tile = TerrainTile(
        normals=normals,
        elevation=dem.astype(np.float32),
        horizon=horizon_rad,
        valid=np.ones_like(dem, dtype=bool),
        lat0=lat_deg[0],
        lon0=min_lon,
        dlat=dlat_deg,
        dlon=dlon_deg,
        gsd_m=float(gsd_m),
        K=K,
        halo_px=10  # Strip border artifacts
    )
    
    pairs_generated = 0
    batch_idx = 0
    pbar = tqdm(total=target_pairs)
    
    # Fully valid procedural albedo mask
    albedo_valid = np.ones_like(dem, dtype=bool)
    
    while pairs_generated < target_pairs:
        # 16 illuminations yields 120 unique combinations
        n_illums = min(16, max(2, target_pairs - pairs_generated + 2))
        renders = render_tile(tile, cfg, rng, n_illuminations=n_illums)
        
        pairs_to_save = []
        for r_a, r_b in itertools.combinations(renders, 2):
            if pairs_generated >= target_pairs:
                break
                
            scale = rng.uniform(*cfg.randomize.scale_ratio)
            rot = np.radians(rng.uniform(*cfg.randomize.rotation))
            tx = rng.uniform(-10, 10)
            ty = rng.uniform(-10, 10)
            
            warp = Similarity(scale=scale, rotation_rad=rot, tx=tx, ty=ty)
            
            pair = make_pair(tile, r_a, r_b, warp, cfg, albedo_valid)
            pairs_to_save.append(pair)
            pairs_generated += 1
            pbar.update(1)
            
        save_pairs(pairs_to_save, args.out_dir, prefix=f"batch_{batch_idx:04d}")
        batch_idx += 1
        
    pbar.close()
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
