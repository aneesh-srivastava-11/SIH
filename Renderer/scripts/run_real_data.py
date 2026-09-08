import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
# pyrefly: ignore [missing-import]
from src.renderer.geometry.dem_loader import load_and_crop, row_center_lats
from src.renderer.geometry.horizon_maps import ground_spacing
from scripts.render_test import render_shading_and_shadow, to_8bit
from PIL import Image

def main():
    # Pick a small footprint inside the SLDEM footprint (0 to 45 E, 0 to 30 S)
    # Longitude 20 to 20.5, Latitude -10.5 to -10.0
    min_lon, max_lon = 20.0, 20.5
    min_lat, max_lat = -10.5, -10.0
    
    dem_path = "SLDEM2015_512_30S_00S_000_045_FLOAT.LBL"
    print(f"Loading and cropping DEM from {dem_path}...")
    print(f"Footprint: Lon {min_lon} to {max_lon}, Lat {min_lat} to {max_lat}")
    
    dem, transform, dlon_deg, dlat_deg = load_and_crop(
        dem_path, min_lon, max_lon, min_lat, max_lat
    )
    
    print(f"DEM cropped shape: {dem.shape}")
    print(f"dlon_deg: {dlon_deg:.5f}, dlat_deg: {dlat_deg:.5f}")
    
    # The PDS3 LBL dataset encodes elevation in Kilometers, but our pipeline expects Meters.
    # We must convert the elevation to meters.
    dem = dem * 1000.0
    
    # Get ground spacing
    lat_deg = row_center_lats(transform, dem.shape[0])
    dx, dy = ground_spacing(lat_deg, dlon_deg, dlat_deg)
    
    # Render at a specific sun geometry
    sun_az = 270.0
    sun_el = 10.0
    
    print(f"Rendering shadow map for Sun Azimuth {sun_az}, Elevation {sun_el}...")
    image, normals, shadow = render_shading_and_shadow(
        dem, dx, dy,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        k_azimuths=16, max_dist_m=3000.0, n_steps=30
    )
    
    print(f"Shadowed fraction: {shadow.mean():.3f}")
    
    img8 = to_8bit(image)
    
    # Save as PNG
    out_filename = "render_output.png"
    Image.fromarray(img8).save(out_filename)
    print(f"Saved render output to {out_filename}")

if __name__ == "__main__":
    main()
