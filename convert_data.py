import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import glob

def get_adaptive_contrast(arr):
    # Mask out zero-padded background pixels (pixel > 0)
    valid_pixels = arr[arr > 0]
    if len(valid_pixels) == 0:
        return arr.astype(np.uint8)
        
    p1, p99 = np.percentile(valid_pixels, (1, 99))
    arr_clipped = np.clip(arr, p1, p99)
    
    # Scale valid range to 0-255 uint8
    arr_norm = cv2.normalize(arr_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply CLAHE to boost local crater contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    arr_enhanced = clahe.apply(arr_norm)
    return arr_enhanced

def clean_pairs_dir(dest_dir):
    """Remove any old .png or .tif files from the pairs directory to enforce our data scope."""
    if not os.path.exists(dest_dir):
        return
    for f in glob.glob(os.path.join(dest_dir, "*.png")) + glob.glob(os.path.join(dest_dir, "*.tif")):
        try:
            os.remove(f)
            print(f"Removed old file: {f}")
        except Exception as e:
            print(f"Warning: could not remove {f} - {e}")

def convert_npy_to_tif():
    src_dir = r"c:\Users\ANEESH\Desktop\SIH\data\cropped"
    dest_dir = r"c:\Users\ANEESH\Desktop\SIH\basebenchmarking\data\pairs"
    
    os.makedirs(dest_dir, exist_ok=True)
    clean_pairs_dir(dest_dir)
    
    # Process only p1 and p3
    target_folders = ["p1", "p3"]
    
    for folder in target_folders:
        folder_path = os.path.join(src_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder {folder} not found in {src_dir}")
            continue
            
        print(f"Processing {folder}...")
        
        # Collect all .npy files in folder
        npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        if len(npy_files) >= 2:
            # Sort for deterministic reference/target assignment
            npy_files.sort()
            ref_npy = os.path.join(folder_path, npy_files[0])
            tgt_npy = os.path.join(folder_path, npy_files[1])
            
            ref_arr = np.load(ref_npy)
            tgt_arr = np.load(tgt_npy)
            
            # Use adaptive contrast instead of simple normalize
            ref_arr = get_adaptive_contrast(ref_arr)
            tgt_arr = get_adaptive_contrast(tgt_arr)
            
            # Load overlap bounds to generate transform
            bounds_path = os.path.join(folder_path, "overlap_bounds.json")
            if os.path.exists(bounds_path):
                with open(bounds_path, "r") as f:
                    bounds = json.load(f)
                lon_min = bounds["lon_min"]
                lon_max = bounds["lon_max"]
                lat_min = bounds["lat_min"]
                lat_max = bounds["lat_max"]
            else:
                lon_min, lon_max, lat_min, lat_max = 0, ref_arr.shape[1], 0, ref_arr.shape[0]
                
            ref_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ref_arr.shape[1], ref_arr.shape[0])
            tgt_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, tgt_arr.shape[1], tgt_arr.shape[0])
            
            # Save to pairs directory as GeoTIFF
            ref_path = os.path.join(dest_dir, f"ref_{folder}.tif")
            tgt_path = os.path.join(dest_dir, f"tgt_{folder}.tif")
            
            crs = CRS.from_epsg(4326)
            
            with rasterio.open(
                ref_path,
                'w',
                driver='GTiff',
                height=ref_arr.shape[0],
                width=ref_arr.shape[1],
                count=1,
                dtype=ref_arr.dtype,
                crs=crs,
                transform=ref_transform,
            ) as dst:
                dst.write(ref_arr, 1)
                
            with rasterio.open(
                tgt_path,
                'w',
                driver='GTiff',
                height=tgt_arr.shape[0],
                width=tgt_arr.shape[1],
                count=1,
                dtype=tgt_arr.dtype,
                crs=crs,
                transform=tgt_transform,
            ) as dst:
                dst.write(tgt_arr, 1)
                
            print(f"Saved {ref_path} and {tgt_path}")
        else:
            print(f"Insufficient .npy files in {folder}: found {len(npy_files)}")

if __name__ == "__main__":
    convert_npy_to_tif()
