import numpy as np
import cv2
import json
import os
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image

@dataclass
class Similarity:
    scale: float
    rotation_rad: float
    tx: float = 0.0
    ty: float = 0.0

@dataclass
class PairSample:
    image_a: np.ndarray
    image_b: np.ndarray
    transform: np.ndarray  # 3x3 affine matrix mapping A -> warped B
    match_mask: np.ndarray # (H,W) bool
    meta: dict

def make_pair(tile, render_a: dict, render_b: dict, warp: Similarity, cfg, albedo_valid_mask: np.ndarray) -> PairSample:
    """
    render_a/b are dicts from `pipeline.py` with: "image", "params", "shadow_mask"
    GT correspondence in the unwarped tile grid is the IDENTITY.
    Apply similarity warp W to member B. The correspondence from A to warped B is W.
    """
    img_a = render_a["image"]
    img_b_orig = render_b["image"]
    H, W = img_a.shape
    
    # Center of rotation/scaling
    center = (W / 2.0, H / 2.0)
    
    # 1. Compute affine transform matrix M
    # cv2.getRotationMatrix2D expects degrees. Positive is counter-clockwise.
    M = cv2.getRotationMatrix2D(center, np.degrees(warp.rotation_rad), warp.scale)
    M[0, 2] += warp.tx
    M[1, 2] += warp.ty
    
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :] = M
    
    # Warp B image
    interp_b = cv2.INTER_LINEAR if img_b_orig.dtype in (np.uint8, np.uint16, np.float32) else cv2.INTER_NEAREST
    img_b_warped = cv2.warpAffine(img_b_orig, M, (W, H), flags=interp_b, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Apply exclusion rules to form match_mask on A's grid
    exclude = np.zeros((H, W), dtype=bool)
    
    # 1. Outside tile.valid
    exclude |= ~tile.valid
    
    # 2. Inside halo_px of the edge
    halo = getattr(tile, 'halo_px', 0)
    if halo > 0:
        exclude[:halo, :] = True
        exclude[-halo:, :] = True
        exclude[:, :halo] = True
        exclude[:, -halo:] = True
        
    # 3. albedo_valid == False (holes in drape)
    exclude |= ~albedo_valid_mask
    
    # 4. Shadowed in either member
    # Note: shadow_b_orig[p_a] == True implies the corresponding pixel p_w in warped B is also shadowed!
    exclude |= render_a["shadow_mask"]
    exclude |= render_b["shadow_mask"]
    
    # 5. Outside warp's valid support in B
    # A pixel p_a is mapped to p_w = M * p_a. If p_w is outside [0, W-1] x [0, H-1], it is out of bounds.
    y_indices, x_indices = np.indices((H, W))
    pts_a = np.stack([x_indices.ravel(), y_indices.ravel(), np.ones(H*W)], axis=1) # (N, 3)
    pts_w = pts_a @ M.T # (N, 2)
    
    x_w = pts_w[:, 0].reshape((H, W))
    y_w = pts_w[:, 1].reshape((H, W))
    
    # To be safe, valid pixels must land inside the image footprint. 
    # Use -0.5 to W-0.5 to align with pixel centers/OpenCV interpolation bounds.
    exclude_out_of_bounds = (x_w < 0) | (x_w > W - 1) | (y_w < 0) | (y_w > H - 1)
    exclude |= exclude_out_of_bounds
    
    match_mask = ~exclude
    
    # Record metadata
    meta = {
        "sun_az_a": render_a["params"]["sun_az_rad"],
        "sun_el_a": render_a["params"]["sun_el_rad"],
        "sun_az_b": render_b["params"]["sun_az_rad"],
        "sun_el_b": render_b["params"]["sun_el_rad"],
        "delta_az": render_b["params"]["sun_az_rad"] - render_a["params"]["sun_az_rad"],
        "delta_el": render_b["params"]["sun_el_rad"] - render_a["params"]["sun_el_rad"],
        "transform": transform.tolist(),
        "scale": warp.scale,
        "rotation_rad": warp.rotation_rad
    }
    
    return PairSample(
        image_a=img_a,
        image_b=img_b_warped,
        transform=transform,
        match_mask=match_mask,
        meta=meta
    )

def save_pairs(pairs: List[PairSample], output_dir: str, prefix: str):
    """
    Save pairs as PNG and NPZ.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{prefix}_manifest.jsonl")
    
    with open(manifest_path, "w") as f:
        for i, pair in enumerate(pairs):
            pair_id = f"{prefix}_{i:05d}"
            
            # Save PNGs
            path_a = os.path.join(output_dir, f"{pair_id}_a.png")
            path_b = os.path.join(output_dir, f"{pair_id}_b.png")
            
            # Handle float vs uint
            if pair.image_a.dtype == np.uint16:
                img_a = Image.fromarray(pair.image_a, mode='I;16')
                img_b = Image.fromarray(pair.image_b, mode='I;16')
            else:
                img_a = Image.fromarray(pair.image_a)
                img_b = Image.fromarray(pair.image_b)
                
            img_a.save(path_a)
            img_b.save(path_b)
            
            # Save mask and transform to NPZ
            npz_path = os.path.join(output_dir, f"{pair_id}_meta.npz")
            np.savez_compressed(
                npz_path, 
                match_mask=pair.match_mask, 
                transform=pair.transform
            )
            
            # Write to manifest
            manifest_entry = pair.meta.copy()
            manifest_entry["pair_id"] = pair_id
            manifest_entry["image_a"] = os.path.basename(path_a)
            manifest_entry["image_b"] = os.path.basename(path_b)
            manifest_entry["npz_meta"] = os.path.basename(npz_path)
            
            f.write(json.dumps(manifest_entry) + "\n")
