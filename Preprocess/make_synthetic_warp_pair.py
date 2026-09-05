"""
Generate a synthetic-warp ground-truth pair: take one real cropped image,
apply a KNOWN random homography (rotation, scale, translation, mild
perspective) plus a photometric perturbation, and save both images plus
the exact ground-truth transform.

Why this matters (project doc section 9.2): this is the only source of
EXACT ground truth on real imagery. Any matcher's estimated homography
can be compared directly against the known H_gt -- error = ||H_est -
H_gt|| on a grid of points. It does NOT test cross-sensor/cross-
illumination robustness by itself (the photometric perturbation is a
crude stand-in for that) -- pair it with real manual control points
later for the full picture.

Usage:
    python make_synthetic_warp_pair.py --input raw/p1_ohrc.npy --out-dir pairs/p1_synthetic_warp
    python make_synthetic_warp_pair.py --input raw/p1_ohrc.npy --out-dir pairs/p1_synthetic_warp --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2


def random_homography(img_shape, max_rotation_deg=30, scale_range=(0.5, 2.0),
                       max_translation_frac=0.1, max_perspective=0.0002, rng=None):
    """
    Build a random homography: rotation + scale + translation + mild
    perspective warp, centered on the image so rotation/scale don't fling
    content off-frame immediately.
    """
    rng = rng or np.random.default_rng()
    h, w = img_shape[-2], img_shape[-1]
    cx, cy = w / 2, h / 2

    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    scale = rng.uniform(*scale_range)
    tx = rng.uniform(-max_translation_frac, max_translation_frac) * w
    ty = rng.uniform(-max_translation_frac, max_translation_frac) * h

    # Rotation + scale about image center, as a similarity transform
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta) * scale, np.sin(theta) * scale
    M = np.array([
        [cos_t, -sin_t, (1 - cos_t) * cx + sin_t * cy + tx],
        [sin_t,  cos_t, -sin_t * cx + (1 - cos_t) * cy + ty],
        [0,      0,      1],
    ], dtype=np.float64)

    # Mild perspective perturbation on top
    p1 = rng.uniform(-max_perspective, max_perspective)
    p2 = rng.uniform(-max_perspective, max_perspective)
    P = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]], dtype=np.float64)

    H = P @ M
    H = H / H[2, 2]
    return H, {"rotation_deg": angle, "scale": scale, "tx": tx, "ty": ty,
               "perspective_p1": p1, "perspective_p2": p2}


def apply_photometric_perturbation(img, rng, brightness_range=(0.7, 1.3),
                                    gamma_range=(0.7, 1.4), noise_std_frac=0.01):
    """
    Crude stand-in for real illumination differences: brightness scale,
    gamma curve, and a little sensor noise. This is NOT a substitute for
    real cross-illumination testing (no shadows are added/moved) -- it
    only tests the matcher's tolerance to simple photometric changes,
    on top of the exact geometric ground truth from the homography.
    """
    img_f = img.astype(np.float64)
    max_val = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else img_f.max()

    brightness = rng.uniform(*brightness_range)
    gamma = rng.uniform(*gamma_range)

    normed = np.clip(img_f / max_val, 0, 1)
    normed = np.clip(normed * brightness, 0, 1) ** gamma
    noisy = normed + rng.normal(0, noise_std_frac, normed.shape)
    out = np.clip(noisy, 0, 1) * max_val

    return out.astype(img.dtype), {"brightness": brightness, "gamma": gamma,
                                    "noise_std_frac": noise_std_frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a real cropped .npy image")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-dim", type=int, default=2000,
                     help="Downsample input if larger than this on either axis "
                          "(keeps warp + I/O fast; full-res OHRC crops are huge)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.load(args.input)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}. "
                          "If this is a multi-band cube, extract a band first.")

    h, w = img.shape
    if max(h, w) > args.max_dim:
        step = max(h, w) // args.max_dim
        img = img[::step, ::step]
        print(f"Downsampled input from ({h},{w}) to {img.shape} (step={step})")

    H, h_params = random_homography(img.shape, rng=rng)
    h_out, w_out = img.shape

    warped = cv2.warpPerspective(img, H, (w_out, h_out),
                                  flags=cv2.INTER_LINEAR, borderValue=0)

    warped, photo_params = apply_photometric_perturbation(warped, rng)

    np.save(out_dir / "img_original.npy", img)
    np.save(out_dir / "img_warped.npy", warped)
    np.save(out_dir / "H_ground_truth.npy", H)

    with open(out_dir / "ground_truth_params.json", "w") as f:
        json.dump({
            "source_file": str(args.input),
            "image_shape": list(img.shape),
            "homography_row_major": H.tolist(),
            "homography_params": h_params,
            "photometric_params": photo_params,
            "seed": args.seed,
            "usage_note": (
                "H_ground_truth.npy maps pixel coords in img_original.npy to "
                "pixel coords in img_warped.npy: p_warped = H @ p_original "
                "(homogeneous coords). Compare any estimated homography "
                "against this directly, or transform a grid of test points "
                "through both and measure pixel distance."
            ),
        }, f, indent=2)

    print(f"\nSaved to {out_dir}:")
    print(f"  img_original.npy  shape={img.shape}")
    print(f"  img_warped.npy    shape={warped.shape}")
    print(f"  H_ground_truth.npy")
    print(f"  ground_truth_params.json")
    print(f"\nApplied transform: {h_params}")
    print(f"Applied photometric perturbation: {photo_params}")


if __name__ == "__main__":
    main()