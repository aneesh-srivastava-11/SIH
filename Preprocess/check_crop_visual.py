"""
Visual sanity check for crop_to_overlap.py output.

Loads img1_crop.npy and img2_crop.npy from a crop output folder and
saves a side-by-side PNG so you can eyeball whether the two images
actually show the same patch of terrain (even though sun angle /
sharpness / instrument type differ, you should be able to recognize
matching crater shapes, ridge lines, etc. between the two panels).

This does NOT do any actual alignment or matching -- it's a dumb
"does this look like the same place at all" check before spending time
on the real matcher.

Usage:
    python check_crop_visual.py ..\\data\\cropped\\p1 --title "P1: OHRC vs NAC"
    python check_crop_visual.py ..\\data\\cropped\\p3 --title "P3: IIRS band52 vs WAC"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt


def normalize_for_display(arr, pct_clip=1.0, max_dim=1500):
    """
    Stretch any dtype/range to 0-255 for display. Two safety measures:

    1. Downsample large images BEFORE any float conversion. A full-res
       OHRC crop can be ~90000x12000 = ~1 billion pixels; converting
       that to float64 needs ~8GB just for one array. A sanity-check
       plot can't show more than a couple thousand pixels usefully
       anyway, so downsampling first avoids the memory blowup entirely.
    2. Detect and exclude NoData sentinel values before computing
       percentiles. Signed calibrated products (e.g. NAC CDR, int16)
       commonly use the dtype's minimum value (e.g. -32768) to flag
       "no valid data" pixels -- including those in the percentile
       calculation would badly skew the stretch.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    step = max(1, max(h, w) // max_dim)
    small = arr[::step, ::step]

    nodata_mask = None
    if np.issubdtype(small.dtype, np.signedinteger):
        dtype_min = np.iinfo(small.dtype).min
        if np.any(small == dtype_min):
            nodata_mask = (small == dtype_min)
            frac = nodata_mask.mean()
            print(f"  NOTE: detected likely NoData sentinel value ({dtype_min}) "
                  f"in {frac*100:.1f}% of (downsampled) pixels -- excluding from "
                  "contrast stretch and rendering as black.")

    valid = small[~nodata_mask] if nodata_mask is not None else small
    valid = valid.astype(np.float32)

    if valid.size == 0:
        print("  WARNING: entire image is NoData after masking -- nothing to display.")
        return np.zeros_like(small, dtype=np.uint8)

    lo = np.percentile(valid, pct_clip)
    hi = np.percentile(valid, 100 - pct_clip)
    if hi <= lo:
        return np.zeros_like(small, dtype=np.uint8)

    small_f = small.astype(np.float32)
    stretched = np.clip((small_f - lo) / (hi - lo), 0, 1) * 255
    result = stretched.astype(np.uint8)
    if nodata_mask is not None:
        result[nodata_mask] = 0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crop_dir", help="Folder containing img1_crop.npy / img2_crop.npy "
                                      "(output of crop_to_overlap.py)")
    ap.add_argument("--title", default=None, help="Title for the figure")
    ap.add_argument("--out", default=None, help="Output PNG path (default: crop_dir\\sanity_check.png)")
    args = ap.parse_args()

    crop_dir = Path(args.crop_dir)
    img1_path = crop_dir / "img1_crop.npy"
    img2_path = crop_dir / "img2_crop.npy"

    if not img1_path.exists() or not img2_path.exists():
        raise FileNotFoundError(
            f"Expected both {img1_path} and {img2_path} to exist -- "
            "run crop_to_overlap.py first."
        )

    img1 = np.load(img1_path)
    img2 = np.load(img2_path)

    print(f"img1: shape={img1.shape}, dtype={img1.dtype}, range=[{img1.min()},{img1.max()}]")
    print(f"img2: shape={img2.shape}, dtype={img2.dtype}, range=[{img2.min()},{img2.max()}]")

    img1_disp = normalize_for_display(img1)
    img2_disp = normalize_for_display(img2)

    bounds_path = crop_dir / "overlap_bounds.json"
    bounds_note = ""
    if bounds_path.exists():
        with open(bounds_path) as f:
            b = json.load(f)
        bounds_note = (f"  Overlap: lon[{b['lon_min']:.4f},{b['lon_max']:.4f}]  "
                        f"lat[{b['lat_min']:.4f},{b['lat_max']:.4f}]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].imshow(img1_disp, cmap="gray")
    axes[0].set_title(f"img1  {img1.shape}")
    axes[0].axis("off")

    axes[1].imshow(img2_disp, cmap="gray")
    axes[1].set_title(f"img2  {img2.shape}")
    axes[1].axis("off")

    fig_title = args.title or f"Crop sanity check: {crop_dir}"
    fig.suptitle(fig_title + "\n" + bounds_note, fontsize=11)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else crop_dir / "sanity_check.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison image to: {out_path}")
    print("Open it and check: do both panels look like they show the same terrain")
    print("(matching crater rims / ridge lines), even with different sharpness,")
    print("sun angle, or contrast? If one panel looks like noise, a solid color,")
    print("or clearly different terrain, something upstream is wrong -- don't")
    print("proceed to the matcher until this looks visually sane.")


if __name__ == "__main__":
    main()