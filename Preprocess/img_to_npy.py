"""
Dump a raster file's pixel data to a .npy array.

Works for NAC CDR .IMG files (and similar) that GDAL/rasterio can open
as raw pixel data even though they carry no map projection/geotransform
-- exactly the situation we hit with M1529798616LC.IMG (CRS: None,
identity geotransform). This script doesn't need or use any
georeferencing; it just reads band 1 pixel values.

Usage:
    python img_to_npy.py path\\to\\M1529798616LC.IMG
    python img_to_npy.py path\\to\\M1529798616LC.IMG --out data\\nac_array.npy
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to the .IMG (or other raster) file")
    ap.add_argument("--out", default=None,
                     help="Output .npy path (default: same name, .npy extension)")
    ap.add_argument("--band", type=int, default=1,
                     help="Band to read (1-indexed, default: 1)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".npy")

    # NAC CDR .IMG files trigger a harmless "no geotransform" warning --
    # expected here since we're intentionally not relying on georeferencing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(in_path) as src:
            print(f"Opened: {in_path}")
            print(f"  Bands: {src.count}, dtype: {src.dtypes}, shape: ({src.height}, {src.width})")
            if src.crs is not None:
                print(f"  NOTE: this file DOES have a CRS ({src.crs}) -- "
                      "if you expected it to be unreferenced, double check you have the right file.")
            arr = src.read(args.band)

    np.save(out_path, arr)
    print(f"\nSaved array shape={arr.shape}, dtype={arr.dtype} to: {out_path}")
    print(f"Min/Max pixel value: {arr.min()} / {arr.max()}")


if __name__ == "__main__":
    main()