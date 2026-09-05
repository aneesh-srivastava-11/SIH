"""
Crop two images down to their shared geographic overlap region.

v2: unified approach. Earlier versions assumed reference images (NAC/WAC)
had exact map-projected GeoTIFF bounds we could read directly. That's
WRONG for NAC CDR (confirmed empirically: no CRS, no geotransform -- CDR
is radiometrically calibrated but NOT orthorectified). It also needed a
fix for WAC, whose GeoTIFF *is* projected, but in meters on a custom Moon
Equirectangular CRS, not plain lat/lon degrees.

So now EVERY image is handled the same way: get its 4 corner lat/lon
points (from a PDS4/LROC label, or derived from a GeoTIFF's projected
bounds), fit a homography from pixel-space to those corners, and use
that to crop to any lat/lon box. Two source types are supported:

  --imgN-array + --imgN-geometry   : .npy array + corners.json
                                      (OHRC, IIRS, NAC -- no real map
                                      projection, just 4 known corners)
  --imgN-tif                        : a real GeoTIFF (WAC). Corners are
                                      derived automatically from its
                                      embedded transform + CRS.

Usage:
    python crop_to_overlap.py \\
        --img1-array data\\ohrc_array.npy --img1-geometry data\\ohrc_geometry.json \\
        --img2-array data\\nac_array.npy  --img2-geometry data\\nac_geometry.json \\
        --out-dir data\\cropped\\p1

    python crop_to_overlap.py \\
        --img1-array data\\iirs_array.npy --img1-geometry data\\iirs_geometry.json --img1-band 52 \\
        --img2-tif data\\wac_crop\\LunarLROLROC-WAC_MAP2_SIMP.tif \\
        --out-dir data\\cropped\\p3
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import cv2

try:
    import rasterio
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install rasterio --break-system-packages")


MOON_RADIUS_M = 1737400.0  # meters; used only for simple-cylindrical/equirectangular unprojection


def corners_dict_to_array(corners):
    """corners = {'upper_left': [lat, lon], ...} -> (4,2) array, order UL,UR,LR,LL"""
    order = ["upper_left", "upper_right", "lower_right", "lower_left"]
    return np.array([[float(corners[k][0]), float(corners[k][1])] for k in order],
                     dtype=np.float64)


def fit_pixel_to_lonlat_homography(latlon_corners, height, width):
    """Fit pixel(col,row) -> (lon,lat) homography from 4 known corners."""
    pixel_corners = np.array([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    lonlat_corners = latlon_corners[:, [1, 0]]  # swap to (lon, lat)
    H, _ = cv2.findHomography(pixel_corners, lonlat_corners, method=0)
    if H is None:
        raise RuntimeError("Homography fit failed -- check that corners are not degenerate/collinear.")
    return H, np.linalg.inv(H)


def load_corner_source(array_path, geometry_path, band=None):
    """Load an .npy array + its corners.json (OHRC/IIRS/NAC style)."""
    arr = np.load(array_path)
    with open(geometry_path) as f:
        geom = json.load(f)
    corners = geom.get("corners")
    if not corners:
        raise ValueError(f"No 'corners' key found in {geometry_path}")

    if band is not None:
        if arr.ndim != 3:
            raise ValueError(f"--band given but {array_path} is not a 3D cube (shape={arr.shape})")
        arr = arr[band]

    latlon_corners = corners_dict_to_array(corners)
    H, H_inv = fit_pixel_to_lonlat_homography(latlon_corners, arr.shape[-2], arr.shape[-1])
    return arr, latlon_corners, H, H_inv


def detect_equirectangular_radius(crs):
    """
    Pull the spheroid radius out of a WKT string like:
    ...SPHEROID["Moon",1737400,0]...
    Returns MOON_RADIUS_M as a fallback if not found -- reasonable for
    any lunar product, but flagged in case this script is reused for
    another body later.
    """
    wkt = crs.to_wkt()
    match = re.search(r'SPHEROID\["[^"]*",\s*([\d.]+)', wkt)
    if match:
        return float(match.group(1))
    print("WARNING: could not detect spheroid radius from CRS; assuming Moon radius "
          f"({MOON_RADIUS_M} m). Verify this is correct for your data.")
    return MOON_RADIUS_M


def load_geotiff_source(tif_path):
    """
    Load a real GeoTIFF. Handles two cases:
      - CRS is already geographic (degrees) -> use bounds directly.
      - CRS is a projected Equirectangular/SimpleCylindrical CRS in
        meters, centered at (0,0) standard parallel -- this is the
        common case for lunar Map-a-Planet products (verified for our
        WAC crop). Converts corners to lon/lat with the simple
        equirectangular inverse: lon = x/R, lat = y/R (radians).
      - Any other projected CRS -> raises, since we haven't verified
        the correct conversion for it; don't silently produce wrong
        numbers.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs

        # Pixel-corner coordinates in the file's own CRS units
        px_corners = [(0, 0), (w, 0), (w, h), (0, h)]  # UL, UR, LR, LL (col,row)
        proj_corners = [transform * (c, r) for c, r in px_corners]  # (x, y) pairs

        if crs is None:
            raise ValueError(
                f"{tif_path} has no CRS at all (not just non-degrees -- literally unset). "
                "This script can't derive corners without SOME geolocation. "
                "If this is actually a raw, non-georeferenced file (e.g. NAC .IMG opened "
                "via GDAL's fallback), use --imgN-array/--imgN-geometry instead of --imgN-tif."
            )

        if crs.is_geographic:
            lonlat_corners = np.array([[x, y] for x, y in proj_corners], dtype=np.float64)
        else:
            wkt = crs.to_wkt()
            if "Equirectangular" not in wkt and "Simple_Cylindrical" not in wkt:
                raise ValueError(
                    f"{tif_path} uses an unrecognized projected CRS:\n{wkt}\n"
                    "This script only auto-handles geographic (degrees) or lunar "
                    "Equirectangular/SimpleCylindrical (meters) CRS's. Extend "
                    "load_geotiff_source() before using this file, or reproject it "
                    "to plain lat/lon first with gdalwarp."
                )
            R = detect_equirectangular_radius(crs)
            lonlat_corners = np.array([
                [np.degrees(x / R), np.degrees(y / R)] for x, y in proj_corners
            ], dtype=np.float64)

        latlon_corners = lonlat_corners[:, [1, 0]]  # back to (lat, lon) for consistency

    H, H_inv = fit_pixel_to_lonlat_homography(latlon_corners, h, w)
    return arr, latlon_corners, H, H_inv


def crop_by_lonlat_box(arr, H_inv, lon_min, lon_max, lat_min, lat_max):
    box_lonlat = np.array([[lon_min, lat_min], [lon_max, lat_min],
                            [lon_max, lat_max], [lon_min, lat_max]],
                           dtype=np.float64).reshape(-1, 1, 2)
    pixel_pts = cv2.perspectiveTransform(box_lonlat, H_inv).reshape(-1, 2)
    col_min, row_min = pixel_pts.min(axis=0)
    col_max, row_max = pixel_pts.max(axis=0)

    h, w = arr.shape[-2], arr.shape[-1]
    col_min = max(0, int(np.floor(col_min)))
    row_min = max(0, int(np.floor(row_min)))
    col_max = min(w, int(np.ceil(col_max)))
    row_max = min(h, int(np.ceil(row_max)))

    if col_max <= col_min or row_max <= row_min:
        raise ValueError("Computed crop box is empty -- overlap math likely wrong for this image.")

    return arr[row_min:row_max, col_min:col_max], (col_min, row_min, col_max, row_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1-array")
    ap.add_argument("--img1-geometry")
    ap.add_argument("--img1-band", type=int, default=None)
    ap.add_argument("--img1-tif")

    ap.add_argument("--img2-array")
    ap.add_argument("--img2-geometry")
    ap.add_argument("--img2-band", type=int, default=None)
    ap.add_argument("--img2-tif")

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pad-deg", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_one(prefix):
        arr_path = getattr(args, f"{prefix}_array")
        geo_path = getattr(args, f"{prefix}_geometry")
        band = getattr(args, f"{prefix}_band")
        tif_path = getattr(args, f"{prefix}_tif")
        if tif_path:
            return load_geotiff_source(tif_path)
        elif arr_path and geo_path:
            return load_corner_source(arr_path, geo_path, band=band)
        else:
            raise ValueError(
                f"For {prefix}, pass either --{prefix}-tif, or both "
                f"--{prefix}-array and --{prefix}-geometry."
            )

    arr1, corners1, H1, H1_inv = load_one("img1")
    arr2, corners2, H2, H2_inv = load_one("img2")

    print(f"img1: shape={arr1.shape}, lon[{corners1[:,1].min():.4f},{corners1[:,1].max():.4f}] "
          f"lat[{corners1[:,0].min():.4f},{corners1[:,0].max():.4f}]")
    print(f"img2: shape={arr2.shape}, lon[{corners2[:,1].min():.4f},{corners2[:,1].max():.4f}] "
          f"lat[{corners2[:,0].min():.4f},{corners2[:,0].max():.4f}]")

    lon_min = max(corners1[:, 1].min(), corners2[:, 1].min()) - args.pad_deg
    lon_max = min(corners1[:, 1].max(), corners2[:, 1].max()) + args.pad_deg
    lat_min = max(corners1[:, 0].min(), corners2[:, 0].min()) - args.pad_deg
    lat_max = min(corners1[:, 0].max(), corners2[:, 0].max()) + args.pad_deg

    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(
            "No overlap between the two images. Double-check both corner sources are correct."
        )

    print(f"\nOverlap box: lon[{lon_min:.4f},{lon_max:.4f}] lat[{lat_min:.4f},{lat_max:.4f}]")

    crop1, box1 = crop_by_lonlat_box(arr1, H1_inv, lon_min, lon_max, lat_min, lat_max)
    crop2, box2 = crop_by_lonlat_box(arr2, H2_inv, lon_min, lon_max, lat_min, lat_max)

    print(f"img1 crop shape: {crop1.shape}, pixel box: {box1}")
    print(f"img2 crop shape: {crop2.shape}, pixel box: {box2}")

    np.save(out_dir / "img1_crop.npy", crop1)
    np.save(out_dir / "img2_crop.npy", crop2)
    with open(out_dir / "overlap_bounds.json", "w") as f:
        json.dump({
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max,
            "img1_pixel_box": box1, "img2_pixel_box": box2,
        }, f, indent=2)

    print(f"\nSaved: {out_dir / 'img1_crop.npy'}, {out_dir / 'img2_crop.npy'}, "
          f"{out_dir / 'overlap_bounds.json'}")


if __name__ == "__main__":
    main()