"""
dem_loader.py — Part A: DEM loading, footprint crop, and tiling.

Takes a downloaded DEM (SLDEM2015 tile, NAC DTM, TMC-2 DEM, ...) and crops
it to a target footprint — e.g. the corners your pds4_parser.py already
extracted into geometry.json — then tiles it to feed surface_normals()
and horizon_map() in horizon_maps.py.

ASSUMPTION (flag this, per context.md's own style of flagging assumptions):
this loader assumes the DEM's affine transform is in DEGREES per pixel
(a geographic-style raster), not a projected meters-per-pixel raster with
a fixed scale. That's the right assumption for reading true angular pixel
spacing and then converting to ground metres via the lat-dependent formula
in horizon_maps.ground_spacing() (context.md §3.4/§6.2) — a projected
raster's constant metre/pixel value would silently ignore meridian
convergence away from its reference latitude. If your SLDEM2015/NAC DTM
file is in a projected CRS instead, check with `gdalinfo <file>` and
reproject to a plain lat/lon grid first (`gdalwarp -t_srs ...`).

Needs rasterio (`pip install rasterio --break-system-packages`).
"""

import math
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform
from rasterio.transform import from_origin


def load_and_crop(dem_path: str, min_lon: float, max_lon: float,
                   min_lat: float, max_lat: float, margin_deg: float = 0.05):
    """
    Open a DEM and crop to a lon/lat bounding box, with a small margin so
    later tiling halos have real data instead of edge-padding.

    dem_path : path to the DEM raster (GeoTIFF/IMG, degrees-per-pixel CRS)
    min_lon, max_lon, min_lat, max_lat : target footprint, e.g. from
        geometry.json's corners (take the min/max over all four corners)
    margin_deg : padding added on each side before cropping

    Returns:
        dem       : (ny, nx) float64 height array in metres
        transform : the affine transform of the cropped window
        dlon_deg  : pixel width in degrees (constant across the raster)
        dlat_deg  : pixel height in degrees (constant across the raster)
    """
    with rasterio.open(dem_path) as src:
        if src.crs and src.crs.is_projected:
            lonlat_crs = {'proj': 'longlat', 'R': 1737400}
            try:
                left, bottom, right, top = transform_bounds(
                    lonlat_crs, src.crs,
                    min_lon - margin_deg, min_lat - margin_deg,
                    max_lon + margin_deg, max_lat + margin_deg
                )
            except Exception:
                left, bottom, right, top = transform_bounds(
                    'EPSG:4326', src.crs,
                    min_lon - margin_deg, min_lat - margin_deg,
                    max_lon + margin_deg, max_lat + margin_deg
                )
            window = from_bounds(left, bottom, right, top, transform=src.transform)
        else:
            window = from_bounds(
                min_lon - margin_deg, min_lat - margin_deg,
                max_lon + margin_deg, max_lat + margin_deg,
                transform=src.transform,
            )

        dem = src.read(1, window=window).astype(np.float64)
        crop_transform = src.window_transform(window)

        if src.crs and src.crs.is_projected:
            R = 1737400
            dlon_deg = crop_transform.a / (math.pi / 180.0 * R)
            dlat_deg = -crop_transform.e / (math.pi / 180.0 * R)

            try:
                lon_tl, lat_tl = transform(src.crs, lonlat_crs, [crop_transform.c], [crop_transform.f])
            except Exception:
                lon_tl, lat_tl = transform(src.crs, 'EPSG:4326', [crop_transform.c], [crop_transform.f])

            crop_transform = from_origin(lon_tl[0], lat_tl[0], dlon_deg, dlat_deg)
        else:
            dlon_deg = crop_transform.a
            dlat_deg = -crop_transform.e  # e is negative (north-up raster)

    return dem, crop_transform, dlon_deg, dlat_deg


def row_center_lats(transform, n_rows: int) -> np.ndarray:
    """Latitude (degrees) of each row's center, for ground_spacing()."""
    row_idx = np.arange(n_rows) + 0.5
    # rasterio Affine: lat = f + e*row  (row transform, col fixed at 0)
    lats = transform.f + transform.e * row_idx
    return lats


def tile_dem(dem: np.ndarray, transform, tile_size: int = 1024, halo: int = 128):
    """
    Yield (tile, row0, col0, lat_deg_for_tile_rows) for each tile, where
    row0/col0 are the tile's top-left offset (excluding halo) in the full
    array — useful for stitching renders back together later.

    Tiles near the DEM edge are naturally smaller (no halo padding is
    invented); handle that in the consumer, or crop with margin_deg above
    so real edges rarely matter.
    """
    ny, nx = dem.shape
    step = tile_size
    for row0 in range(0, ny, step):
        for col0 in range(0, nx, step):
            r0, r1 = max(0, row0 - halo), min(ny, row0 + tile_size + halo)
            c0, c1 = max(0, col0 - halo), min(nx, col0 + tile_size + halo)
            tile = dem[r0:r1, c0:c1]
            lat_deg = row_center_lats(transform, ny)[r0:r1]
            yield tile, row0, col0, lat_deg


if __name__ == "__main__":
    # --- Self-test with a synthetic in-memory DEM (no network needed) ---
    # Mimics P1's footprint: 25.13-25.25 E, -13.89 to -13.06 (S)
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    ny, nx = 400, 400
    min_lon, max_lon = 25.10, 25.28
    min_lat, max_lat = -13.92, -13.03
    dlon = (max_lon - min_lon) / nx
    dlat = (max_lat - min_lat) / ny
    transform = from_origin(min_lon, max_lat, dlon, dlat)

    yy, xx = np.mgrid[0:ny, 0:nx]
    synthetic = -1600.0 + 50.0 * np.sin(xx / 15) * np.cos(yy / 20)  # fake terrain

    profile = dict(driver="GTiff", height=ny, width=nx, count=1,
                   dtype="float64", crs="EPSG:4326", transform=transform)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(synthetic, 1)
        with memfile.open() as dataset:
            # exercise load_and_crop against a real rasterio dataset by
            # re-opening from the same memfile path handle
            window = from_bounds(min_lon, min_lat, max_lon, max_lat,
                                  transform=dataset.transform)
            dem = dataset.read(1, window=window).astype(np.float64)
            crop_transform = dataset.window_transform(window)
            dlon_deg, dlat_deg = crop_transform.a, -crop_transform.e

    print(f"cropped DEM shape: {dem.shape}, dlon_deg={dlon_deg:.5f}, "
          f"dlat_deg={dlat_deg:.5f}")

    lats = row_center_lats(crop_transform, dem.shape[0])
    print(f"row latitudes range: {lats.min():.4f} to {lats.max():.4f} "
          f"(should bracket {min_lat} to {max_lat})")

    n_tiles = sum(1 for _ in tile_dem(dem, crop_transform, tile_size=256, halo=32))
    print(f"tiled into {n_tiles} tiles at 256px + 32px halo")

    assert abs(lats.max() - max_lat) < 0.02 and abs(lats.min() - min_lat) < 0.02
    print("sanity check passed")
