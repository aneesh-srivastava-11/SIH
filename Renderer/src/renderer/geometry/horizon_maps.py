"""
horizon_maps.py — Part A: surface normals + horizon-map shadow precompute.

Computes, for every DEM pixel and each of K sun-azimuth bins, the maximum
horizon elevation angle visible in that direction. At render time, a pixel
is in shadow for a given sun (azimuth, elevation) iff:

    sun_elevation < horizon_angle[y, x, nearest_azimuth_bin]

This amortizes shadow-casting to an O(1) lookup per pixel per render,
instead of re-marching rays for every new sun angle (context.md §6.4b).
"""

import numpy as np

R_MOON = 1_737_400.0  # metres, D_MOON sphere, eccentricity 0


def ground_spacing(lat_deg: np.ndarray, dlon_deg: float, dlat_deg: float):
    """
    Per-row ground spacing (metres) for an equirectangular DEM tile.

    lat_deg  : 1-D array of pixel-row center latitudes (degrees), len = ny
    dlon_deg : pixel spacing in longitude (degrees)
    dlat_deg : pixel spacing in latitude (degrees)

    Returns (dx, dy): dx is (ny,) and varies with latitude; dy is (ny,) and
    is constant. context.md §3.4/§6.2 — getting this wrong tilts every
    normal and shadow by a silent, latitude-dependent amount.
    """
    dx = dlon_deg * (np.pi / 180.0) * R_MOON * np.cos(np.radians(lat_deg))
    dy = np.full_like(dx, dlat_deg * (np.pi / 180.0) * R_MOON)
    return dx, dy


def surface_normals(h: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    """
    Horn 3x3 surface normals from a height grid (what `gdaldem slope` uses).

    h  : (ny, nx) height in metres
    dx : (ny,) ground spacing in x (metres), one value per row (lat-dependent)
    dy : (ny,) ground spacing in y (metres)

    Returns n : (ny, nx, 3) unit normal vectors, n = (-dh/dx, -dh/dy, 1)/|.|
    """
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    ky = kx.T

    hp = np.pad(h, 1, mode="edge")
    dzdx = np.zeros_like(h, dtype=np.float64)
    dzdy = np.zeros_like(h, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            dzdx += kx[i, j] * hp[i:i + h.shape[0], j:j + h.shape[1]]
            dzdy += ky[i, j] * hp[i:i + h.shape[0], j:j + h.shape[1]]

    dzdx /= (8.0 * dx[:, None])
    dzdy /= (8.0 * dy[:, None])

    nx_, ny_, nz_ = -dzdx, -dzdy, np.ones_like(h, dtype=np.float64)
    norm = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2)
    return np.stack([nx_ / norm, ny_ / norm, nz_ / norm], axis=-1)


def sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """s = (sin A cos e, cos A cos e, sin e) — context.md §6.2."""
    A, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.sin(A) * np.cos(e), np.cos(A) * np.cos(e), np.sin(e)])


def horizon_map(h: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                 k_azimuths: int = 16,
                 max_dist_m: float = 5000.0,
                 n_steps: int = 40):
    """
    Precompute the horizon-elevation-angle map (context.md §6.4b).

    h          : (ny, nx) height in metres
    dx, dy     : ground spacing per row, from ground_spacing()
    k_azimuths : number of azimuth bins K (start at 16; raise to 32 once
                 you've timed it — R12 flags this precompute as a likely
                 bottleneck)
    max_dist_m : how far to march before giving up on a taller horizon
    n_steps    : number of march steps; geometric spacing gives more
                 resolution near the pixel (where nearby terrain matters
                 most) and coarser resolution far away

    Returns:
        horizon_deg  : (ny, nx, k_azimuths) max horizon elevation angle
                       (degrees) seen from each pixel in each direction
        azimuths_deg : (k_azimuths,) bin centers, for indexing at render time
    """
    ny, nx = h.shape
    azimuths_deg = np.linspace(0, 360, k_azimuths, endpoint=False)
    steps = np.geomspace(1.0, max_dist_m, n_steps)

    yy, xx = np.indices((ny, nx))
    horizon_deg = np.zeros((ny, nx, k_azimuths), dtype=np.float32)

    for k, az in enumerate(azimuths_deg):
        az_rad = np.radians(az)
        dir_x, dir_y = np.sin(az_rad), np.cos(az_rad)  # x=east, y=north

        max_angle = np.full((ny, nx), -90.0, dtype=np.float64)

        for dist in steps:
            off_x = dist * dir_x / dx[:, None]   # (ny,1) broadcasts over nx
            off_y = dist * dir_y / dy[:, None]

            src_x = xx + off_x
            src_y = yy + off_y
            valid = (src_x >= 0) & (src_x < nx - 1) & (src_y >= 0) & (src_y < ny - 1)

            x0 = np.clip(src_x.astype(np.int64), 0, nx - 2)
            y0 = np.clip(src_y.astype(np.int64), 0, ny - 2)
            fx, fy = src_x - x0, src_y - y0

            h00, h01 = h[y0, x0], h[y0, x0 + 1]
            h10, h11 = h[y0 + 1, x0], h[y0 + 1, x0 + 1]
            h_sample = (h00 * (1 - fx) * (1 - fy) + h01 * fx * (1 - fy) +
                        h10 * (1 - fx) * fy + h11 * fx * fy)

            angle = np.degrees(np.arctan2(h_sample - h, dist))
            angle = np.where(valid, angle, -90.0)
            max_angle = np.maximum(max_angle, angle)

        horizon_deg[:, :, k] = max_angle

    return horizon_deg, azimuths_deg


def shadow_mask(sun_azimuth_deg: float, sun_elevation_deg: float,
                 horizon_deg: np.ndarray, azimuths_deg: np.ndarray) -> np.ndarray:
    """O(1)-per-pixel shadow lookup at render time. True = in shadow."""
    k = np.argmin(np.abs(((azimuths_deg - sun_azimuth_deg + 180) % 360) - 180))
    return sun_elevation_deg < horizon_deg[:, :, k]


if __name__ == "__main__":
    import time

    # --- Synthetic sanity check: one hill on a flat plain ---------------
    ny, nx = 200, 200
    yy, xx = np.mgrid[0:ny, 0:nx]
    h = 50.0 * np.exp(-(((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 25 ** 2)))

    lat_deg = np.linspace(-13.06, -13.89, ny)  # matches P1's footprint
    dx, dy = ground_spacing(lat_deg, dlon_deg=0.26 / 111320,
                             dlat_deg=0.26 / 111320)
    # (dlon/dlat here approximate 0.26 m pixels in degrees at this latitude —
    #  replace with your DEM's actual pixel spacing in degrees)

    n = surface_normals(h, dx, dy)

    t0 = time.time()
    horizon_deg, az = horizon_map(h, dx, dy, k_azimuths=16,
                                   max_dist_m=300.0, n_steps=30)
    print(f"horizon_map on {ny}x{nx} tile took {time.time() - t0:.2f}s "
          f"(scale this to your real tile size to sanity-check R12)")

    # P1's real geometry from the parsed label:
    sun_az, sun_el = 270.918646, 9.909709
    mask = shadow_mask(sun_az, sun_el, horizon_deg, az)
    print(f"shadowed fraction at sun_az={sun_az}, sun_el={sun_el}: "
          f"{mask.mean():.3f}")

    # Sanity: at 9.9 deg elevation with a 50 m hill, there should be a
    # long shadow on the side away from the sun. Confirm it's non-trivial
    # but not the whole tile.
    assert 0.0 < mask.mean() < 0.9, "shadow fraction looks degenerate — check sun vector/azimuth convention"
    print("sanity check passed")
