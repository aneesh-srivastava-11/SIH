"""
render_test.py — Part A: composite a validation render from your own
normals + horizon-map shadow mask.

This uses plain Lambertian shading (A * n.s), NOT the Lunar-Lambertian
model — that belongs to Part B (context.md §6.3). The point here is
narrower: confirm your geometry (normals + shadows) is correct before
anyone layers reflectance/albedo/sensor noise on top of it. Constant
albedo is deliberate — it isolates whether the *shape* is right.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.renderer.geometry.horizon_maps import surface_normals, sun_vector, horizon_map, shadow_mask


def render_shading_and_shadow(h: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                               sun_azimuth_deg: float, sun_elevation_deg: float,
                               k_azimuths: int = 16, max_dist_m: float = 5000.0,
                               n_steps: int = 40, albedo: float = 0.12):
    """
    Full pipeline for one test render: normals -> horizon map -> shadow
    mask -> composite. Returns the rendered image plus the intermediate
    normals/shadow mask, since you'll want to inspect those separately
    when comparing against the real OHRC crop.

    albedo : constant reflectance, ~0.10-0.14 is a reasonable mare/highland
             stand-in (context.md doesn't give you a real number here --
             flagged as a placeholder, replace with Part B's albedo once
             it exists)
    """
    n = surface_normals(h, dx, dy)
    s = sun_vector(sun_azimuth_deg, sun_elevation_deg)

    shading = np.clip(n @ s, 0.0, None)  # max(0, n.s), context.md §6.3 form (1)

    horizon_deg, azimuths_deg = horizon_map(h, dx, dy, k_azimuths=k_azimuths,
                                             max_dist_m=max_dist_m, n_steps=n_steps)
    shadow = shadow_mask(sun_azimuth_deg, sun_elevation_deg, horizon_deg, azimuths_deg)

    image = albedo * shading
    image[shadow] = 0.0  # hard-edged cast shadows, no atmosphere to scatter light in

    return image, n, shadow


def to_8bit(image: np.ndarray) -> np.ndarray:
    """Normalise a [0, max] float render to 8-bit for a side-by-side
    comparison against the real OHRC crop (which is already uint8)."""
    img = image / max(image.max(), 1e-9)
    return (img * 255).astype(np.uint8)


if __name__ == "__main__":
    from src.renderer.geometry.dem_loader import load_and_crop, row_center_lats
    from src.renderer.geometry.horizon_maps import ground_spacing

    # --- Self-test on a synthetic DEM, using P1's real sun geometry -----
    ny, nx = 256, 256
    yy, xx = np.mgrid[0:ny, 0:nx]
    h = -1600.0 + 40.0 * np.exp(-(((xx - 130) ** 2 + (yy - 100) ** 2) / (2 * 30 ** 2)))

    lat_deg = np.linspace(-13.06, -13.89, ny)
    dlon_deg = dlat_deg = 0.26 / 111320  # placeholder pixel spacing in degrees
    dx, dy = ground_spacing(lat_deg, dlon_deg, dlat_deg)

    # P1's real geometry from your parsed label:
    sun_az, sun_el = 270.918646, 9.909709

    image, n, shadow = render_shading_and_shadow(
        h, dx, dy, sun_az, sun_el, k_azimuths=16, max_dist_m=300.0, n_steps=30
    )

    print(f"render shape: {image.shape}, max value: {image.max():.4f}")
    print(f"shadowed fraction: {shadow.mean():.3f}")
    print(f"mean normal z-component (should be near 1 on flat ground, "
          f"lower on slopes): {n[..., 2].mean():.4f}")

    img8 = to_8bit(image)
    np.save("test_render_p1_geometry.npy", img8)
    print("saved 8-bit test render to test_render_p1_geometry.npy — "
          "diff this against your real cropped OHRC array next")

    assert 0.0 < shadow.mean() < 0.9
    assert image.max() > 0.0
    print("sanity check passed")
