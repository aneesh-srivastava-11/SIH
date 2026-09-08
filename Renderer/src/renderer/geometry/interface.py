from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TerrainTile:
    normals:      np.ndarray  # (H, W, 3) float32, unit-length, +Z up, map-grid frame
    elevation:    np.ndarray  # (H, W)    float32, METRES
    horizon:      np.ndarray  # (H, W, K) float32, max horizon elevation angle in RADIANS,
                              #                    per azimuth bin, bin k centred at 2πk/K
    valid:        np.ndarray  # (H, W)    bool, False = nodata / void-fill / halo
    lat0:         float       # tile origin, degrees
    lon0:         float       # tile origin, degrees
    dlat:         float       # degrees per pixel
    dlon:         float       # degrees per pixel
    gsd_m:        float       # ground sample distance, metres (at tile centre lat)
    K:            int         # number of azimuth bins (16 or 32)
    halo_px:      int         # halo width; strip before emitting pairs

def cast_shadow(tile: TerrainTile, sun_az_rad: float, sun_el_rad: float) -> np.ndarray:
    """(H, W) bool. True = in shadow. O(1) per pixel via horizon-map lookup."""
    # Convert azimuth to nearest bin index
    sun_az_mod = sun_az_rad % (2 * np.pi)
    bin_size = 2 * np.pi / tile.K
    k = int(np.round(sun_az_mod / bin_size)) % tile.K
    
    return sun_el_rad < tile.horizon[:, :, k]
