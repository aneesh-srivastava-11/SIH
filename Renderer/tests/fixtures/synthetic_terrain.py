import numpy as np
from src.renderer.geometry.interface import TerrainTile

def _brute_force_horizon(elevation: np.ndarray, gsd_m: float, K: int, max_dist_m: float = 5000.0) -> np.ndarray:
    """Brute force compute horizon array in radians for synthetic tests."""
    ny, nx = elevation.shape
    horizon = np.full((ny, nx, K), -np.pi/2, dtype=np.float32)
    azimuths_rad = np.linspace(0, 2*np.pi, K, endpoint=False)
    
    steps = np.arange(1.0, max_dist_m, gsd_m)
    yy, xx = np.indices((ny, nx))
    
    for k, az_rad in enumerate(azimuths_rad):
        dir_x, dir_y = np.sin(az_rad), np.cos(az_rad)
        max_angle = np.full((ny, nx), -np.pi/2, dtype=np.float64)
        
        for dist in steps:
            off_x = dist * dir_x / gsd_m
            off_y = dist * dir_y / gsd_m
            
            src_x = np.round(xx + off_x).astype(np.int32)
            src_y = np.round(yy + off_y).astype(np.int32)
            
            valid = (src_x >= 0) & (src_x < nx) & (src_y >= 0) & (src_y < ny)
            
            h_sample = np.where(valid, elevation[np.clip(src_y, 0, ny-1), np.clip(src_x, 0, nx-1)], elevation)
            
            angle = np.arctan2(h_sample - elevation, dist)
            angle = np.where(valid, angle, -np.pi/2)
            max_angle = np.maximum(max_angle, angle)
            
        horizon[:, :, k] = max_angle.astype(np.float32)
        
    return horizon

def flat_plane(ny: int = 100, nx: int = 100) -> TerrainTile:
    h = np.zeros((ny, nx), dtype=np.float32)
    normals = np.zeros((ny, nx, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    K = 16
    horizon = np.zeros((ny, nx, K), dtype=np.float32)
    valid = np.ones((ny, nx), dtype=bool)
    
    return TerrainTile(
        normals=normals, elevation=h, horizon=horizon, valid=valid,
        lat0=0.0, lon0=0.0, dlat=1e-5, dlon=1e-5, gsd_m=1.0, K=K, halo_px=0
    )

def constant_slope(theta_rad: float, phi_rad: float, ny: int = 100, nx: int = 100) -> TerrainTile:
    """Tilted plane. dip theta, aspect phi (azimuth)."""
    # s_x = sin(theta)sin(phi), s_y = sin(theta)cos(phi), s_z = cos(theta)
    n_x = np.sin(theta_rad) * np.sin(phi_rad)
    n_y = np.sin(theta_rad) * np.cos(phi_rad)
    n_z = np.cos(theta_rad)
    
    normals = np.full((ny, nx, 3), [n_x, n_y, n_z], dtype=np.float32)
    
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    h = -(n_x * xx + n_y * yy) / n_z
    
    K = 16
    horizon = np.zeros((ny, nx, K), dtype=np.float32)
    azimuths_rad = np.linspace(0, 2*np.pi, K, endpoint=False)
    for k, az_rad in enumerate(azimuths_rad):
        dh_ddist = -(n_x * np.sin(az_rad) + n_y * np.cos(az_rad)) / n_z
        horizon[:, :, k] = np.arctan(dh_ddist)
        
    valid = np.ones((ny, nx), dtype=bool)
    
    return TerrainTile(
        normals=normals, elevation=h.astype(np.float32), horizon=horizon, valid=valid,
        lat0=0.0, lon0=0.0, dlat=1e-5, dlon=1e-5, gsd_m=1.0, K=K, halo_px=0
    )

def single_cone(h_max: float = 50.0, r_px: float = 20.0, ny: int = 100, nx: int = 100) -> TerrainTile:
    """Axisymmetric cone."""
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    dist = np.sqrt((xx - nx/2)**2 + (yy - ny/2)**2)
    h = np.maximum(0.0, h_max * (1.0 - dist / r_px)).astype(np.float32)
    
    dy, dx = np.gradient(h)
    normals = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= norm
    normals = normals.astype(np.float32)
    
    K = 16
    horizon = _brute_force_horizon(h, 1.0, K, max_dist_m=nx*2)
    valid = np.ones((ny, nx), dtype=bool)
    
    return TerrainTile(
        normals=normals, elevation=h, horizon=horizon, valid=valid,
        lat0=0.0, lon0=0.0, dlat=1e-5, dlon=1e-5, gsd_m=1.0, K=K, halo_px=0
    )

def gaussian_bump(ny: int = 100, nx: int = 100) -> TerrainTile:
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    h = 50.0 * np.exp(-(((xx - nx/2)**2 + (yy - ny/2)**2) / (2 * 10**2)))
    h = h.astype(np.float32)
    
    dy, dx = np.gradient(h)
    normals = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= norm
    normals = normals.astype(np.float32)
    
    K = 16
    horizon = _brute_force_horizon(h, 1.0, K, max_dist_m=nx*2)
    valid = np.ones((ny, nx), dtype=bool)
    
    return TerrainTile(
        normals=normals, elevation=h, horizon=horizon, valid=valid,
        lat0=0.0, lon0=0.0, dlat=1e-5, dlon=1e-5, gsd_m=1.0, K=K, halo_px=0
    )

def checkerboard_nodata(ny: int = 100, nx: int = 100) -> TerrainTile:
    yy, xx = np.indices((ny, nx))
    valid = ((xx // 10) + (yy // 10)) % 2 == 0
    tile = flat_plane(ny, nx)
    return TerrainTile(
        normals=tile.normals, elevation=tile.elevation, horizon=tile.horizon, valid=valid,
        lat0=0.0, lon0=0.0, dlat=1e-5, dlon=1e-5, gsd_m=1.0, K=16, halo_px=0
    )
