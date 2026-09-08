import pytest
import numpy as np
from src.renderer.albedo import photometric_flatten, inpaint_albedo, procedural_albedo, drape
from src.renderer.reflectance import reflectance, sun_vector

def dummy_L_table(alpha_rad):
    return 0.5

def test_at1_round_trip():
    """AT-1: Round-trip: flatten a synthetic I = A·R_LL at geometry 0, recover Â ≈ A."""
    H, W = 100, 100
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    
    A = np.full((H, W), 0.15, dtype=np.float32)
    s0 = sun_vector(np.radians(45), np.radians(30))
    v0 = np.array([0, 0, 1], dtype=np.float32)
    
    I_real = reflectance(normals, s0, v0, A, L_table=dummy_L_table, model="lunar_lambert")
    
    A_hat, valid_mask = photometric_flatten(I_real, normals, s0, v0, L_table=dummy_L_table)
    
    assert np.all(valid_mask)
    np.testing.assert_allclose(A_hat[valid_mask], A[valid_mask], atol=1e-5)

def test_at2_valid_mask():
    """AT-2: albedo_valid is False exactly on the source shadow mask."""
    H, W = 100, 100
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    
    A = np.full((H, W), 0.15, dtype=np.float32)
    s0 = sun_vector(np.radians(45), np.radians(30))
    v0 = np.array([0, 0, 1], dtype=np.float32)
    
    I_real = reflectance(normals, s0, v0, A, L_table=dummy_L_table, model="lunar_lambert")
    
    source_shadow_mask = np.zeros((H, W), dtype=bool)
    source_shadow_mask[10:20, 10:20] = True  # True means shadowed
    
    A_hat, valid_mask = photometric_flatten(I_real, normals, s0, v0, L_table=dummy_L_table, source_shadow_mask=source_shadow_mask)
    
    assert np.array_equal(valid_mask, ~source_shadow_mask)
