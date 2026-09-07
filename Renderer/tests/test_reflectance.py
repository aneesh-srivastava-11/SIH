import pytest
import numpy as np
from src.renderer.geometry.interface import TerrainTile
from tests.fixtures.synthetic_terrain import flat_plane, constant_slope, single_cone, gaussian_bump
from src.renderer.reflectance import (
    sun_vector, cos_incidence, cos_emission, phase_angle,
    lunar_lambert_L, reflectance
)

@pytest.fixture
def fixtures():
    return [
        flat_plane(),
        constant_slope(np.radians(10), np.radians(45)),
        single_cone(),
        gaussian_bump()
    ]

# A mock L-table function that just returns the input value L directly for testing 
# when we want to force L=0 or L=1. In reality, L_table is an interpolator.
def dummy_L_table(alpha_rad, constant_L=None):
    if constant_L is not None:
        return constant_L
    return 0.5  # default

def test_ut1_lunar_lambert_L0_is_lambert(fixtures):
    """UT-1: model="lunar_lambert", L=0 ≡ model="lambert" to 1e-6, on every fixture."""
    s = sun_vector(az_rad=np.radians(45), el_rad=np.radians(30))
    v = np.array([0, 0, 1], dtype=np.float32)
    albedo = np.full((100, 100), 0.1, dtype=np.float32)
    
    for tile in fixtures:
        R_lambert = reflectance(tile.normals, s, v, albedo, L_table=dummy_L_table, model="lambert")
        R_ll = reflectance(tile.normals, s, v, albedo, L_table=lambda a: dummy_L_table(a, constant_L=0.0), model="lunar_lambert")
        np.testing.assert_allclose(R_ll, R_lambert, atol=1e-6)

def test_ut2_lunar_lambert_L1(fixtures):
    """UT-2: L=1 ≡ 2·µ₀/(µ₀+µ) to 1e-6."""
    s = sun_vector(az_rad=np.radians(45), el_rad=np.radians(30))
    v = np.array([0, 0, 1], dtype=np.float32)
    albedo = np.full((100, 100), 0.1, dtype=np.float32)
    
    for tile in fixtures:
        R_lommel = reflectance(tile.normals, s, v, albedo, L_table=dummy_L_table, model="lommel_seeliger")
        R_ll = reflectance(tile.normals, s, v, albedo, L_table=lambda a: dummy_L_table(a, constant_L=1.0), model="lunar_lambert")
        np.testing.assert_allclose(R_ll, R_lommel, atol=1e-6)

def test_ut3_factor_of_2_check():
    """UT-3: On flat_plane, sun at zenith, nadir view: R == A for all L in [0,1]."""
    tile = flat_plane()
    s = sun_vector(az_rad=0, el_rad=np.radians(90))  # zenith
    v = np.array([0, 0, 1], dtype=np.float32)
    albedo = np.full((100, 100), 0.15, dtype=np.float32)
    
    for L_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        R_ll = reflectance(tile.normals, s, v, albedo, L_table=lambda a: dummy_L_table(a, constant_L=L_val), model="lunar_lambert")
        np.testing.assert_allclose(R_ll, albedo, atol=1e-6)

def test_ut4_monotonicity():
    """UT-4: R monotonically decreases as incidence increases at fixed emission; R = 0 for i ≥ 90°."""
    tile = flat_plane()
    v = np.array([0, 0, 1], dtype=np.float32)
    albedo = np.full((100, 100), 0.1, dtype=np.float32)
    
    prev_R = None
    # Sun elevation from 90 down to 0 corresponds to incidence from 0 to 90
    for el_deg in range(90, -1, -10):
        s = sun_vector(az_rad=0, el_rad=np.radians(el_deg))
        R = reflectance(tile.normals, s, v, albedo, L_table=dummy_L_table, model="lunar_lambert")
        R_val = R[0, 0]
        
        if el_deg == 0:
            # i = 90 deg -> R = 0
            assert np.isclose(R_val, 0.0, atol=1e-6)
        
        if prev_R is not None:
            assert R_val <= prev_R + 1e-6
            
        prev_R = R_val

def test_ut5_linearity_in_albedo():
    """UT-5: R is exactly linear in A (doubling albedo doubles radiance)."""
    tile = gaussian_bump()
    s = sun_vector(az_rad=np.radians(45), el_rad=np.radians(30))
    v = np.array([0, 0, 1], dtype=np.float32)
    albedo1 = np.full((100, 100), 0.1, dtype=np.float32)
    albedo2 = np.full((100, 100), 0.2, dtype=np.float32)
    
    R1 = reflectance(tile.normals, s, v, albedo1, L_table=dummy_L_table, model="lunar_lambert")
    R2 = reflectance(tile.normals, s, v, albedo2, L_table=dummy_L_table, model="lunar_lambert")
    
    np.testing.assert_allclose(R2, 2 * R1, atol=1e-6)

def test_ut6_phase_angle_convention():
    """UT-6: With nadir v, assert phase_angle(s, v) == π/2 − el to 1e-6 across the elevation sweep."""
    v = np.array([0, 0, 1], dtype=np.float32)
    for el_deg in range(0, 91, 10):
        el_rad = np.radians(el_deg)
        s = sun_vector(az_rad=np.radians(33), el_rad=el_rad)
        alpha = phase_angle(s, v)
        assert np.isclose(alpha, np.pi/2 - el_rad, atol=1e-6)

def test_ut7_constant_slope_closed_form():
    """UT-7: constant_slope: reflectance matches the closed-form value."""
    theta = np.radians(10)
    phi = np.radians(45)
    tile = constant_slope(theta, phi)
    
    s = sun_vector(az_rad=np.radians(45), el_rad=np.radians(30))
    v = np.array([0, 0, 1], dtype=np.float32)
    albedo = np.full((100, 100), 0.1, dtype=np.float32)
    
    R = reflectance(tile.normals, s, v, albedo, L_table=lambda a: dummy_L_table(a, constant_L=0.0), model="lambert")
    
    # closed form for lambert: R = A * mu0
    # mu0 = n . s
    n_x = np.sin(theta) * np.sin(phi)
    n_y = np.sin(theta) * np.cos(phi)
    n_z = np.cos(theta)
    
    s_x = np.sin(np.radians(45)) * np.cos(np.radians(30))
    s_y = np.cos(np.radians(45)) * np.cos(np.radians(30))
    s_z = np.sin(np.radians(30))
    
    mu0 = n_x * s_x + n_y * s_y + n_z * s_z
    expected_R = albedo[0,0] * max(0, mu0)
    
    np.testing.assert_allclose(R[0, 0], expected_R, atol=1e-6)
