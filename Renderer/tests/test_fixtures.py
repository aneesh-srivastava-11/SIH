import pytest
from src.renderer.geometry.interface import cast_shadow
from tests.fixtures.synthetic_terrain import flat_plane, constant_slope, single_cone, gaussian_bump, checkerboard_nodata
import numpy as np

def test_fixtures_import():
    # Acceptance criteria: Fixtures import with Part A absent; pytest collects and runs
    tile = flat_plane()
    assert tile.normals.shape == (100, 100, 3)
    
    tile = constant_slope(np.radians(10), np.radians(45))
    assert tile.elevation.shape == (100, 100)
    
    tile = single_cone()
    assert tile.horizon.shape == (100, 100, 16)
    
    tile = gaussian_bump()
    assert tile.valid.shape == (100, 100)
    
    tile = checkerboard_nodata()
    assert not tile.valid.all()

def test_cast_shadow_stub():
    tile = flat_plane()
    # For a flat plane, the horizon is 0 everywhere.
    # If sun elevation is > 0, shadow should be False.
    shadows = cast_shadow(tile, sun_az_rad=np.radians(90), sun_el_rad=np.radians(10))
    assert not np.any(shadows)
    
    # If sun elevation is < 0, shadow should be True.
    shadows = cast_shadow(tile, sun_az_rad=np.radians(90), sun_el_rad=np.radians(-10))
    assert np.all(shadows)
