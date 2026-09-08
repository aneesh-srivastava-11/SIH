import pytest
import numpy as np
import cv2
import os
import json
from src.renderer.pairing import Similarity, make_pair, PairSample, save_pairs
from tests.fixtures.synthetic_terrain import flat_plane
from src.renderer.config import load_config

@pytest.fixture
def dummy_cfg():
    return load_config("configs/render.yaml")

def test_pt1_identity_unshadowed(dummy_cfg):
    """PT-1: With warp=identity and unshadowed, match_mask is all-True inside halo-stripped region."""
    tile = flat_plane()
    H, W = tile.elevation.shape
    render_a = {
        "image": np.ones((H, W), dtype=np.uint8) * 100,
        "shadow_mask": np.zeros((H, W), dtype=bool),
        "params": {"sun_az_rad": 0.0, "sun_el_rad": 1.0}
    }
    render_b = {
        "image": np.ones((H, W), dtype=np.uint8) * 200,
        "shadow_mask": np.zeros((H, W), dtype=bool),
        "params": {"sun_az_rad": 1.0, "sun_el_rad": 1.0}
    }
    
    object.__setattr__(tile, 'halo_px', 10)
    
    warp = Similarity(scale=1.0, rotation_rad=0.0)
    albedo_valid = np.ones((H, W), dtype=bool)
    
    pair = make_pair(tile, render_a, render_b, warp, dummy_cfg, albedo_valid)
    
    expected_valid = np.zeros((H, W), dtype=bool)
    expected_valid[10:-10, 10:-10] = True
    
    np.testing.assert_array_equal(pair.match_mask, expected_valid)

def test_pt2_mapping_validity(dummy_cfg):
    """PT-2: Every match_mask==True pixel maps under transform to an in-bounds, valid, unshadowed pixel in B."""
    tile = flat_plane()
    H, W = tile.elevation.shape
    
    shadow_b = np.zeros((H, W), dtype=bool)
    shadow_b[20:40, 20:40] = True
    
    render_a = {
        "image": np.random.randint(0, 255, (H, W), dtype=np.uint8),
        "shadow_mask": np.zeros((H, W), dtype=bool),
        "params": {"sun_az_rad": 0.0, "sun_el_rad": 1.0}
    }
    render_b = {
        "image": np.random.randint(0, 255, (H, W), dtype=np.uint8),
        "shadow_mask": shadow_b,
        "params": {"sun_az_rad": 1.0, "sun_el_rad": 1.0}
    }
    
    warp = Similarity(scale=1.5, rotation_rad=np.radians(30), tx=5.0, ty=-5.0)
    albedo_valid = np.ones((H, W), dtype=bool)
    
    pair = make_pair(tile, render_a, render_b, warp, dummy_cfg, albedo_valid)
    
    ys, xs = np.where(pair.match_mask)
    if len(xs) > 10000:
        idx = np.random.choice(len(xs), 10000, replace=False)
        xs = xs[idx]
        ys = ys[idx]
        
    pts_a = np.stack([xs, ys, np.ones_like(xs)], axis=1) # (N, 3)
    
    # Map from image A to warped image B
    pts_w = pts_a @ pair.transform[:2, :].T
    x_w = pts_w[:, 0]
    y_w = pts_w[:, 1]
    
    # Assert all are perfectly in bounds
    assert np.all(x_w >= 0)
    assert np.all(x_w <= W - 1)
    assert np.all(y_w >= 0)
    assert np.all(y_w <= H - 1)
    
    # Assert they are unshadowed in B. Since they map to B, and B was warped, 
    # the unwarped B at (xs, ys) must be unshadowed.
    assert not np.any(shadow_b[ys, xs])

def test_pt3_determinism(dummy_cfg):
    """PT-3: Same inputs -> byte identical output."""
    tile = flat_plane()
    H, W = tile.elevation.shape
    render_a = {
        "image": np.ones((H, W), dtype=np.uint8) * 100,
        "shadow_mask": np.zeros((H, W), dtype=bool),
        "params": {"sun_az_rad": 0.0, "sun_el_rad": 1.0}
    }
    render_b = {
        "image": np.ones((H, W), dtype=np.uint8) * 200,
        "shadow_mask": np.zeros((H, W), dtype=bool),
        "params": {"sun_az_rad": 1.0, "sun_el_rad": 1.0}
    }
    
    warp = Similarity(scale=1.2, rotation_rad=0.5)
    albedo_valid = np.ones((H, W), dtype=bool)
    
    pair1 = make_pair(tile, render_a, render_b, warp, dummy_cfg, albedo_valid)
    pair2 = make_pair(tile, render_a, render_b, warp, dummy_cfg, albedo_valid)
    
    np.testing.assert_array_equal(pair1.image_b, pair2.image_b)
    np.testing.assert_array_equal(pair1.match_mask, pair2.match_mask)
    np.testing.assert_array_equal(pair1.transform, pair2.transform)
