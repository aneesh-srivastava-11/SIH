import pytest
import numpy as np
from src.renderer.config import load_config, config_hash
from src.renderer.pipeline import render_tile
from tests.fixtures.synthetic_terrain import gaussian_bump

def test_it1_pipeline_execution():
    """IT-1: Full pipeline on gaussian_bump, 8 illuminations -> 8 renders (implies 28 pairs), no NaN/inf."""
    cfg = load_config("configs/render.yaml")
    rng = np.random.default_rng(42)
    tile = gaussian_bump()
    
    renders = render_tile(tile, cfg, rng, n_illuminations=8)
    
    assert len(renders) == 8
    for r in renders:
        img = r["image"]
        assert not np.any(np.isnan(img))
        assert not np.any(np.isinf(img))

def test_it2_determinism():
    """IT-2: Config hash changes ⟹ output changes; config unchanged ⟹ output byte-identical."""
    cfg1 = load_config("configs/render.yaml")
    cfg2 = load_config("configs/render.yaml")
    cfg2.sensor.gain = 2.0 # change config
    
    assert config_hash(cfg1) != config_hash(cfg2)
    
    tile = gaussian_bump()
    
    rng1 = np.random.default_rng(42)
    r1 = render_tile(tile, cfg1, rng1, n_illuminations=1)[0]
    
    rng1_dup = np.random.default_rng(42)
    r1_dup = render_tile(tile, cfg1, rng1_dup, n_illuminations=1)[0]
    
    rng2 = np.random.default_rng(42)
    r2 = render_tile(tile, cfg2, rng2, n_illuminations=1)[0]
    
    np.testing.assert_array_equal(r1["image"], r1_dup["image"])
    assert not np.array_equal(r1["image"], r2["image"])
