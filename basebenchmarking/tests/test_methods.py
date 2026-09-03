"""
Unit tests for registration baseline methods.
"""

import os
import pytest
import numpy as np

from pipeline.config import BenchmarkConfig
from methods.sift import SIFTMethod
from methods.akaze import AKAZEMethod
from tests.fixtures.generate_fixtures import generate_synthetic_pair


@pytest.fixture
def synthetic_pair_paths(tmp_path):
    r, t, g = generate_synthetic_pair(str(tmp_path), "test_fixture")
    return r, t, g


def test_sift_availability():
    avail, reason = SIFTMethod.is_available()
    assert isinstance(avail, bool)
    assert isinstance(reason, str)


def test_akaze_availability():
    avail, reason = AKAZEMethod.is_available()
    assert isinstance(avail, bool)
    assert isinstance(reason, str)


def test_sift_execution_on_synthetic_fixture(synthetic_pair_paths):
    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    sift = SIFTMethod()
    result = sift.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "SIFT"
    assert result.pair_id == "test_synth"
    assert result.num_keypoints_ref > 0
    assert result.num_keypoints_tgt > 0
    assert result.runtime_ms > 0
    assert result.success is True
    assert result.transform is not None


def test_akaze_execution_on_synthetic_fixture(synthetic_pair_paths):
    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    akaze = AKAZEMethod()
    result = akaze.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "AKAZE"
    assert result.pair_id == "test_synth"
    assert result.num_keypoints_ref > 0
    assert result.num_keypoints_tgt > 0
    assert result.runtime_ms > 0
    assert result.success is True
    assert result.transform is not None
