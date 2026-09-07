"""
Unit tests for registration baseline methods.
"""

import os
import pytest
import numpy as np

from pipeline.config import BenchmarkConfig
from methods.sift import SIFTMethod
from methods.asift import ASIFTMethod
from methods.akaze import AKAZEMethod
from methods.rift2 import RIFT2Method
from methods.superpoint_lightglue import SuperPointLightGlueMethod
from methods.efficient_loftr import EfficientLoFTRMethod
from methods.arosics_method import AROSICSMethod
from tests.fixtures.generate_fixtures import generate_synthetic_pair


@pytest.fixture
def synthetic_pair_paths(tmp_path):
    r, t, g = generate_synthetic_pair(str(tmp_path), "test_fixture")
    return r, t, g


def test_sift_availability():
    avail, reason = SIFTMethod.is_available()
    assert isinstance(avail, bool)
    assert isinstance(reason, str)


def test_asift_availability():
    avail, reason = ASIFTMethod.is_available()
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


def test_asift_execution_on_synthetic_fixture(synthetic_pair_paths):
    avail, reason = ASIFTMethod.is_available()
    if not avail:
        pytest.skip(f"ASIFT unavailable: {reason}")

    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    asift = ASIFTMethod()
    result = asift.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "ASIFT"
    assert result.pair_id == "test_synth"
    assert result.num_keypoints_ref > 0
    assert result.num_keypoints_tgt > 0
    assert result.runtime_ms > 0


def test_akaze_execution_on_synthetic_fixture(synthetic_pair_paths):
    avail, reason = AKAZEMethod.is_available()
    if not avail:
        pytest.skip(f"AKAZE unavailable: {reason}")

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


def test_rift2_execution_on_synthetic_fixture(synthetic_pair_paths):
    avail, reason = RIFT2Method.is_available()
    if not avail:
        pytest.skip(f"RIFT2 (GNU Octave) unavailable: {reason}")

    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    rift2 = RIFT2Method()
    result = rift2.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "RIFT2"
    assert result.pair_id == "test_synth"
    assert result.runtime_ms > 0


def test_superpoint_lightglue_execution_on_synthetic_fixture(synthetic_pair_paths):
    avail, reason = SuperPointLightGlueMethod.is_available()
    if not avail:
        pytest.skip(f"SuperPoint + LightGlue unavailable: {reason}")

    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    sp_lg = SuperPointLightGlueMethod()
    result = sp_lg.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "SuperPoint + LightGlue"
    assert result.pair_id == "test_synth"
    assert result.runtime_ms > 0


def test_efficient_loftr_execution_on_synthetic_fixture(synthetic_pair_paths):
    avail, reason = EfficientLoFTRMethod.is_available()
    if not avail:
        pytest.skip(f"EfficientLoFTR unavailable: {reason}")

    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    loftr = EfficientLoFTRMethod()
    result = loftr.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "EfficientLoFTR"
    assert result.pair_id == "test_synth"
    assert result.runtime_ms > 0


def test_arosics_execution_on_synthetic_fixture(synthetic_pair_paths):
    avail, reason = AROSICSMethod.is_available()
    if not avail:
        pytest.skip(f"AROSICS unavailable: {reason}")

    ref_path, tgt_path, gt_path = synthetic_pair_paths
    from pipeline.dataset import DatasetLoader

    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    ref_img = loader.load_image(ref_path)
    tgt_img = loader.load_image(tgt_path)

    arosics = AROSICSMethod()
    result = arosics.run(ref_img, tgt_img, config, pair_id="test_synth")

    assert result.method == "AROSICS"
    assert result.pair_id == "test_synth"
    assert result.runtime_ms > 0

