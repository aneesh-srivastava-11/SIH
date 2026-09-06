"""
Tests for Matcher Adapters (MatchAnything, FineTunedLoFTR, FineTunedRoMa).
"""

from types import SimpleNamespace
import numpy as np
import pytest

from finetuningiirscrossmodal.methods.matchanything import MatchAnythingMethod
from finetuningiirscrossmodal.methods.finetuned_loftr import FineTunedLoFTRMethod
from finetuningiirscrossmodal.methods.finetuned_roma import FineTunedRoMaMethod


@pytest.fixture
def dummy_config():
    return SimpleNamespace(
        ransac_reproj_threshold=3.0,
        image_width=640,
        image_height=480,
    )


@pytest.fixture
def synthetic_pair():
    ref = np.zeros((480, 640, 3), dtype=np.uint8)
    tgt = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add synthetic feature dots
    cv2_img = ref.copy()
    cv2_img[100:150, 100:150] = 255
    tgt_img = tgt.copy()
    tgt_img[105:155, 105:155] = 255
    return cv2_img, tgt_img


def test_matchanything_instantiation_and_run(synthetic_pair, dummy_config):
    method = MatchAnythingMethod()
    assert method.name == "MatchAnything"

    avail, msg = method.is_available()
    assert avail is True

    ref, tgt = synthetic_pair
    res = method.run(ref, tgt, dummy_config, pair_id="test_matchanything")

    assert res.method == "MatchAnything"
    assert res.pair_id == "test_matchanything"
    assert isinstance(res.success, bool)


def test_finetuned_loftr_instantiation_and_run(synthetic_pair, dummy_config):
    method = FineTunedLoFTRMethod(checkpoint_path="checkpoints/non_existent.pth")
    assert method.name == "FineTuned_EfficientLoFTR"

    avail, msg = method.is_available()
    assert avail is True

    ref, tgt = synthetic_pair
    res = method.run(ref, tgt, dummy_config, pair_id="test_eloftr")

    assert res.method == "FineTuned_EfficientLoFTR"
    assert isinstance(res.success, bool)


def test_finetuned_roma_instantiation_and_run(synthetic_pair, dummy_config):
    method = FineTunedRoMaMethod(checkpoint_path="checkpoints/non_existent.pth")
    assert method.name == "FineTuned_RoMa"

    avail, msg = method.is_available()
    assert avail is True

    ref, tgt = synthetic_pair
    res = method.run(ref, tgt, dummy_config, pair_id="test_roma")

    assert res.method == "FineTuned_RoMa"
    assert isinstance(res.success, bool)
