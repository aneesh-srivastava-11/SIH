"""
Tests for Training Infrastructure and Loss Functions.
"""

import tempfile
import torch
import pytest

from finetuningiirscrossmodal.training.train_eloftr import DummyLoFTRModule, compute_eloftr_loss
from finetuningiirscrossmodal.training.train_roma import DummyRoMaModule, compute_roma_loss


def test_dummy_eloftr_forward_and_loss():
    model = DummyLoFTRModule(feature_dim=64)
    img0 = torch.randn(2, 1, 128, 128)
    img1 = torch.randn(2, 1, 128, 128)
    H_gt = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)

    batch = {
        "image0": img0,
        "image1": img1,
        "homography": H_gt,
    }

    loss, metrics = compute_eloftr_loss(model, batch, device="cpu")
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss" in metrics


def test_dummy_roma_forward_and_loss():
    model = DummyRoMaModule(feature_dim=64)
    img0 = torch.randn(2, 1, 64, 64)
    img1 = torch.randn(2, 1, 64, 64)
    H_gt = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)

    batch = {
        "image0": img0,
        "image1": img1,
        "homography": H_gt,
    }

    loss, metrics = compute_roma_loss(model, batch, device="cpu")
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert "loss" in metrics
