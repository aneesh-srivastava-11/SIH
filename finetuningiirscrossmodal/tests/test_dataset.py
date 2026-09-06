"""
Tests for Rendered Pairs Dataset and DataLoader utilities.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import cv2
import pytest

from finetuningiirscrossmodal.training.dataset import RenderedPairsDataset, get_dataloader


def test_empty_dataset_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = RenderedPairsDataset(data_dir=tmpdir, img_size=(480, 640))
        assert len(dataset) == 0


def test_dataset_discovery_and_resizing():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy synthetic reference and target images
        ref_img = np.ones((100, 100), dtype=np.uint8) * 128
        tgt_img = np.ones((100, 100), dtype=np.uint8) * 200

        cv2.imwrite(str(tmp_path / "reference0.png"), ref_img)
        cv2.imwrite(str(tmp_path / "target0.png"), tgt_img)

        # Create GT homography JSON
        homography_data = {
            "homography": [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 3.0],
                [0.0, 0.0, 1.0]
            ]
        }
        with open(tmp_path / "pair_0.json", "w") as f:
            json.dump(homography_data, f)

        # Initialize dataset (asking for 481, 641 -> should auto adjust to div by 8)
        dataset = RenderedPairsDataset(data_dir=str(tmp_path), img_size=(481, 641))
        assert len(dataset) == 1

        # Check dimension rounding (481 -> 488, 641 -> 648)
        item = dataset[0]
        assert item["image0"].shape == (1, 488, 648)
        assert item["image1"].shape == (1, 488, 648)
        assert item["homography"].shape == (3, 3)


def test_dataloader_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for i in range(3):
            cv2.imwrite(str(tmp_path / f"reference{i}.png"), np.zeros((100, 100), dtype=np.uint8))
            cv2.imwrite(str(tmp_path / f"target{i}.png"), np.zeros((100, 100), dtype=np.uint8))
            with open(tmp_path / f"pair_{i}.json", "w") as f:
                json.dump({"homography": np.eye(3).tolist()}, f)

        loader = get_dataloader(data_dir=str(tmp_path), batch_size=2, split="all")
        assert len(loader.dataset) == 3
