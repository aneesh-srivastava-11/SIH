"""
Unit tests for DatasetLoader and ground truth matrix parser.
"""

import os
import json
import pytest
import numpy as np
import cv2

from pipeline.config import BenchmarkConfig
from pipeline.dataset import DatasetLoader, ImagePair


def test_discover_pairs_empty_dir(tmp_path):
    config = BenchmarkConfig()
    config.dataset.pairs_dir = str(tmp_path)
    loader = DatasetLoader(config)

    pairs = loader.discover_pairs()
    assert pairs == []


def test_discover_flat_pairs(tmp_path):
    config = BenchmarkConfig()
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    config.dataset.pairs_dir = str(pairs_dir)

    # Create dummy images: reference1.png and target1.png
    dummy_img = np.zeros((10, 10), dtype=np.uint8)
    cv2.imwrite(str(pairs_dir / "reference1.png"), dummy_img)
    cv2.imwrite(str(pairs_dir / "target1.png"), dummy_img)

    loader = DatasetLoader(config)
    pairs = loader.discover_pairs()

    assert len(pairs) == 1
    assert pairs[0].pair_id == "pair_1"
    assert os.path.exists(pairs[0].reference_path)
    assert os.path.exists(pairs[0].target_path)


def test_load_ground_truth(tmp_path):
    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    gt_file = tmp_path / "gt.json"
    gt_matrix = [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]
    gt_file.write_text(json.dumps({"transform": gt_matrix}))

    matrix = loader.load_ground_truth(str(gt_file))
    assert matrix is not None
    assert matrix.shape == (3, 3)
    assert matrix[0, 2] == 10.0
    assert matrix[1, 2] == 20.0


def test_load_ground_truth_affine(tmp_path):
    config = BenchmarkConfig()
    loader = DatasetLoader(config)

    gt_file = tmp_path / "gt_affine.json"
    affine_matrix = [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]]  # 2x3 matrix
    gt_file.write_text(json.dumps({"matrix": affine_matrix}))

    matrix = loader.load_ground_truth(str(gt_file))
    assert matrix is not None
    assert matrix.shape == (3, 3)  # Automatically expanded to 3x3
    assert matrix[2, 2] == 1.0
