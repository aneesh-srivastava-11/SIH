"""
Unit tests for metrics calculation module.
"""

import pytest
import numpy as np

from evaluation.metrics import (
    compute_inlier_ratio,
    compute_homography_rmse,
    compute_reprojection_errors,
    is_registration_success,
)


def test_compute_inlier_ratio():
    assert compute_inlier_ratio(10, 100) == 0.1
    assert compute_inlier_ratio(50, 50) == 1.0
    assert compute_inlier_ratio(0, 100) == 0.0
    assert compute_inlier_ratio(None, 100) is None
    assert compute_inlier_ratio(10, 0) is None


def test_compute_homography_rmse_identity():
    H_gt = np.eye(3, dtype=np.float64)
    H_pred = np.eye(3, dtype=np.float64)

    rmse_overall, rmse_x, rmse_y = compute_homography_rmse(H_pred, H_gt)
    assert pytest.approx(rmse_overall, abs=1e-5) == 0.0
    assert pytest.approx(rmse_x, abs=1e-5) == 0.0
    assert pytest.approx(rmse_y, abs=1e-5) == 0.0


def test_compute_homography_rmse_known_shift():
    H_gt = np.eye(3, dtype=np.float64)
    H_pred = np.eye(3, dtype=np.float64)
    H_pred[0, 2] = 3.0  # Shift x by 3 pixels
    H_pred[1, 2] = 4.0  # Shift y by 4 pixels

    rmse_overall, rmse_x, rmse_y = compute_homography_rmse(H_pred, H_gt)
    assert pytest.approx(rmse_x, abs=1e-4) == 3.0
    assert pytest.approx(rmse_y, abs=1e-4) == 4.0
    assert pytest.approx(rmse_overall, abs=1e-4) == 5.0  # sqrt(3^2 + 4^2) = 5.0


def test_compute_homography_rmse_null():
    rmse_overall, rmse_x, rmse_y = compute_homography_rmse(None, np.eye(3))
    assert rmse_overall is None
    assert rmse_x is None
    assert rmse_y is None


def test_compute_reprojection_errors():
    H_gt = np.eye(3, dtype=np.float64)
    H_pred = np.eye(3, dtype=np.float64)
    H_pred[0, 2] = 5.0  # Constant offset 5px

    mean_err, median_err, max_err = compute_reprojection_errors(H_pred, H_gt)
    assert pytest.approx(mean_err, abs=1e-4) == 5.0
    assert pytest.approx(median_err, abs=1e-4) == 5.0
    assert pytest.approx(max_err, abs=1e-4) == 5.0


def test_is_registration_success():
    H = np.eye(3).tolist()
    assert is_registration_success(20, 0.5, H, min_inliers=10, min_inlier_ratio=0.1) is True
    assert is_registration_success(5, 0.5, H, min_inliers=10, min_inlier_ratio=0.1) is False
    assert is_registration_success(20, 0.05, H, min_inliers=10, min_inlier_ratio=0.1) is False
    assert is_registration_success(20, 0.5, None, min_inliers=10, min_inlier_ratio=0.1) is False
