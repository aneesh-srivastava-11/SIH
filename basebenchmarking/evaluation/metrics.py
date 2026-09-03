"""
Evaluation Metrics Calculations for Image Registration Benchmarking.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np


def compute_inlier_ratio(num_inliers: Optional[int], num_matches: Optional[int]) -> Optional[float]:
    """Calculate ratio of inliers to candidate matches."""
    if num_inliers is None or num_matches is None or num_matches == 0:
        return None
    return float(num_inliers / num_matches)


def compute_homography_rmse(
    H_pred: Optional[np.ndarray],
    H_gt: Optional[np.ndarray],
    image_shape: Tuple[int, int] = (512, 512),
    grid_size: int = 20,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute Root Mean Square Error (RMSE overall, RMSE X, RMSE Y) between predicted
    homography H_pred and ground truth homography H_gt over a uniform grid of test points.

    Returns:
        Tuple of (rmse_overall, rmse_x, rmse_y) or (None, None, None) if ground truth unavailable.
    """
    if H_pred is None or H_gt is None:
        return None, None, None

    H_pred = np.array(H_pred, dtype=np.float64)
    H_gt = np.array(H_gt, dtype=np.float64)

    if H_pred.shape != (3, 3) or H_gt.shape != (3, 3):
        return None, None, None

    h, w = image_shape
    xs = np.linspace(0, w - 1, grid_size)
    ys = np.linspace(0, h - 1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)])  # (3, N)

    # Transform points using predicted H
    pred_pts_h = H_pred @ pts
    pred_pts = pred_pts_h[:2] / (pred_pts_h[2:] + 1e-12)

    # Transform points using ground truth H
    gt_pts_h = H_gt @ pts
    gt_pts = gt_pts_h[:2] / (gt_pts_h[2:] + 1e-12)

    # Error vectors
    diff_x = pred_pts[0] - gt_pts[0]
    diff_y = pred_pts[1] - gt_pts[1]

    rmse_x = float(np.sqrt(np.mean(diff_x**2)))
    rmse_y = float(np.sqrt(np.mean(diff_y**2)))
    rmse_overall = float(np.sqrt(np.mean(diff_x**2 + diff_y**2)))

    return rmse_overall, rmse_x, rmse_y


def compute_reprojection_errors(
    H_pred: Optional[np.ndarray],
    H_gt: Optional[np.ndarray],
    image_shape: Tuple[int, int] = (512, 512),
    grid_size: int = 20,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute mean, median, and maximum reprojection errors across grid points.

    Returns:
        Tuple of (mean_error, median_error, max_error)
    """
    if H_pred is None or H_gt is None:
        return None, None, None

    H_pred = np.array(H_pred, dtype=np.float64)
    H_gt = np.array(H_gt, dtype=np.float64)

    if H_pred.shape != (3, 3) or H_gt.shape != (3, 3):
        return None, None, None

    h, w = image_shape
    xs = np.linspace(0, w - 1, grid_size)
    ys = np.linspace(0, h - 1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)])

    pred_pts_h = H_pred @ pts
    pred_pts = pred_pts_h[:2] / (pred_pts_h[2:] + 1e-12)

    gt_pts_h = H_gt @ pts
    gt_pts = gt_pts_h[:2] / (gt_pts_h[2:] + 1e-12)

    distances = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=0))

    mean_err = float(np.mean(distances))
    median_err = float(np.median(distances))
    max_err = float(np.max(distances))

    return mean_err, median_err, max_err


def is_registration_success(
    num_inliers: Optional[int],
    inlier_ratio: Optional[float],
    H_matrix: Optional[List[List[float]]],
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.10,
) -> bool:
    """Determine whether registration run was successful based on threshold criteria."""
    if H_matrix is None:
        return False
    if num_inliers is not None and num_inliers < min_inliers:
        return False
    if inlier_ratio is not None and inlier_ratio < min_inlier_ratio:
        return False
    return True
