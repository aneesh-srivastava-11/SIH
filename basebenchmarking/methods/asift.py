"""
ASIFT (Affine-SIFT) Image Registration Baseline Method.
Implements Affine-SIFT simulation using OpenCV's cv2.AffineFeature wrapper around SIFT (Morel & Yu 2009).
"""

from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
import cv2

from methods.base import RegistrationMethod, RegistrationResult


class ASIFTMethod(RegistrationMethod):
    """Off-the-shelf ASIFT (Affine-SIFT) Baseline Adapter."""

    def __init__(self):
        self._matches: Optional[np.ndarray] = None
        self._transform: Optional[np.ndarray] = None
        self._vis_data: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "ASIFT"

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        try:
            if not hasattr(cv2, "AffineFeature") or not hasattr(cv2, "SIFT_create"):
                return False, "OpenCV cv2.AffineFeature or cv2.SIFT_create is unavailable"
            sift = cv2.SIFT_create()
            detector = cv2.AffineFeature.create(sift)
            if detector is not None:
                return True, "Available"
            return False, "OpenCV AffineFeature returned None"
        except AttributeError:
            return False, "OpenCV installation does not support AffineFeature"
        except Exception as e:
            return False, f"ASIFT initialization failed: {e}"

    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        start_time = time.perf_counter()

        ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        tgt_gray = target_image if len(target_image.shape) == 2 else cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        nfeatures = getattr(config.matching, "sift_nfeatures", 0)
        ratio_thresh = config.matching.ratio_test_threshold
        min_matches = config.matching.min_matches
        ransac_thresh = config.ransac.reproj_threshold

        try:
            backend_sift = cv2.SIFT_create(nfeatures=nfeatures)
            detector = cv2.AffineFeature.create(backend_sift)
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                num_keypoints_ref=0,
                num_keypoints_tgt=0,
                num_matches=0,
                num_good_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                runtime_ms=total_time,
                error_message=f"Failed to initialize ASIFT detector: {e}",
            )

        t0 = time.perf_counter()
        kp1, des1 = detector.detectAndCompute(ref_gray, None)
        kp2, des2 = detector.detectAndCompute(tgt_gray, None)
        prep_time = (time.perf_counter() - t0) * 1000.0

        num_kp_ref = len(kp1) if kp1 else 0
        num_kp_tgt = len(kp2) if kp2 else 0

        if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_tgt=num_kp_tgt,
                num_matches=0,
                num_good_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                runtime_ms=total_time,
                preprocessing_time_ms=prep_time,
                error_message=f"Insufficient keypoints detected by ASIFT (ref: {num_kp_ref}, tgt: {num_kp_tgt}, min: {min_matches})",
            )

        # FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        t1 = time.perf_counter()
        matches = flann.knnMatch(des1, des2, k=2)
        infer_time = (time.perf_counter() - t1) * 1000.0

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        num_raw_matches = len(matches)
        num_good = len(good_matches)

        if num_good < min_matches:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_tgt=num_kp_tgt,
                num_matches=num_raw_matches,
                num_good_matches=num_good,
                num_inliers=0,
                inlier_ratio=0.0,
                runtime_ms=total_time,
                preprocessing_time_ms=prep_time,
                inference_time_ms=infer_time,
                error_message=f"Too few good ASIFT matches after ratio test ({num_good} < {min_matches})",
            )

        # RANSAC Homography estimation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        inlier_mask = mask.ravel().tolist() if mask is not None else []
        num_inliers = int(np.sum(inlier_mask)) if mask is not None else 0
        inlier_ratio = float(num_inliers / num_good) if num_good > 0 else 0.0

        self._transform = H
        match_coords = []
        for i, m in enumerate(good_matches):
            pt_ref = kp1[m.queryIdx].pt
            pt_tgt = kp2[m.trainIdx].pt
            match_coords.append((pt_ref[0], pt_ref[1], pt_tgt[0], pt_tgt[1]))
        self._matches = np.array(match_coords) if match_coords else None

        self._vis_data = {
            "kp1": [(kp.pt[0], kp.pt[1]) for kp in kp1],
            "kp2": [(kp.pt[0], kp.pt[1]) for kp in kp2],
            "good_matches": [(m.queryIdx, m.trainIdx) for m in good_matches],
            "inlier_mask": inlier_mask,
        }

        total_time = (time.perf_counter() - start_time) * 1000.0
        success = H is not None and num_inliers >= config.evaluation.success_min_inliers and inlier_ratio >= config.evaluation.success_min_inlier_ratio
        error_msg = None if success else f"Homography failed or insufficient inliers ({num_inliers} < {config.evaluation.success_min_inliers})"

        return RegistrationResult(
            method=self.name,
            pair_id=pair_id,
            success=success,
            num_keypoints_ref=num_kp_ref,
            num_keypoints_tgt=num_kp_tgt,
            num_matches=num_raw_matches,
            num_good_matches=num_good,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            transform=H,
            runtime_ms=total_time,

            preprocessing_time_ms=prep_time,
            inference_time_ms=infer_time,
            error_message=error_msg,

        )

    def get_matches(self) -> Optional[np.ndarray]:
        return self._matches

    def get_transform(self) -> Optional[np.ndarray]:
        return self._transform

    def get_visualization_data(self) -> Dict[str, Any]:
        return self._vis_data

