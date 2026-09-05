"""
RIFT2 (Radiation-variation Insensitive Feature Transform 2) Baseline Adapter.
"""

from typing import Optional, Dict, Any, Tuple
import os
import sys
import time
import numpy as np
import cv2

from methods.base import RegistrationMethod, RegistrationResult


class RIFT2Method(RegistrationMethod):
    """
    Off-the-shelf RIFT2 Baseline Adapter.
    Wraps the official RIFT2 research code (LJY-RS/RIFT2-multimodal-matching-rotation).
    If not installed locally, returns is_available() == False with installation instructions.
    """

    def __init__(self):
        self._matches: Optional[np.ndarray] = None
        self._transform: Optional[np.ndarray] = None
        self._vis_data: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "RIFT2"

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        # Check if RIFT2 module is importable or vendored in vendor/rift2
        try:
            import RIFT2  # type: ignore
            return True, "Available (imported RIFT2)"
        except ImportError:
            pass

        # Check vendor directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vendor_path = os.path.join(base_dir, "vendor", "rift2")
        if os.path.exists(vendor_path):
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)
            try:
                import RIFT2  # type: ignore
                return True, "Available (vendored RIFT2)"
            except ImportError as e:
                return False, f"Vendored RIFT2 found at vendor/rift2/ but import failed: {e}"

        return False, (
            "RIFT2 repository not found. To enable RIFT2:\n"
            "  git clone https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation vendor/rift2\n"
            "or add it to your PYTHONPATH."
        )

    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        start_time = time.perf_counter()

        avail, reason = self.is_available()
        if not avail:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=0.0,
                error_message=f"RIFT2 unavailable: {reason}",
            )

        ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        tgt_gray = target_image if len(target_image.shape) == 2 else cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        min_matches = config.matching.min_matches
        ransac_thresh = config.ransac.reproj_threshold

        try:
            import RIFT2  # type: ignore

            t0 = time.perf_counter()
            # Execute RIFT2 feature extraction & matching
            # Note: RIFT2 API returns matched points (m1, m2) and homography
            if hasattr(RIFT2, "rift2_match"):
                m1, m2, H = RIFT2.rift2_match(ref_gray, tgt_gray)
            elif hasattr(RIFT2, "RIFT2"):
                matcher = RIFT2.RIFT2()
                m1, m2, H = matcher.match(ref_gray, tgt_gray)
            else:
                raise AttributeError("RIFT2 module does not expose expected match interface function")

            prep_time = (time.perf_counter() - t0) * 1000.0

            num_matches = len(m1) if m1 is not None else 0
            if num_matches < min_matches or H is None:
                total_time = (time.perf_counter() - start_time) * 1000.0
                return RegistrationResult(
                    method=self.name,
                    pair_id=pair_id,
                    success=False,
                    num_matches=num_matches,
                    num_inliers=0,
                    inlier_ratio=0.0,
                    runtime_ms=total_time,
                    preprocessing_time_ms=prep_time,
                    error_message=f"RIFT2 returned insufficient matches ({num_matches} < {min_matches}) or null homography",
                )

            # RANSAC verification on matched coordinates
            src_pts = np.float32(m1).reshape(-1, 1, 2)
            dst_pts = np.float32(m2).reshape(-1, 1, 2)
            H_refined, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

            final_H = H_refined if H_refined is not None else H
            inlier_mask = mask.ravel().tolist() if mask is not None else [1] * num_matches
            num_inliers = int(np.sum(inlier_mask))
            inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0

            self._transform = final_H
            match_coords = [(m1[i][0], m1[i][1], m2[i][0], m2[i][1]) for i in range(num_matches)]
            self._matches = np.array(match_coords) if match_coords else None

            self._vis_data = {
                "good_matches": [(i, i) for i in range(num_matches)],
                "inlier_mask": inlier_mask,
            }

            total_time = (time.perf_counter() - start_time) * 1000.0
            success = final_H is not None and num_inliers >= config.evaluation.success_min_inliers

            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=success,
                num_matches=num_matches,
                num_good_matches=num_matches,
                num_inliers=num_inliers,
                inlier_ratio=round(inlier_ratio, 4),
                transform=final_H.tolist() if final_H is not None else None,
                runtime_ms=round(total_time, 2),
                preprocessing_time_ms=round(prep_time, 2),
                device="cpu",
                matches=match_coords,
                metadata={"descriptor": "PhaseCongruency_MIM"},
            )

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=total_time,
                error_message=f"RIFT2 execution error: {str(e)}",
            )

    def get_matches(self) -> Optional[np.ndarray]:
        return self._matches

    def get_transform(self) -> Optional[np.ndarray]:
        return self._transform

    def get_visualization_data(self) -> Dict[str, Any]:
        return self._vis_data

