"""
RIFT2 (Radiation-variation Insensitive Feature Transform 2) Baseline Adapter.
Uses oct2py to call Octave and execute the MATLAB scripts.
"""

from typing import Optional, Dict, Any, Tuple
import os
import time
import numpy as np
import cv2

from methods.base import RegistrationMethod, RegistrationResult


class RIFT2Method(RegistrationMethod):
    def __init__(self):
        self._matches: Optional[np.ndarray] = None
        self._transform: Optional[np.ndarray] = None
        self._vis_data: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "RIFT2"

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        try:
            import oct2py
            # Try initializing to ensure octave is available
            with oct2py.Oct2Py() as oc:
                oc.eval("1+1")
            return True, "Available via oct2py"
        except Exception as e:
            return False, f"oct2py or Octave not available: {e}"

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
            import oct2py
            
            t0 = time.perf_counter()
            
            # Use Oct2Py to interface with the RIFT2 code
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vendor_path = os.path.join(base_dir, "vendor", "rift2")
            
            oc = oct2py.Oct2Py()
            # Suppress octave warnings
            oc.eval("warning('off', 'all');")
            # Load the image package for imfilter etc
            oc.eval("pkg load image")
            oc.addpath(vendor_path)
            
            def get_rift_features(img):
                # 1. Phase congruency
                m, _, _, _, _, eo, _, _ = oc.phasecong3(img.astype(float), 4, 6, 3, 1.6, 0.75, 1.0, 0.5, 3.0, -1, nout=8)
                
                # Normalize m
                m_min, m_max = np.min(m), np.max(m)
                if m_max > m_min:
                    m = (m - m_min) / (m_max - m_min)
                
                # 2. FAST detection in Python
                m_uint8 = (m * 255).astype(np.uint8)
                fast = cv2.FastFeatureDetector_create(threshold=1, nonmaxSuppression=True)
                kp = fast.detect(m_uint8, None)
                
                # Sort by response and take top keypoints from config
                max_kps = getattr(config.matching, 'max_keypoints', 5000) if hasattr(config, 'matching') else 5000
                kp = sorted(kp, key=lambda x: x.response, reverse=True)[:max_kps]
                kpts_loc = np.array([[k.pt[0], k.pt[1]] for k in kp]).T # 2 x N
                
                if kpts_loc.size == 0:
                    return None, None, None
                
                # 3. Orientation
                # kptsOrientation expects (key, im, is_ori, patch_size)
                kpts_ori = oc.kptsOrientation(kpts_loc, img.astype(float), 1, 96, nout=1)
                
                # 4. Describe
                # FeatureDescribe expects (im, eo, kpts, patch_size, no, nbin)
                des = oc.FeatureDescribe(img.astype(float), eo, kpts_ori, 96, 6, 6, nout=1)
                
                return kpts_ori, des, kp

            # Extract features
            kpts_ref, des_ref, kp1 = get_rift_features(ref_gray)
            kpts_tgt, des_tgt, kp2 = get_rift_features(tgt_gray)
            
            prep_time = (time.perf_counter() - t0) * 1000.0

            if kpts_ref is None or kpts_tgt is None:
                raise ValueError("Could not extract RIFT2 features.")

            # 5. Matching in Python using OpenCV BFMatcher
            # RIFT descriptors are float, shape is (216, N) in MATLAB, so it's (216, N) in numpy
            # Transpose to (N, 216) and convert to float32
            des_ref = des_ref.T.astype(np.float32)
            des_tgt = des_tgt.T.astype(np.float32)

            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            raw_matches = matcher.knnMatch(des_ref, des_tgt, k=2)
            
            # Ratio test
            ratio_thresh = getattr(config.matching, 'ratio_test_threshold', 0.95) if hasattr(config, 'matching') else 0.95
            good_matches = []
            for match in raw_matches:
                if len(match) == 2 and match[0].distance < ratio_thresh * match[1].distance:
                    good_matches.append(match[0])
            
            # If crossCheck is True, or if we use ratio test, we get good_matches
            # RIFT2 demo actually uses MatchThreshold=100 and MaxRatio=1 (so crossCheck essentially)
            # Let's just use strict cross check via BFMatcher directly for consistency with their demo
            matcher_cc = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            good_matches = matcher_cc.match(des_ref, des_tgt)

            m1 = np.float32([kpts_ref[:2, m.queryIdx] for m in good_matches])
            m2 = np.float32([kpts_tgt[:2, m.trainIdx] for m in good_matches])

            num_matches = len(m1)
            
            if num_matches < min_matches:
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
                    error_message=f"RIFT2 returned insufficient matches ({num_matches} < {min_matches})",
                )

            # RANSAC
            H_refined, mask = cv2.findHomography(m1, m2, cv2.RANSAC, ransac_thresh)

            inlier_mask = mask.ravel().tolist() if mask is not None else [1] * num_matches
            num_inliers = int(np.sum(inlier_mask))
            inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0

            self._transform = H_refined
            match_coords = [(m1[i][0], m1[i][1], m2[i][0], m2[i][1]) for i in range(num_matches)]
            self._matches = np.array(match_coords) if match_coords else None

            self._vis_data = {
                "good_matches": [(i, i) for i in range(num_matches)],
                "inlier_mask": inlier_mask,
            }

            total_time = (time.perf_counter() - start_time) * 1000.0
            success = H_refined is not None and num_inliers >= config.evaluation.success_min_inliers
            error_msg = None if success else f"Homography failed or insufficient inliers ({num_inliers} < {config.evaluation.success_min_inliers})"

            # close octave instance to free resources
            oc.exit()

            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=success,
                num_matches=num_matches,
                num_good_matches=num_matches,
                num_inliers=num_inliers,
                inlier_ratio=round(inlier_ratio, 4),
                transform=H_refined.tolist() if H_refined is not None else None,
                runtime_ms=round(total_time, 2),
                preprocessing_time_ms=round(prep_time, 2),
                device="cpu",
                error_message=error_msg,
                matches=match_coords,
                metadata={"descriptor": "PhaseCongruency_MIM_oct2py"},
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
