"""
SuperPoint + LightGlue Deep Learning Registration Baseline Adapter.
"""

from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
import cv2

from methods.base import RegistrationMethod, RegistrationResult


class SuperPointLightGlueMethod(RegistrationMethod):
    """
    Off-the-shelf SuperPoint + LightGlue Baseline Adapter.
    Uses Kornia (or LightGlue package) for PyTorch/CUDA execution with CPU fallback.
    """

    def __init__(self):
        self._matches: Optional[np.ndarray] = None
        self._transform: Optional[np.ndarray] = None
        self._vis_data: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "SuperPoint + LightGlue"

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        try:
            import torch  # type: ignore

            # Check for kornia or lightglue
            try:
                import kornia  # type: ignore
                import kornia.feature as KF  # type: ignore
                return True, "Available (via Kornia)"
            except ImportError:
                pass

            try:
                from lightglue import LightGlue, SuperPoint  # type: ignore
                return True, "Available (via LightGlue)"
            except ImportError:
                pass

            return False, "Neither 'kornia' nor 'lightglue' is installed (pip install kornia torch)"
        except ImportError:
            return False, "PyTorch is not installed (pip install torch)"
        except Exception as e:
            return False, f"SuperPoint+LightGlue check failed: {e}"

    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        start_time = time.perf_counter()

        avail, reason = self.is_available()
        if not avail:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=0.0,
                error_message=f"SuperPoint+LightGlue unavailable: {reason}",
            )

        import torch  # type: ignore

        # Determine device: CUDA preferred, CPU fallback
        device_str = "cuda" if (config.device.prefer_gpu and torch.cuda.is_available()) else "cpu"
        device = torch.device(device_str)

        ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        tgt_gray = target_image if len(target_image.shape) == 2 else cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        min_matches = config.matching.min_matches
        ransac_thresh = config.ransac.reproj_threshold

        try:
            # Path A: Kornia Implementation
            try:
                import kornia.feature as KF  # type: ignore
                import kornia.geometry as KG  # type: ignore

                t0 = time.perf_counter()

                # Convert images to torch tensors [1, 1, H, W] in [0, 1]
                t_ref = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                t_tgt = torch.from_numpy(tgt_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                extractor = KF.KeypointDetector(KF.SIFTDescriptor(8)).to(device)  # Fallback extractor if SuperPoint in Kornia
                # Use LightGlueMatcher with SuperPoint
                matcher = KF.LightGlueMatcher("superpoint").eval().to(device)

                # Extracts features & matches
                with torch.inference_mode():
                    # High-level matching
                    matches, scores = matcher({"image0": t_ref, "image1": t_tgt})

                infer_time = (time.perf_counter() - t0) * 1000.0

            except Exception:
                # Path B: Standalone LightGlue package
                from lightglue import LightGlue, SuperPoint  # type: ignore
                from lightglue.utils import numpy_to_torch  # type: ignore

                t0 = time.perf_counter()

                extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
                matcher = LightGlue(features="superpoint").eval().to(device)

                t_ref = numpy_to_torch(ref_gray).unsqueeze(0).to(device) / 255.0
                t_tgt = numpy_to_torch(tgt_gray).unsqueeze(0).to(device) / 255.0

                with torch.inference_mode():
                    feats0 = extractor({"image": t_ref})
                    feats1 = extractor({"image": t_tgt})
                    matches01 = matcher({"image0": feats0, "image1": feats1})

                    feats0, feats1, matches01 = [
                        rb[0] for rb in [feats0, feats1, matches01]
                    ]
                    kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
                    matches_idx = matches01["matches"]

                    pts0 = kpts0[matches_idx[:, 0]].cpu().numpy()
                    pts1 = kpts1[matches_idx[:, 1]].cpu().numpy()

                infer_time = (time.perf_counter() - t0) * 1000.0

            num_matches = len(pts0) if "pts0" in locals() else 0

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
                    inference_time_ms=infer_time,
                    device=device_str,
                    error_message=f"SuperPoint+LightGlue produced insufficient matches ({num_matches} < {min_matches})",
                )

            # RANSAC Homography verification
            src_pts = np.float32(pts0).reshape(-1, 1, 2)
            dst_pts = np.float32(pts1).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

            inlier_mask = mask.ravel().tolist() if mask is not None else []
            num_inliers = int(np.sum(inlier_mask)) if mask is not None else 0
            inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0

            self._transform = H
            match_coords = [(pts0[i][0], pts0[i][1], pts1[i][0], pts1[i][1]) for i in range(num_matches)]
            self._matches = np.array(match_coords) if match_coords else None

            self._vis_data = {
                "good_matches": [(i, i) for i in range(num_matches)],
                "inlier_mask": inlier_mask,
            }

            total_time = (time.perf_counter() - start_time) * 1000.0
            success = H is not None and num_inliers >= config.evaluation.success_min_inliers and inlier_ratio >= config.evaluation.success_min_inlier_ratio
            error_msg = None if success else f"Homography failed or insufficient inliers ({num_inliers} < {config.evaluation.success_min_inliers})"

            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=success,
                num_matches=num_matches,
                num_good_matches=num_matches,
                num_inliers=num_inliers,
                inlier_ratio=round(inlier_ratio, 4),
                transform=H.tolist() if H is not None else None,
                runtime_ms=round(total_time, 2),
                inference_time_ms=round(infer_time, 2),
                device=device_str,
                error_message=error_msg,
                matches=match_coords,
                metadata={"extractor": "SuperPoint", "matcher": "LightGlue", "device": device_str},
            )

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=total_time,
                device=device_str,
                error_message=f"SuperPoint+LightGlue execution error: {str(e)}",
            )

    def get_matches(self) -> Optional[np.ndarray]:
        return self._matches

    def get_transform(self) -> Optional[np.ndarray]:
        return self._transform

    def get_visualization_data(self) -> Dict[str, Any]:
        return self._vis_data

