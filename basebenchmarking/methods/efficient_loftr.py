"""
EfficientLoFTR Deep Learning Registration Baseline Adapter.
"""

from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
import cv2

from methods.base import RegistrationMethod, RegistrationResult


class EfficientLoFTRMethod(RegistrationMethod):
    """
    Off-the-shelf EfficientLoFTR Baseline Adapter.
    Uses Hugging Face `transformers` (or official repository import) for dense correspondence matching.
    """

    def __init__(self):
        self._matches: Optional[np.ndarray] = None
        self._transform: Optional[np.ndarray] = None
        self._vis_data: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "EfficientLoFTR"

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        try:
            import torch  # type: ignore
            try:
                from transformers import EfficientLoFTRForKeypointMatching  # type: ignore
                return True, "Available (via Hugging Face transformers)"
            except ImportError:
                pass

            try:
                from src.loftr import LoFTR  # type: ignore
                return True, "Available (via local EfficientLoFTR repo)"
            except ImportError:
                pass

            return False, "Neither 'transformers' nor local 'EfficientLoFTR' source is installed (pip install transformers torch)"
        except ImportError:
            return False, "PyTorch is not installed (pip install torch)"
        except Exception as e:
            return False, f"EfficientLoFTR check failed: {e}"

    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        start_time = time.perf_counter()

        avail, reason = self.is_available()
        if not avail:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=0.0,
                error_message=f"EfficientLoFTR unavailable: {reason}",
            )

        import torch  # type: ignore

        device_str = "cuda" if (config.device.prefer_gpu and torch.cuda.is_available()) else "cpu"
        device = torch.device(device_str)

        min_matches = config.matching.min_matches
        ransac_thresh = config.ransac.reproj_threshold

        try:
            t0 = time.perf_counter()

            # Path A: Hugging Face transformers
            try:
                from transformers import EfficientLoFTRForKeypointMatching, AutoImageProcessor  # type: ignore

                ref_rgb = reference_image if len(reference_image.shape) == 3 else cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)
                if len(reference_image.shape) == 3:
                    ref_rgb = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2RGB)
                    
                tgt_rgb = target_image if len(target_image.shape) == 3 else cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)
                if len(target_image.shape) == 3:
                    tgt_rgb = cv2.cvtColor(tgt_rgb, cv2.COLOR_BGR2RGB)

                # Try loading pretrained EfficientLoFTR from Hugging Face hub
                model_id = "zju-community/efficientloftr"
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = EfficientLoFTRForKeypointMatching.from_pretrained(model_id).eval().to(device)

                inputs = processor(images=[ref_rgb, tgt_rgb], return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)

                h1, w1 = reference_image.shape[:2]
                h2, w2 = target_image.shape[:2]
                
                res = processor.post_process_keypoint_matching(outputs, [[(h1, w1), (h2, w2)]], threshold=0.2)[0]
                
                pts0 = res["keypoints0"].cpu().numpy()
                pts1 = res["keypoints1"].cpu().numpy()

            except Exception as e:
                # Path B: Local src.loftr repository fallback
                from src.loftr import LoFTR, full_default_cfg  # type: ignore
                
                ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
                tgt_gray = target_image if len(target_image.shape) == 2 else cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
                h1, w1 = ref_gray.shape
                h2, w2 = tgt_gray.shape

                # Target dims divisible by 8
                new_w1, new_h1 = (w1 // 8) * 8, (h1 // 8) * 8
                new_w2, new_h2 = (w2 // 8) * 8, (h2 // 8) * 8

                ref_resized = cv2.resize(ref_gray, (new_w1, new_h1)) if (new_w1 != w1 or new_h1 != h1) else ref_gray
                tgt_resized = cv2.resize(tgt_gray, (new_w2, new_h2)) if (new_w2 != w2 or new_h2 != h2) else tgt_gray

                matcher = LoFTR(config=full_default_cfg).eval().to(device)

                t_ref = torch.from_numpy(ref_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                t_tgt = torch.from_numpy(tgt_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                batch = {"image0": t_ref, "image1": t_tgt}
                with torch.inference_mode():
                    matcher(batch)
                    pts0 = batch["mkpts0_f"].cpu().numpy()
                    pts1 = batch["mkpts1_f"].cpu().numpy()
                    
                num_matches = len(pts0) if pts0 is not None else 0
                # Scale match coordinates back to original image dimensions if resized
                if num_matches > 0 and (new_w1 != w1 or new_h1 != h1 or new_w2 != w2 or new_h2 != h2):
                    scale_x1, scale_y1 = w1 / new_w1, h1 / new_h1
                    scale_x2, scale_y2 = w2 / new_w2, h2 / new_h2

                    pts0[:, 0] *= scale_x1
                    pts0[:, 1] *= scale_y1
                    pts1[:, 0] *= scale_x2
                    pts1[:, 1] *= scale_y2

            infer_time = (time.perf_counter() - t0) * 1000.0
            num_matches = len(pts0) if pts0 is not None else 0

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
                    error_message=f"EfficientLoFTR produced insufficient dense matches ({num_matches} < {min_matches})",
                )

            # RANSAC Homography Verification
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
                metadata={"model": "EfficientLoFTR", "dense_matching": True, "device": device_str},
            )

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=total_time,
                device=device_str,
                error_message=f"EfficientLoFTR execution error: {str(e)}",
            )

    def get_matches(self) -> Optional[np.ndarray]:
        return self._matches

    def get_transform(self) -> Optional[np.ndarray]:
        return self._transform

    def get_visualization_data(self) -> Dict[str, Any]:
        return self._vis_data

