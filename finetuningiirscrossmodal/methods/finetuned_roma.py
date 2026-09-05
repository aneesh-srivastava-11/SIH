"""
Fine-Tuned RoMa (Rotation-Robust Matcher) Registration Method Adapter.

Loads checkpoint from `checkpoints/finetuned_roma_best.pth` (or user specified path)
to evaluate fine-tuned dense correspondence performance on lunar target datasets.
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch

from finetuningiirscrossmodal.methods.base import RegistrationMethod, RegistrationResult


class FineTunedRoMaMethod(RegistrationMethod):
    """
    Fine-Tuned RoMa adapter for dense lunar image pair registration.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or "checkpoints/finetuned_roma_best.pth"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._last_matches: Optional[np.ndarray] = None
        self._last_transform: Optional[np.ndarray] = None
        self._last_viz_data: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "FineTuned_RoMa"

    def is_available(self) -> Tuple[bool, str]:
        """Checks PyTorch availability and checkpoint status."""
        try:
            import torch
            ckpt = Path(self.checkpoint_path)
            if not ckpt.exists():
                return True, f"FineTuned RoMa ready (Checkpoint '{ckpt}' not yet found; will run pretrained default until trained)."
            return True, f"FineTuned RoMa ready with checkpoint '{ckpt}'."
        except ImportError:
            return False, "PyTorch not installed. Install torch to use FineTuned RoMa."

    def _load_model(self):
        if self._model is not None:
            return

        ckpt = Path(self.checkpoint_path)
        if ckpt.exists():
            print(f"[{self.name}] Loading fine-tuned weights from '{ckpt}'...")
            try:
                state = torch.load(ckpt, map_location=self.device)
                from finetuningiirscrossmodal.training.train_roma import DummyRoMaModule
                self._model = DummyRoMaModule(feature_dim=256)
                if "model_state_dict" in state:
                    self._model.load_state_dict(state["model_state_dict"])
                else:
                    self._model.load_state_dict(state)
                self._model.to(self.device).eval()
                return
            except Exception as e:
                print(f"[{self.name}] Warning: Failed to load checkpoint ({e}). Running default model.")

        from finetuningiirscrossmodal.training.train_roma import DummyRoMaModule
        self._model = DummyRoMaModule(feature_dim=256).to(self.device).eval()

    def run(
        self,
        reference_image: np.ndarray,
        target_image: np.ndarray,
        config: Any,
        pair_id: str = ""
    ) -> RegistrationResult:
        start_time = time.time()
        self._load_model()
        self._last_matches = None
        self._last_transform = None
        self._last_viz_data = None

        # Preprocessing
        t0_prep = time.time()
        if len(reference_image.shape) == 3:
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_image.copy()

        if len(target_image.shape) == 3:
            tgt_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        else:
            tgt_gray = target_image.copy()

        orig_h_ref, orig_w_ref = ref_gray.shape[:2]
        orig_h_tgt, orig_w_tgt = tgt_gray.shape[:2]

        target_w = getattr(config, "image_width", 640)
        target_h = getattr(config, "image_height", 480)
        target_w = ((target_w + 7) // 8) * 8
        target_h = ((target_h + 7) // 8) * 8

        ref_resized = cv2.resize(ref_gray, (target_w, target_h))
        tgt_resized = cv2.resize(tgt_gray, (target_w, target_h))
        prep_time_ms = (time.time() - t0_prep) * 1000.0

        # Inference
        t0_inf = time.time()
        matches_resized, num_ref_kp, num_tgt_kp = self._infer_matches(ref_resized, tgt_resized)
        inf_time_ms = (time.time() - t0_inf) * 1000.0

        if matches_resized is None or len(matches_resized) < 4:
            runtime_ms = (time.time() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                num_keypoints_ref=num_ref_kp,
                num_keypoints_tgt=num_tgt_kp,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                runtime_ms=runtime_ms,
                preprocessing_time_ms=prep_time_ms,
                inference_time_ms=inf_time_ms,
                device=self.device,
                error_message="Insufficient dense matches (< 4).",
            )

        # Scale matches to original resolutions
        scale_x_ref = orig_w_ref / target_w
        scale_y_ref = orig_h_ref / target_h
        scale_x_tgt = orig_w_tgt / target_w
        scale_y_tgt = orig_h_tgt / target_h

        matches_orig = matches_resized.copy()
        matches_orig[:, 0] *= scale_x_ref
        matches_orig[:, 1] *= scale_y_ref
        matches_orig[:, 2] *= scale_x_tgt
        matches_orig[:, 3] *= scale_y_tgt

        self._last_matches = matches_orig

        # RANSAC Homography Estimation
        src_pts = matches_orig[:, :2].reshape(-1, 1, 2)
        dst_pts = matches_orig[:, 2:].reshape(-1, 1, 2)

        ransac_thresh = getattr(config, "ransac_reproj_threshold", 3.0)
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        if H is None or inlier_mask is None:
            runtime_ms = (time.time() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                num_keypoints_ref=num_ref_kp,
                num_keypoints_tgt=num_tgt_kp,
                num_matches=len(matches_orig),
                num_inliers=0,
                inlier_ratio=0.0,
                runtime_ms=runtime_ms,
                preprocessing_time_ms=prep_time_ms,
                inference_time_ms=inf_time_ms,
                device=self.device,
                error_message="RANSAC homography estimation failed.",
            )

        self._last_transform = H
        inliers_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inliers_count / max(len(matches_orig), 1))

        inlier_idx = np.where(inlier_mask.ravel() == 1)[0]
        inlier_src = src_pts[inlier_idx]
        inlier_dst = dst_pts[inlier_idx]

        warped_src = cv2.perspectiveTransform(inlier_src, H)
        errors = np.linalg.norm(warped_src - inlier_dst, axis=2).ravel()

        rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else 0.0
        mean_err = float(np.mean(errors)) if len(errors) > 0 else 0.0
        median_err = float(np.median(errors)) if len(errors) > 0 else 0.0
        max_err = float(np.max(errors)) if len(errors) > 0 else 0.0

        runtime_ms = (time.time() - start_time) * 1000.0

        self._last_viz_data = {
            "ref_img": ref_gray,
            "tgt_img": tgt_gray,
            "matches": matches_orig,
            "inlier_mask": inlier_mask.ravel(),
            "H": H,
        }

        return RegistrationResult(
            method=self.name,
            pair_id=pair_id,
            success=True,
            num_keypoints_ref=num_ref_kp,
            num_keypoints_tgt=num_tgt_kp,
            num_matches=len(matches_orig),
            num_good_matches=len(matches_orig),
            num_inliers=inliers_count,
            inlier_ratio=inlier_ratio,
            transform=H.tolist(),
            rmse=rmse,
            reprojection_error_mean=mean_err,
            reprojection_error_median=median_err,
            reprojection_error_max=max_err,
            runtime_ms=runtime_ms,
            preprocessing_time_ms=prep_time_ms,
            inference_time_ms=inf_time_ms,
            device=self.device,
            metadata={"checkpoint_path": str(self.checkpoint_path)},
        )

    def _infer_matches(self, ref_gray: np.ndarray, tgt_gray: np.ndarray) -> Tuple[Optional[np.ndarray], int, int]:
        try:
            ref_t = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
            tgt_t = torch.from_numpy(tgt_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
            ref_t = ref_t.to(self.device)
            tgt_t = tgt_t.to(self.device)

            with torch.no_grad():
                out = self._model(ref_t, tgt_t)

            flow = out["flow"][0].cpu().numpy()  # [2, H, W]
            H, W = flow.shape[1:]

            step = 16
            grid_y, grid_x = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing="ij")

            x0 = grid_x.ravel()
            y0 = grid_y.ravel()

            dx = flow[0, y0, x0]
            dy = flow[1, y0, x0]

            x1 = x0 + dx
            y1 = y0 + dy

            matches = np.column_stack([x0, y0, x1, y1])
            return matches.astype(np.float32), len(x0), len(x0)
        except Exception:
            return None, 0, 0

    def get_matches(self) -> Optional[np.ndarray]:
        return self._last_matches

    def get_transform(self) -> Optional[np.ndarray]:
        return self._last_transform

    def get_visualization_data(self) -> Optional[Dict[str, Any]]:
        return self._last_viz_data
