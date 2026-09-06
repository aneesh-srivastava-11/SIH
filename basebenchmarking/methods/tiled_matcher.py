"""
Tiled Matcher Wrapper for high-resolution remote sensing images.
Splits images into a grid of patches, matches them locally, and aggregates keypoints.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
import time

from methods.base import RegistrationMethod, RegistrationResult

class TiledMatcher(RegistrationMethod):
    """
    A wrapper that takes any RegistrationMethod and applies it in a patch-based (tiled) manner.
    This preserves high-resolution details that would otherwise be destroyed by global downsampling.
    """
    
    def __init__(self, base_method: RegistrationMethod, grid_size: Tuple[int, int] = (4, 4)):
        self.base_method = base_method
        self.grid_size = grid_size  # (rows, cols)
        
        self._matches = None
        self._transform = None
        self._vis_data = {}
        
    @property
    def name(self) -> str:
        return f"Tiled {self.base_method.name}"
        
    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        t0 = time.time()
        
        h_ref, w_ref = reference_image.shape[:2]
        h_tgt, w_tgt = target_image.shape[:2]
        
        rows, cols = self.grid_size
        
        all_matches = []
        
        for r in range(rows):
            for c in range(cols):
                # Normalized coordinates for the tile
                u_min, u_max = c / cols, (c + 1) / cols
                v_min, v_max = r / rows, (r + 1) / rows
                
                # Extract ref tile
                ref_x1, ref_x2 = int(u_min * w_ref), int(u_max * w_ref)
                ref_y1, ref_y2 = int(v_min * h_ref), int(v_max * h_ref)
                ref_tile = reference_image[ref_y1:ref_y2, ref_x1:ref_x2]
                
                # Extract tgt tile
                tgt_x1, tgt_x2 = int(u_min * w_tgt), int(u_max * w_tgt)
                tgt_y1, tgt_y2 = int(v_min * h_tgt), int(v_max * h_tgt)
                tgt_tile = target_image[tgt_y1:tgt_y2, tgt_x1:tgt_x2]
                
                # Skip if tiles are too small
                if ref_tile.shape[0] < 10 or ref_tile.shape[1] < 10 or tgt_tile.shape[0] < 10 or tgt_tile.shape[1] < 10:
                    continue
                    
                # Run base method on tiles
                # We need to catch exceptions in case the base method fails on a blank tile
                try:
                    res = self.base_method.run(ref_tile, tgt_tile, config, pair_id=f"{pair_id}_r{r}_c{c}")
                    
                    if res.success and res.matches is not None:
                        tile_matches = res.matches
                        # Shift keypoints to global coordinates
                        for m in tile_matches:
                            x_ref, y_ref, x_tgt, y_tgt = m
                            global_m = (
                                x_ref + ref_x1,
                                y_ref + ref_y1,
                                x_tgt + tgt_x1,
                                y_tgt + tgt_y1
                            )
                            all_matches.append(global_m)
                except Exception as e:
                    print(f"Warning: TiledMatcher tile r{r} c{c} failed: {e}")
                    continue
                    
        # Now we have accumulated matches from all tiles.
        # Run global RANSAC
        if len(all_matches) < config.matching.min_matches:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                error_message=f"Only {len(all_matches)} total matches found across all tiles.",
                runtime_ms=(time.time() - t0) * 1000
            )
            
        src_pts = np.array([[m[0], m[1]] for m in all_matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([[m[2], m[3]] for m in all_matches], dtype=np.float32).reshape(-1, 1, 2)
        
        ransac_method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
        try:
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                ransac_method,
                config.ransac.reproj_threshold,
                maxIters=config.ransac.max_iters,
                confidence=config.ransac.confidence
            )
        except Exception as e:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                error_message=f"Global RANSAC failed: {e}",
                runtime_ms=(time.time() - t0) * 1000
            )
            
        if H is None:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                error_message="Global RANSAC returned None.",
                runtime_ms=(time.time() - t0) * 1000
            )
            
        inliers_mask_list = mask.ravel().tolist() if mask is not None else []
        num_inliers = sum(inliers_mask_list)
        inlier_ratio = num_inliers / len(all_matches) if len(all_matches) > 0 else 0.0
        
        success = (num_inliers >= config.evaluation.success_min_inliers) and (inlier_ratio >= config.evaluation.success_min_inlier_ratio)
        
        self._matches = np.array(all_matches)
        self._transform = H
        self._vis_data = {"inlier_mask": inliers_mask_list}
        
        return RegistrationResult(
            method=self.name,
            pair_id=pair_id,
            success=success,
            num_matches=len(all_matches),
            num_good_matches=len(all_matches),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            transform=H.tolist(),
            runtime_ms=(time.time() - t0) * 1000,
            matches=all_matches
        )
        
    def get_matches(self) -> Optional[np.ndarray]:
        return self._matches
        
    def get_transform(self) -> Optional[np.ndarray]:
        return self._transform
        
    def get_visualization_data(self) -> Dict[str, Any]:
        return self._vis_data
        
    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        return True, "Wrapper available"
