"""
AROSICS Geospatial Co-registration Baseline Adapter.
"""

from typing import Optional, Dict, Any, Tuple
import os
import tempfile
import time
import numpy as np
import cv2

from methods.base import RegistrationMethod, RegistrationResult


class AROSICSMethod(RegistrationMethod):
    """
    Off-the-shelf AROSICS (Automated Satellite Image Co-Registration) Baseline Adapter.
    AROSICS calculates sub-pixel translation shift vectors via FFT phase correlation.
    Note: AROSICS is a co-registration method that estimates spatial shift (dx, dy)
    rather than sparse keypoint matches. Keypoint/match fields are set to None.
    """

    def __init__(self):
        self._matches: Optional[np.ndarray] = None
        self._transform: Optional[np.ndarray] = None
        self._vis_data: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "AROSICS"

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        try:
            import arosics  # type: ignore
            return True, "Available"
        except ImportError:
            return False, "Package 'arosics' is not installed (conda install -c conda-forge arosics)"
        except Exception as e:
            return False, f"AROSICS import failed: {e}"

    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        start_time = time.perf_counter()

        avail, reason = self.is_available()
        if not avail:
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=0.0,
                error_message=f"AROSICS unavailable: {reason}",
            )

        ref_gray = reference_image if len(reference_image.shape) == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        tgt_gray = target_image if len(target_image.shape) == 2 else cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        try:
            import arosics  # type: ignore

            t0 = time.perf_counter()

            # AROSICS expects GeoTIFF files with CRS metadata.
            # Create temporary GeoTIFFs with dummy geotransforms for non-georeference benchmark pairs.
            with tempfile.TemporaryDirectory() as tmpdir:
                ref_path = os.path.join(tmpdir, "ref_geo.tif")
                tgt_path = os.path.join(tmpdir, "tgt_geo.tif")

                self._write_dummy_geotiff(ref_path, ref_gray)
                self._write_dummy_geotiff(tgt_path, tgt_gray)

                # Initialize AROSICS COREG shift calculation
                CR = arosics.COREG(ref_path, tgt_path, wp=(0, 0), ws=(256, 256), max_shift=50)
                CR.calculate_spatial_shifts()

                infer_time = (time.perf_counter() - t0) * 1000.0

                success = bool(CR.success)
                dx = float(CR.x_shift_px) if hasattr(CR, "x_shift_px") else 0.0
                dy = float(CR.y_shift_px) if hasattr(CR, "y_shift_px") else 0.0
                ssim = float(CR.ssim_improved) if hasattr(CR, "ssim_improved") else None

                # Construct 3x3 translation homography matrix
                H = np.eye(3, dtype=np.float64)
                H[0, 2] = dx
                H[1, 2] = dy
                self._transform = H

                self._vis_data = {
                    "shift_x_px": dx,
                    "shift_y_px": dy,
                    "ssim": ssim,
                    "coreg_info": "AROSICS FFT phase correlation translation shift",
                }

                total_time = (time.perf_counter() - start_time) * 1000.0

                return RegistrationResult(
                    method=self.name,
                    pair_id=pair_id,
                    success=success,
                    num_keypoints_ref=None,  # Not applicable to AROSICS
                    num_keypoints_tgt=None,
                    num_matches=None,
                    num_good_matches=None,
                    num_inliers=None,
                    inlier_ratio=None,
                    transform=H.tolist(),
                    runtime_ms=round(total_time, 2),
                    inference_time_ms=round(infer_time, 2),
                    device="cpu",
                    metadata={
                        "shift_x_px": dx,
                        "shift_y_px": dy,
                        "ssim_improved": ssim,
                        "note": "AROSICS outputs sub-pixel spatial shift vector, not sparse keypoints.",
                    },
                )

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                method=self.name,
                pair_id=pair_id,
                success=False,
                runtime_ms=total_time,
                error_message=f"AROSICS execution error: {str(e)}",
            )

    @staticmethod
    def _write_dummy_geotiff(path: str, img: np.ndarray) -> None:
        """Write array as GeoTIFF with dummy EPSG:4326 geotransform if GDAL/rasterio available."""
        try:
            import rasterio  # type: ignore
            from rasterio.transform import from_origin  # type: ignore

            transform = from_origin(0.0, 100.0, 1.0, 1.0)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=img.shape[0],
                width=img.shape[1],
                count=1,
                dtype=img.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(img, 1)
        except ImportError:
            # Fallback to OpenCV if rasterio unavailable
            cv2.imwrite(path, img)

    def get_matches(self) -> Optional[np.ndarray]:
        return None

    def get_transform(self) -> Optional[np.ndarray]:
        return self._transform

    def get_visualization_data(self) -> Dict[str, Any]:
        return self._vis_data

