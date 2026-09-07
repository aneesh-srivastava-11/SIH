"""
Dataset loader and validator for flat image pair structures and ground truth homography matrices.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import os
import glob
import json
import numpy as np
import cv2


@dataclass
class ImagePair:
    """Represents a discovered reference-target image pair."""

    pair_id: str
    reference_path: str
    target_path: str
    ground_truth_path: Optional[str] = None


class DatasetLoader:
    """Discovers, validates, and loads image pairs and optional ground truth transforms."""

    def __init__(self, config: Any):
        self.config = config
        self.pairs_dir = os.path.abspath(os.path.join(config.base_dir, config.dataset.pairs_dir))
        self.gt_dir = os.path.abspath(os.path.join(config.base_dir, config.dataset.ground_truth_dir))
        self.supported_formats = config.dataset.supported_formats

    def discover_pairs(self) -> List[ImagePair]:
        """
        Scan `data/cropped/` for pairs.
        Supports both:
        1. Flat naming: `reference1.png` + `target1.png`, `reference_001.jpg` + `target_001.jpg`
        2. Subfolder naming: `pair_001/reference.png` + `pair_001/target.png`
           Also handles cases where a subfolder has exactly two supported files.
        """
        pairs: List[ImagePair] = []

        if not os.path.exists(self.pairs_dir):
            return pairs

        # Strategy 1: Look for subdirectories (e.g. p1/, p3/)
        subdirs = [d for d in os.listdir(self.pairs_dir) if os.path.isdir(os.path.join(self.pairs_dir, d))]
        for sd in subdirs:
            pair_folder = os.path.join(self.pairs_dir, sd)
            ref_path, tgt_path = self._find_pair_in_folder(pair_folder)
            if ref_path and tgt_path:
                gt_path = self._find_ground_truth(sd)
                pairs.append(ImagePair(pair_id=sd, reference_path=ref_path, target_path=tgt_path, ground_truth_path=gt_path))

        # Strategy 2: Flat directory pattern (e.g. reference1.png / target1.png or ref1.png / tgt1.png)
        # If no pairs found in subdirs, try flat directory
        if not pairs:
            pairs = self._discover_flat_pairs()

        pairs.sort(key=lambda p: p.pair_id)
        return pairs

    def _discover_flat_pairs(self) -> List[ImagePair]:
        """Scan flat directory for matching reference and target files."""
        pairs: List[ImagePair] = []
        files = os.listdir(self.pairs_dir)

        # Collect all reference images
        ref_files = {}
        for f in files:
            ext = f.split(".")[-1].lower()
            if ext not in self.supported_formats:
                continue
            lower_name = f.lower()
            if lower_name.startswith("reference") or lower_name.startswith("ref_") or lower_name.startswith("ref"):
                # Extract pair index/suffix
                suffix = lower_name.replace("reference", "").replace("ref_", "").replace("ref", "").split(".")[0]
                ref_files[suffix] = f

        # Match with target images
        for suffix, ref_filename in ref_files.items():
            # Look for target with same suffix
            target_filename = None
            for f in files:
                ext = f.split(".")[-1].lower()
                if ext not in self.supported_formats:
                    continue
                lower_name = f.lower()
                if lower_name.startswith("target") or lower_name.startswith("tgt_") or lower_name.startswith("tgt"):
                    cur_suffix = lower_name.replace("target", "").replace("tgt_", "").replace("tgt", "").split(".")[0]
                    if cur_suffix == suffix:
                        target_filename = f
                        break

            if target_filename:
                pair_id = f"pair_{suffix}" if suffix else "pair_1"
                ref_path = os.path.join(self.pairs_dir, ref_filename)
                tgt_path = os.path.join(self.pairs_dir, target_filename)
                gt_path = self._find_ground_truth(pair_id) or self._find_ground_truth(suffix)

                pairs.append(ImagePair(pair_id=pair_id, reference_path=ref_path, target_path=tgt_path, ground_truth_path=gt_path))

        return pairs

    def _find_pair_in_folder(self, folder_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Find reference and target images inside a pair subfolder."""
        files = os.listdir(folder_path)
        ref_path, tgt_path = None, None
        supported_files = []

        for f in files:
            ext = f.split(".")[-1].lower()
            if ext not in self.supported_formats:
                continue
            lower = f.lower()
            if "sanity_check" in lower:
                continue
            supported_files.append(f)
            lower = f.lower()
            if "ref" in lower or "source" in lower:
                ref_path = os.path.join(folder_path, f)
            elif "target" in lower or "tgt" in lower or "template" in lower:
                tgt_path = os.path.join(folder_path, f)

        # Fallback: if exactly two supported files, just use them sorted
        if not (ref_path and tgt_path) and len(supported_files) == 2:
            supported_files.sort()
            ref_path = os.path.join(folder_path, supported_files[0])
            tgt_path = os.path.join(folder_path, supported_files[1])

        return ref_path, tgt_path

    def _find_ground_truth(self, pair_key: str) -> Optional[str]:
        """Look for matching ground truth JSON file in `data/ground_truth/`."""
        if not os.path.exists(self.gt_dir):
            return None

        possible_names = [f"{pair_key}.json", f"gt_{pair_key}.json", f"pair_{pair_key}.json"]
        for name in possible_names:
            path = os.path.join(self.gt_dir, name)
            if os.path.exists(path):
                return path

        return None

    def load_image(self, image_path: str, convert_to_grayscale: bool = True) -> np.ndarray:
        """
        Load an image from disk (supports PNG, JPG, TIFF, NPY, QUB, IMG).
        Optionally converts to grayscale and normalizes to 8-bit.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        ext = image_path.split(".")[-1].lower()
        if ext == "npy":
            img = np.load(image_path)
        elif ext in ["qub", "img", "tif", "tiff"]:
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    img = src.read(1) # Read the first band
            except ImportError:
                # Fallback to OpenCV if rasterio isn't installed
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                raise ValueError(f"Failed to read raster file {image_path} with rasterio: {e}")
        else:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
        if img is None:
            raise ValueError(f"Failed to read image at path (corrupt or unsupported format): {image_path}")

        # Handle 16-bit or float images -> normalize to 8-bit uint8
        if img.dtype != np.uint8:
            # Exclude NoData sentinels (e.g. dtype min for signed integers like -32768)
            if np.issubdtype(img.dtype, np.signedinteger):
                dtype_min = np.iinfo(img.dtype).min
                valid_mask = img > dtype_min
            elif np.issubdtype(img.dtype, np.floating):
                valid_mask = np.isfinite(img)
            else:
                valid_mask = np.ones_like(img, dtype=bool)

            if np.any(valid_mask):
                valid_pixels = img[valid_mask]
                p_min = np.percentile(valid_pixels, 1)
                p_max = np.percentile(valid_pixels, 99)
                img = np.clip(img, p_min, p_max)
                if p_max > p_min:
                    img = ((img - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if convert_to_grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply max_dimension resizing to prevent OOM on massive satellite images
        if hasattr(self.config, 'preprocessing') and self.config.preprocessing.max_dimension:
            max_dim = self.config.preprocessing.max_dimension
            h, w = img.shape[:2]
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return img

    def load_ground_truth(self, gt_path: Optional[str]) -> Optional[np.ndarray]:
        """
        Load 3x3 homography matrix or 2x3 affine matrix from ground truth JSON.
        Returns 3x3 float64 numpy matrix or None if unavailable/invalid.
        """
        if not gt_path or not os.path.exists(gt_path):
            return None

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            transform_data = data.get("transform") or data.get("homography") or data.get("matrix")
            if transform_data is None:
                return None

            matrix = np.array(transform_data, dtype=np.float64)

            # If 2x3 affine matrix, convert to 3x3 homography matrix
            if matrix.shape == (2, 3):
                homography = np.eye(3, dtype=np.float64)
                homography[:2, :] = matrix
                return homography
            elif matrix.shape == (3, 3):
                return matrix

            return None
        except Exception:
            return None

    def validate_pair(self, pair: ImagePair) -> Tuple[bool, str]:
        """Check if an ImagePair is readable and valid."""
        if not os.path.exists(pair.reference_path):
            return False, f"Reference image missing: {pair.reference_path}"
        if not os.path.exists(pair.target_path):
            return False, f"Target image missing: {pair.target_path}"

        try:
            ref = self.load_image(pair.reference_path)
            tgt = self.load_image(pair.target_path)
            if ref.size == 0 or tgt.size == 0:
                return False, "One or both images are empty (0 pixels)."
        except Exception as e:
            return False, f"Image loading error: {str(e)}"

        return True, "Valid"
