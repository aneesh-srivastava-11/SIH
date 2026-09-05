"""
Common Method Interface and Result Schema for Image Registration Benchmarking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
import datetime
import numpy as np


@dataclass
class RegistrationResult:
    """Standardized result dataclass produced by all registration methods."""

    method: str
    pair_id: str
    success: bool
    num_keypoints_ref: Optional[int] = None
    num_keypoints_tgt: Optional[int] = None
    num_matches: Optional[int] = None
    num_good_matches: Optional[int] = None
    num_inliers: Optional[int] = None
    inlier_ratio: Optional[float] = None
    transform: Optional[List[List[float]]] = None  # 3x3 homography matrix as list of lists
    rmse: Optional[float] = None
    rmse_x: Optional[float] = None
    rmse_y: Optional[float] = None
    reprojection_error_mean: Optional[float] = None
    reprojection_error_median: Optional[float] = None
    reprojection_error_max: Optional[float] = None
    runtime_ms: Optional[float] = None
    preprocessing_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    device: str = "cpu"
    error_message: Optional[str] = None
    matches: Optional[List[Tuple[float, float, float, float]]] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to JSON-serializable dictionary."""
        data = asdict(self)
        # Ensure numpy types in metadata or fields are plain Python types
        return self._sanitize_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistrationResult":
        """Reconstruct result object from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    @staticmethod
    def _sanitize_dict(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: RegistrationResult._sanitize_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [RegistrationResult._sanitize_dict(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj


class RegistrationMethod(ABC):
    """Abstract Base Class for all image registration baseline methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable identifier for the method."""
        pass

    @abstractmethod
    def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
        """
        Execute image registration/matching on reference and target images.

        Args:
            reference_image: Grayscale or RGB numpy array
            target_image: Grayscale or RGB numpy array
            config: BenchmarkConfig instance
            pair_id: Identifier string for the image pair being evaluated

        Returns:
            RegistrationResult containing standardized metrics and transform
        """
        pass

    @abstractmethod
    def get_matches(self) -> Optional[np.ndarray]:
        """Return match coordinates array (N, 4) where each row is [x_ref, y_ref, x_tgt, y_tgt]."""
        pass

    @abstractmethod
    def get_transform(self) -> Optional[np.ndarray]:
        """Return estimated 3x3 homography transformation matrix."""
        pass

    @abstractmethod
    def get_visualization_data(self) -> Dict[str, Any]:
        """Return data necessary for visual rendering (keypoints, inlier masks, etc.)."""
        pass

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        """
        Check whether necessary dependencies are installed and system is ready.

        Returns:
            Tuple of (is_available: bool, reason_if_unavailable: str)
        """
        return True, "Available"
