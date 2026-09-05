"""
Base Registration Method Interface & Result Dataclass.

Attempts to import from sibling package `basebenchmarking.methods.base`.
If not available in PYTHONPATH, falls back to local definitions to ensure complete autonomy.
"""

import sys
from pathlib import Path

# Add parent directory to sys.path to enable importing basebenchmarking if present
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from basebenchmarking.methods.base import RegistrationResult, RegistrationMethod
except ImportError:
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
        transform: Optional[List[List[float]]] = None
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
            data = asdict(self)
            return self._sanitize_dict(data)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "RegistrationResult":
            valid_keys = cls.__dataclass_fields__.keys()
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered_data)

        @staticmethod
        def _sanitize_dict(obj: Any) -> Any:
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
            pass

        @abstractmethod
        def run(self, reference_image: np.ndarray, target_image: np.ndarray, config: Any, pair_id: str = "") -> RegistrationResult:
            pass

        @abstractmethod
        def get_matches(self) -> Optional[np.ndarray]:
            pass

        @abstractmethod
        def get_transform(self) -> Optional[np.ndarray]:
            pass

        @abstractmethod
        def get_visualization_data(self) -> Optional[Dict[str, Any]]:
            pass

        @abstractmethod
        def is_available(self) -> Tuple[bool, str]:
            pass
