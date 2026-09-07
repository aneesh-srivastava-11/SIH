"""
Configuration Manager for Image Registration Benchmarking Pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os
import yaml


@dataclass
class DatasetConfig:
    pairs_dir: str = "../data/cropped"
    ground_truth_dir: str = "../data/ground_truth"
    raw_dir: str = "../data/raw"
    supported_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "tif", "tiff", "npy", "qub", "img"])
    naming_pattern: str = "reference{n}"


@dataclass
class OutputConfig:
    results_dir: str = "results"
    raw_dir: str = "results/raw"
    aggregated_dir: str = "results/aggregated"
    visualizations_dir: str = "results/visualizations"


@dataclass
class PreprocessingConfig:
    convert_to_grayscale: bool = True
    max_dimension: Optional[int] = None
    normalize_8bit: bool = True


@dataclass
class MatchingConfig:
    ratio_test_threshold: float = 0.75
    min_matches: int = 4
    flann_trees: int = 5
    flann_checks: int = 50
    max_keypoints: int = 2048
    confidence_threshold: float = 0.2


@dataclass
class RansacConfig:
    method: str = "RANSAC"
    reproj_threshold: float = 5.0
    max_iters: int = 5000
    confidence: float = 0.999


@dataclass
class EvaluationConfig:
    success_min_inliers: int = 10
    success_min_inlier_ratio: float = 0.10


@dataclass
class VisualizationConfig:
    generate: bool = True
    match_visualization: bool = True
    overlay_visualization: bool = True
    max_matches_drawn: int = 200


@dataclass
class DeviceConfig:
    prefer_gpu: bool = True
    device: str = "cuda"
    fallback_to_cpu: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"


@dataclass
class TilingConfig:
    enabled_for_dl: bool = True
    grid_size: Tuple[int, int] = (3, 3)
    min_dimension_threshold: int = 2000


# Centralized default method list (avoid duplication)
DEFAULT_ENABLED_METHODS = ["sift", "asift", "akaze", "rift2", "superpoint_lightglue", "efficient_loftr", "arosics"]


@dataclass
class BenchmarkConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    enabled_methods: List[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_METHODS))
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    base_dir: str = ""

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "BenchmarkConfig":
        """Load configuration from YAML file with .env and environment variable overrides."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load .env file if it exists
        env_file = os.path.join(base_dir, ".env")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

        # Find default configuration file if none provided
        if not config_path:
            config_path = os.path.join(base_dir, "configs", "default.yaml")
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))

        raw_cfg: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}

        # Parse subsections
        dataset_cfg = raw_cfg.get("dataset", {})
        output_cfg = raw_cfg.get("output", {})
        methods_cfg = raw_cfg.get("methods", {})
        prep_cfg = raw_cfg.get("preprocessing", {})
        match_cfg = raw_cfg.get("matching", {})
        ransac_cfg = raw_cfg.get("ransac", {})
        eval_cfg = raw_cfg.get("evaluation", {})
        vis_cfg = raw_cfg.get("visualization", {})
        dev_cfg = raw_cfg.get("device", {})
        log_cfg = raw_cfg.get("logging", {})
        tiling_cfg = raw_cfg.get("tiling", {})

        # Environment variable overrides
        if os.environ.get("BENCHMARK_DATA_DIR"):
            dataset_cfg["pairs_dir"] = os.environ["BENCHMARK_DATA_DIR"]
        if os.environ.get("BENCHMARK_GT_DIR"):
            dataset_cfg["ground_truth_dir"] = os.environ["BENCHMARK_GT_DIR"]
        if os.environ.get("BENCHMARK_OUTPUT_DIR"):
            output_cfg["results_dir"] = os.environ["BENCHMARK_OUTPUT_DIR"]

        # Sibling directory resolution fallback if paths don't exist under base_dir
        for path_key, default_val in [("pairs_dir", "../data/cropped"), ("ground_truth_dir", "../data/ground_truth")]:
            rel_path = dataset_cfg.get(path_key, default_val)
            abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
            sibling_path = os.path.abspath(os.path.join(base_dir, "..", rel_path))
            if not os.path.exists(abs_path) and os.path.exists(sibling_path):
                dataset_cfg[path_key] = sibling_path
            else:
                dataset_cfg[path_key] = abs_path

        # Handle grid_size tuple from YAML list
        if "grid_size" in tiling_cfg and isinstance(tiling_cfg["grid_size"], list):
            tiling_cfg["grid_size"] = tuple(tiling_cfg["grid_size"])

        config = cls(
            dataset=DatasetConfig(**dataset_cfg),
            output=OutputConfig(**output_cfg),
            enabled_methods=methods_cfg.get("enabled", list(DEFAULT_ENABLED_METHODS)),
            preprocessing=PreprocessingConfig(**prep_cfg),
            matching=MatchingConfig(**match_cfg),
            ransac=RansacConfig(**ransac_cfg),
            evaluation=EvaluationConfig(**eval_cfg),
            visualization=VisualizationConfig(**vis_cfg),
            device=DeviceConfig(**dev_cfg),
            logging=LoggingConfig(**log_cfg),
            tiling=TilingConfig(**tiling_cfg),
            base_dir=base_dir,
        )

        return config
