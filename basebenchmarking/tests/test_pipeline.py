"""
Unit tests for pipeline runner and method registry.
"""

import os
import pytest
from pipeline.config import BenchmarkConfig
from pipeline.registry import MethodRegistry
from pipeline.runner import BenchmarkRunner


def test_registry_discovery():
    MethodRegistry.discover()
    all_methods = MethodRegistry.list_all()

    assert "sift" in all_methods
    assert "akaze" in all_methods
    assert "rift2" in all_methods
    assert "superpoint_lightglue" in all_methods
    assert "efficient_loftr" in all_methods
    assert "arosics" in all_methods


def test_registry_get_enabled():
    config = BenchmarkConfig()
    methods = MethodRegistry.get_enabled(config)

    # At least SIFT and AKAZE should be available on any OpenCV installation
    names = [m.name for m in methods]
    assert "SIFT" in names
    assert "AKAZE" in names


def test_runner_empty_dataset(tmp_path):
    config_file = tmp_path / "config.yaml"
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()

    pairs_dir_str = pairs_dir.as_posix()
    config_content = f"""
dataset:
  pairs_dir: '{pairs_dir_str}'
methods:
  enabled:
    - sift
"""
    config_file.write_text(config_content)

    runner = BenchmarkRunner(str(config_file))
    exit_code = runner.run()

    # Empty dataset must exit cleanly with code 0
    assert exit_code == 0

