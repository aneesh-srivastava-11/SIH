"""
Cross-Method Comparison and ISRO Benchmark Paper Reproduction Module.
"""

from typing import Dict, Any, List, Optional
import os
import yaml
import pandas as pd


class ISROBenchmarkComparison:
    """Compares current baseline run results against published ISRO paper figures."""

    def __init__(self, isro_yaml_path: Optional[str] = None):
        if not isro_yaml_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            isro_yaml_path = os.path.join(base_dir, "configs", "isro_benchmark.yaml")

        self.paper_data: Dict[str, Any] = {}
        if os.path.exists(isro_yaml_path):
            with open(isro_yaml_path, "r", encoding="utf-8") as f:
                self.paper_data = yaml.safe_load(f) or {}

    def get_comparison_table(self, current_aggregated: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate comparison rows: [Dataset, Method, ISRO Reported RMSE X, Ours RMSE X, ISRO Time, Ours Time].
        """
        reported_datasets = self.paper_data.get("reported_results", {})
        rows = []

        for dataset_id, method_results in reported_datasets.items():
            for method_name, isro_metrics in method_results.items():
                our_data = current_aggregated.get(method_name) or current_aggregated.get(method_name.replace("+", " + "))

                our_rmse_x = our_data.get("mean_rmse_x") if our_data else None
                our_rmse_y = our_data.get("mean_rmse_y") if our_data else None
                our_time_sec = (our_data.get("mean_runtime_ms") / 1000.0) if (our_data and our_data.get("mean_runtime_ms")) else None

                rows.append({
                    "dataset": dataset_id,
                    "method": method_name,
                    "isro_status": isro_metrics.get("status", "N/A"),
                    "isro_rmse_x": isro_metrics.get("rmse_x"),
                    "our_rmse_x": our_rmse_x,
                    "isro_rmse_y": isro_metrics.get("rmse_y"),
                    "our_rmse_y": our_rmse_y,
                    "isro_time_sec": isro_metrics.get("runtime_sec"),
                    "our_time_sec": round(our_time_sec, 3) if our_time_sec is not None else None,
                })

        return rows

    def get_paper_metadata(self) -> Dict[str, Any]:
        """Return bibliographic details of ISRO benchmark paper."""
        return self.paper_data.get("paper", {})


class MethodComparison:
    """Side-by-side comparative analysis of N methods across image pairs."""

    @staticmethod
    def compare_pairs(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create pivot table comparing method performance pair by pair."""
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        pivot = df.pivot(index="pair_id", columns="method", values=["success", "rmse", "runtime_ms", "inlier_ratio"])
        return pivot
