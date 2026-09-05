"""
Benchmark Evaluator and Aggregator for multi-method dataset summaries.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd

from methods.base import RegistrationResult
from evaluation.metrics import compute_homography_rmse, compute_reprojection_errors


class BenchmarkEvaluator:
    """Evaluates raw results against ground truth and produces dataset summaries."""

    def __init__(self, config: Any):
        self.config = config

    def evaluate_result(self, result: RegistrationResult, H_gt: Optional[np.ndarray]) -> RegistrationResult:
        """
        Enrich a RegistrationResult with ground truth metrics if ground truth homography is available.
        Never fabricates metrics if H_gt is None.
        """
        if H_gt is None or result.transform is None:
            result.rmse = None
            result.rmse_x = None
            result.rmse_y = None
            result.reprojection_error_mean = None
            result.reprojection_error_median = None
            result.reprojection_error_max = None
            return result

        H_pred = np.array(result.transform, dtype=np.float64)
        rmse_overall, rmse_x, rmse_y = compute_homography_rmse(H_pred, H_gt)
        mean_err, median_err, max_err = compute_reprojection_errors(H_pred, H_gt)

        result.rmse = round(rmse_overall, 4) if rmse_overall is not None else None
        result.rmse_x = round(rmse_x, 4) if rmse_x is not None else None
        result.rmse_y = round(rmse_y, 4) if rmse_y is not None else None
        result.reprojection_error_mean = round(mean_err, 4) if mean_err is not None else None
        result.reprojection_error_median = round(median_err, 4) if median_err is not None else None
        result.reprojection_error_max = round(max_err, 4) if max_err is not None else None

        return result

    def aggregate_results(self, results: List[RegistrationResult]) -> Dict[str, Any]:
        """
        Aggregate per-pair results into method-level performance summaries.
        Returns dict keyed by method name with statistics: success rate, mean RMSE, mean runtime, etc.
        """
        if not results:
            return {}

        df = pd.DataFrame([r.to_dict() for r in results])
        aggregated: Dict[str, Any] = {}

        for method_name, group in df.groupby("method"):
            total_runs = len(group)
            success_count = int(group["success"].sum())
            success_rate = float(success_count / total_runs) if total_runs > 0 else 0.0

            successful_runs = group[group["success"] == True]

            # Filter non-null metrics for averaging
            valid_rmse = group["rmse"].dropna()
            valid_rmse_x = group["rmse_x"].dropna()
            valid_rmse_y = group["rmse_y"].dropna()
            valid_inlier_ratio = group["inlier_ratio"].dropna()
            valid_runtime = group["runtime_ms"].dropna()

            agg_entry = {
                "method": str(method_name),
                "total_pairs": total_runs,
                "successful_pairs": success_count,
                "failed_pairs": total_runs - success_count,
                "success_rate": round(success_rate, 4),
                "mean_rmse": round(float(valid_rmse.mean()), 4) if not valid_rmse.empty else None,
                "median_rmse": round(float(valid_rmse.median()), 4) if not valid_rmse.empty else None,
                "mean_rmse_x": round(float(valid_rmse_x.mean()), 4) if not valid_rmse_x.empty else None,
                "mean_rmse_y": round(float(valid_rmse_y.mean()), 4) if not valid_rmse_y.empty else None,
                "mean_inlier_ratio": round(float(valid_inlier_ratio.mean()), 4) if not valid_inlier_ratio.empty else None,
                "mean_runtime_ms": round(float(valid_runtime.mean()), 2) if not valid_runtime.empty else None,
                "median_runtime_ms": round(float(valid_runtime.median()), 2) if not valid_runtime.empty else None,
                "mean_inliers": round(float(group["num_inliers"].dropna().mean()), 1) if "num_inliers" in group and not group["num_inliers"].dropna().empty else None,
                "failures": group[group["success"] == False][["pair_id", "error_message"]].to_dict(orient="records"),
            }
            aggregated[str(method_name)] = agg_entry

        return aggregated

    def generate_leaderboard_dataframe(self, aggregated: Dict[str, Any]) -> pd.DataFrame:
        """Convert aggregated summary dict into a sorted leaderboard DataFrame."""
        if not aggregated:
            return pd.DataFrame(columns=["Method", "Success Rate", "Mean RMSE", "Median RMSE", "Mean Inlier Ratio", "Mean Runtime (ms)", "Failures"])

        rows = []
        for method_name, data in aggregated.items():
            rows.append({
                "Method": method_name,
                "Success Rate": f"{data['success_rate'] * 100:.1f}%",
                "Mean RMSE": f"{data['mean_rmse']:.4f}" if data['mean_rmse'] is not None else "N/A (No GT)",
                "Median RMSE": f"{data['median_rmse']:.4f}" if data['median_rmse'] is not None else "N/A",
                "Mean Inlier Ratio": f"{data['mean_inlier_ratio'] * 100:.1f}%" if data['mean_inlier_ratio'] is not None else "N/A",
                "Mean Runtime (ms)": f"{data['mean_runtime_ms']:.1f}" if data['mean_runtime_ms'] is not None else "N/A",
                "Failures": data["failed_pairs"],
            })

        df = pd.DataFrame(rows)
        # Sort by Success Rate desc, then Mean RMSE asc
        df = df.sort_values(by=["Success Rate", "Failures"], ascending=[False, True])
        return df

    def save_aggregated_results(self, aggregated: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
        """Save aggregated results to JSON and CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "summary.json")
        csv_path = os.path.join(output_dir, "leaderboard.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)

        df_leaderboard = self.generate_leaderboard_dataframe(aggregated)
        df_leaderboard.to_csv(csv_path, index=False)

        return json_path, csv_path
