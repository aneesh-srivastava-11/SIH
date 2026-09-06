"""
Benchmark Runner CLI Application and Visualization Engine.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import json
import logging
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pipeline.config import BenchmarkConfig
from pipeline.dataset import DatasetLoader, ImagePair
from pipeline.registry import MethodRegistry
from evaluation.evaluator import BenchmarkEvaluator
from methods.base import RegistrationResult, RegistrationMethod

logger = logging.getLogger("benchmark")


class VisualizationGenerator:
    """Renders match lines, warped registration overlays, and aggregate performance charts."""

    @staticmethod
    def render_matches(
        ref_img: np.ndarray,
        tgt_img: np.ndarray,
        matches: Optional[np.ndarray],
        inlier_mask: Optional[List[int]] = None,
        max_draw: int = 150,
    ) -> np.ndarray:
        """Draw match correspondence lines between reference and target images side-by-side."""
        h1, w1 = ref_img.shape[:2]
        h2, w2 = tgt_img.shape[:2]
        out_h = max(h1, h2)
        out_w = w1 + w2

        # Convert to BGR for drawing colored lines
        ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape) == 2 else ref_img.copy()
        tgt_bgr = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR) if len(tgt_img.shape) == 2 else tgt_img.copy()

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = ref_bgr
        canvas[:h2, w1 : w1 + w2] = tgt_bgr

        if matches is None or len(matches) == 0:
            cv2.putText(canvas, "No Matches Found", (w1 // 2 - 80, out_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return canvas

        indices = np.arange(len(matches))
        if len(matches) > max_draw:
            indices = np.random.choice(len(matches), max_draw, replace=False)

        for idx in indices:
            m = matches[idx]
            pt1 = (int(round(m[0])), int(round(m[1])))
            pt2 = (int(round(m[2])) + w1, int(round(m[3])))

            is_inlier = bool(inlier_mask[idx]) if (inlier_mask and idx < len(inlier_mask)) else True
            color = (0, 255, 0) if is_inlier else (0, 0, 255)  # Green for inlier, Red for outlier

            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def render_overlay(ref_img: np.ndarray, tgt_img: np.ndarray, H: Optional[np.ndarray]) -> np.ndarray:
        """Render checkerboard/alpha overlay of warped target onto reference image frame."""
        ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape) == 2 else ref_img.copy()
        tgt_bgr = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR) if len(tgt_img.shape) == 2 else tgt_img.copy()

        h, w = ref_bgr.shape[:2]
        if H is None:
            return ref_bgr

        try:
            warped = cv2.warpPerspective(tgt_bgr, np.array(H, dtype=np.float64), (w, h))
            # Blended alpha composition
            overlay = cv2.addWeighted(ref_bgr, 0.5, warped, 0.5, 0)
            return overlay
        except Exception:
            return ref_bgr

    @staticmethod
    def generate_aggregate_plots(aggregated: Dict[str, Any], output_dir: str) -> List[str]:
        """Generate static summary charts (bar charts for success rate, RMSE, runtime)."""
        os.makedirs(output_dir, exist_ok=True)
        saved_plots = []

        if not aggregated:
            return saved_plots

        methods = list(aggregated.keys())
        success_rates = [aggregated[m]["success_rate"] * 100 for m in methods]
        rmses = [aggregated[m]["mean_rmse"] if aggregated[m]["mean_rmse"] is not None else 0.0 for m in methods]
        runtimes = [aggregated[m]["mean_runtime_ms"] if aggregated[m]["mean_runtime_ms"] is not None else 0.0 for m in methods]

        plt.style.use("dark_background")

        # 1. Success Rate Bar Chart
        plt.figure(figsize=(9, 5))
        bars = plt.bar(methods, success_rates, color="#00d4ff", edgecolor="white", alpha=0.85)
        plt.title("Registration Success Rate by Baseline Method (%)", fontsize=14, pad=12)
        plt.ylabel("Success Rate (%)", fontsize=12)
        plt.ylim(0, 105)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plot1 = os.path.join(output_dir, "success_rate_by_method.png")
        plt.savefig(plot1, dpi=150)
        plt.close()
        saved_plots.append(plot1)

        # 2. Mean Runtime Bar Chart
        plt.figure(figsize=(9, 5))
        bars = plt.bar(methods, runtimes, color="#ff006e", edgecolor="white", alpha=0.85)
        plt.title("Mean Execution Runtime by Baseline Method (ms)", fontsize=14, pad=12)
        plt.ylabel("Runtime (ms)", fontsize=12)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plot2 = os.path.join(output_dir, "runtime_by_method.png")
        plt.savefig(plot2, dpi=150)
        plt.close()
        saved_plots.append(plot2)

        return saved_plots


class BenchmarkRunner:
    """Orchestrates execution of all enabled registration baselines across discovered dataset pairs."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = BenchmarkConfig.load(config_path)
        self._setup_logging()

        self.dataset_loader = DatasetLoader(self.config)
        self.evaluator = BenchmarkEvaluator(self.config)

    def _setup_logging(self) -> None:
        log_dir = os.path.join(self.config.base_dir, self.config.logging.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"benchmark_{int(time.time())}.log")

        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def run(self) -> int:
        """Execute benchmark pipeline. Returns exit code (0 for success)."""
        logger.info("========================================")
        logger.info("IMAGE REGISTRATION BENCHMARK PLATFORM")
        logger.info("========================================")

        # 1. Discover Pairs
        pairs = self.dataset_loader.discover_pairs()
        if not pairs:
            logger.info("\nNo benchmark image pairs found.")
            logger.info(f"Expected location: {self.dataset_loader.pairs_dir}")
            logger.info("To run the benchmark, add pair images (e.g. reference1.png, target1.png) to data/pairs/\n")
            logger.info("No benchmark was executed. Exiting cleanly.")
            return 0

        logger.info(f"Dataset pairs found: {len(pairs)}")

        # 2. Discover & Instantiate Methods
        methods = MethodRegistry.get_enabled(self.config)
        all_methods_status = MethodRegistry.list_all()

        logger.info("\nBaseline Methods Status:")
        for name, (avail, reason) in all_methods_status.items():
            status_symbol = "[OK]" if avail else "[X]"
            logger.info(f"  {status_symbol} {name.upper()}: {reason}")

        if not methods:
            logger.error("No active methods available to execute. Exiting.")
            return 1

        logger.info(f"\nRunning benchmark for {len(methods)} active methods across {len(pairs)} pairs...\n")

        raw_out_dir = os.path.join(self.config.base_dir, self.config.output.raw_dir)
        vis_out_dir = os.path.join(self.config.base_dir, self.config.output.visualizations_dir)
        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(vis_out_dir, exist_ok=True)

        all_results: List[RegistrationResult] = []

        # 3. Execution Loop
        for method in methods:
            logger.info(f"--- Executing Method: {method.name} ---")

            for idx, pair in enumerate(pairs, 1):
                # Validate pair readability
                is_valid, err_msg = self.dataset_loader.validate_pair(pair)
                if not is_valid:
                    logger.warning(f"  [{idx}/{len(pairs)}] Pair '{pair.pair_id}' invalid: {err_msg}")
                    continue

                try:
                    ref_img = self.dataset_loader.load_image(pair.reference_path, self.config.preprocessing.convert_to_grayscale)
                    tgt_img = self.dataset_loader.load_image(pair.target_path, self.config.preprocessing.convert_to_grayscale)
                    H_gt = self.dataset_loader.load_ground_truth(pair.ground_truth_path)

                    # Execute registration method
                    result = method.run(ref_img, tgt_img, self.config, pair_id=pair.pair_id)

                    # Enrich result with GT metrics if available
                    result = self.evaluator.evaluate_result(result, H_gt)
                    all_results.append(result)

                    # Save per-pair raw JSON result
                    res_file = os.path.join(raw_out_dir, f"{method.name.lower().replace(' ', '_')}_{pair.pair_id}.json")
                    with open(res_file, "w", encoding="utf-8") as f:
                        json.dump(result.to_dict(), f, indent=2)

                    # Render and save visualizations if enabled
                    if self.config.visualization.generate:
                        method_vis_dir = os.path.join(vis_out_dir, method.name.lower().replace(" ", "_"))
                        os.makedirs(method_vis_dir, exist_ok=True)

                        vis_data = method.get_visualization_data()
                        inlier_mask = vis_data.get("inlier_mask")

                        matches_img = VisualizationGenerator.render_matches(
                            ref_img, tgt_img, method.get_matches(), inlier_mask, self.config.visualization.max_matches_drawn
                        )
                        cv2.imwrite(os.path.join(method_vis_dir, f"{pair.pair_id}_matches.png"), matches_img)

                        overlay_img = VisualizationGenerator.render_overlay(ref_img, tgt_img, method.get_transform())
                        cv2.imwrite(os.path.join(method_vis_dir, f"{pair.pair_id}_overlay.png"), overlay_img)

                    status_str = "SUCCESS" if result.success else f"FAILED ({result.error_message or 'Unspecified'})"
                    logger.info(f"  [{idx}/{len(pairs)}] Pair '{pair.pair_id}': {status_str} (inliers: {result.num_inliers or 0}, time: {result.runtime_ms}ms)")

                except Exception as e:
                    logger.error(f"  [{idx}/{len(pairs)}] Pair '{pair.pair_id}' threw uncaught exception: {e}")

        # 4. Aggregation and Summaries
        logger.info("\nAggregating benchmark statistics...")
        aggregated = self.evaluator.aggregate_results(all_results)

        agg_dir = os.path.join(self.config.base_dir, self.config.output.aggregated_dir)
        json_path, csv_path = self.evaluator.save_aggregated_results(aggregated, agg_dir)

        # 5. Generate Aggregate Plots
        VisualizationGenerator.generate_aggregate_plots(aggregated, vis_out_dir)

        logger.info(f"Benchmark complete.")
        logger.info(f"Raw results written to: {raw_out_dir}")
        logger.info(f"Summary JSON written to: {json_path}")
        logger.info(f"Leaderboard CSV written to: {csv_path}")

        # Print Leaderboard to Console
        df_lb = self.evaluator.generate_leaderboard_dataframe(aggregated)
        logger.info("\n" + "=" * 50)
        logger.info("FINAL METHOD LEADERBOARD")
        logger.info("=" * 50)
        logger.info("\n" + df_lb.to_string(index=False) + "\n")

        return 0


def main():
    runner = BenchmarkRunner()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
