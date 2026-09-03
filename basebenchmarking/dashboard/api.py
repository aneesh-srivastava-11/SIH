"""
REST API Endpoints for Image Registration Benchmarking Dashboard.
"""

from typing import Dict, Any, List, Optional
import os
import json
import flask
from flask import Blueprint, jsonify, request, send_file

from pipeline.config import BenchmarkConfig
from pipeline.dataset import DatasetLoader
from pipeline.registry import MethodRegistry
from evaluation.comparison import ISROBenchmarkComparison

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _get_config() -> BenchmarkConfig:
    return BenchmarkConfig.load()


@api_bp.route("/overview", methods=["GET"])
def get_overview():
    """Return overview metrics: dataset size, method count, completed runs, best performer."""
    config = _get_config()
    loader = DatasetLoader(config)
    pairs = loader.discover_pairs()

    summary_file = os.path.join(config.base_dir, config.output.aggregated_dir, "summary.json")
    aggregated: Dict[str, Any] = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                aggregated = json.load(f)
        except Exception:
            pass

    methods_status = MethodRegistry.list_all()
    num_pairs = len(pairs)
    num_methods = len(methods_status)
    has_data = bool(aggregated and num_pairs > 0)

    best_method = None
    best_rmse = None
    best_success_rate = None

    if has_data:
        # Find best method by success rate and RMSE
        sorted_methods = sorted(
            aggregated.values(),
            key=lambda x: (x.get("success_rate", 0), -(x.get("mean_rmse") or 9999)),
            reverse=True,
        )
        if sorted_methods:
            best = sorted_methods[0]
            best_method = best.get("method")
            best_rmse = best.get("mean_rmse")
            best_success_rate = best.get("success_rate")

    return jsonify({
        "has_data": has_data,
        "dataset_size": num_pairs,
        "total_methods": num_methods,
        "enabled_methods": len([m for m, (a, _) in methods_status.items() if a]),
        "completed_runs": sum(m.get("total_pairs", 0) for m in aggregated.values()) if aggregated else 0,
        "best_method": best_method,
        "best_rmse": best_rmse,
        "best_success_rate": best_success_rate,
        "pairs_dir": config.dataset.pairs_dir,
    })


@api_bp.route("/leaderboard", methods=["GET"])
def get_leaderboard():
    """Return leaderboard summary data for all evaluated algorithms."""
    config = _get_config()
    summary_file = os.path.join(config.base_dir, config.output.aggregated_dir, "summary.json")

    if not os.path.exists(summary_file):
        return jsonify({"has_data": False, "leaderboard": []})

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            aggregated = json.load(f)
        leaderboard_list = list(aggregated.values())
        # Sort by success rate desc
        leaderboard_list.sort(key=lambda x: (x.get("success_rate", 0), -(x.get("mean_rmse") or 9999)), reverse=True)
        return jsonify({"has_data": True, "leaderboard": leaderboard_list})
    except Exception as e:
        return jsonify({"has_data": False, "error": str(e), "leaderboard": []})


@api_bp.route("/methods", methods=["GET"])
def get_methods():
    """Return list of all registered methods and availability status."""
    status = MethodRegistry.list_all()
    methods_list = []
    for name, (avail, reason) in status.items():
        methods_list.append({
            "id": name,
            "name": name.upper(),
            "available": avail,
            "reason": reason,
        })
    return jsonify(methods_list)


@api_bp.route("/pairs", methods=["GET"])
def get_pairs():
    """Return list of discovered benchmark pair IDs."""
    config = _get_config()
    loader = DatasetLoader(config)
    pairs = loader.discover_pairs()
    return jsonify([{"pair_id": p.pair_id, "has_gt": p.ground_truth_path is not None} for p in pairs])


@api_bp.route("/pair/<pair_id>", methods=["GET"])
def get_pair_details(pair_id: str):
    """Return all method results and visualization paths for a specific image pair."""
    config = _get_config()
    raw_dir = os.path.join(config.base_dir, config.output.raw_dir)
    vis_dir = os.path.join(config.base_dir, config.output.visualizations_dir)

    results = []
    if os.path.exists(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.endswith(f"_{pair_id}.json"):
                path = os.path.join(raw_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        method_name = data.get("method", "").lower().replace(" ", "_")
                        data["matches_vis_url"] = f"/api/visualizations/{method_name}/{pair_id}/matches"
                        data["overlay_vis_url"] = f"/api/visualizations/{method_name}/{pair_id}/overlay"
                        results.append(data)
                except Exception:
                    pass

    return jsonify({"pair_id": pair_id, "results": results})


@api_bp.route("/visualizations/<method>/<pair_id>/<vis_type>", methods=["GET"])
def get_visualization_image(method: str, pair_id: str, vis_type: str):
    """Serve visualization PNG image (matches or overlay)."""
    config = _get_config()
    filename = f"{pair_id}_matches.png" if vis_type == "matches" else f"{pair_id}_overlay.png"
    img_path = os.path.join(config.base_dir, config.output.visualizations_dir, method.lower().replace(" ", "_"), filename)

    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/png")
    
    # Generate placeholder transparent 1x1 image if missing
    placeholder = os.path.join(config.base_dir, "dashboard", "static", "placeholder.png")
    if os.path.exists(placeholder):
        return send_file(placeholder, mimetype="image/png")

    return jsonify({"error": "Visualization image not found"}), 404


@api_bp.route("/isro-comparison", methods=["GET"])
def get_isro_comparison():
    """Return comparison table between current benchmark runs and ISRO paper numbers."""
    config = _get_config()
    summary_file = os.path.join(config.base_dir, config.output.aggregated_dir, "summary.json")
    aggregated = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                aggregated = json.load(f)
        except Exception:
            pass

    isro = ISROBenchmarkComparison()
    table_data = isro.get_comparison_table(aggregated)
    metadata = isro.get_paper_metadata()

    return jsonify({"metadata": metadata, "table": table_data})
