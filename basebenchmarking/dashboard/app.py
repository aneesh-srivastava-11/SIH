"""
Flask Application Server for Image Registration Benchmarking Dashboard.
"""

import os
import sys
from flask import Flask, render_template

from pipeline.config import BenchmarkConfig
from dashboard.api import api_bp


def create_app() -> Flask:
    """Factory function to initialize Flask application."""
    config = BenchmarkConfig.load()
    base_dir = config.base_dir

    template_dir = os.path.join(base_dir, "dashboard", "templates")
    static_dir = os.path.join(base_dir, "dashboard", "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
    )

    # Register API blueprint
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


def main():
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"==================================================")
    print(f"  IMAGE REGISTRATION BENCHMARK DASHBOARD  ")
    print(f"==================================================")
    print(f"  Server running on http://127.0.0.1:{port}")
    print(f"  Press Ctrl+C to exit.")
    print(f"==================================================")
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
