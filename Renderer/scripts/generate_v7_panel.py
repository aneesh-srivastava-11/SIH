import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import json
from src.renderer.validation import validate_render_vs_real

def generate_v7_panel(rendered_img: np.ndarray, real_img: np.ndarray, shadow_mask: np.ndarray, output_path: str):
    """
    Generate the V7 Evidence Panel data and layout.
    """
    metrics = validate_render_vs_real(rendered_img, real_img, shadow_mask)
    
    print("========================================")
    print("        V7 EVIDENCE PANEL (GATE B)      ")
    print("========================================")
    print(f" NCC (Normalized Cross-Correlation): {metrics['ncc']:.4f}")
    print(f" SSIM (Structural Similarity):       {metrics['ssim']:.4f}")
    print(f" Gradient Histogram Distance (L1):   {metrics['grad_hist_l1']:.4f}")
    print("========================================")
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    # Mocking the pipeline to demonstrate the V7 panel script
    print("Simulating Render-vs-Real comparison for Vikram site...")
    H, W = 512, 512
    # Mock render
    rendered = np.random.normal(100, 20, (H, W)).astype(np.float32)
    # Mock real (slightly noisy version of render)
    real = rendered + np.random.normal(0, 10, (H, W)).astype(np.float32)
    # Mock shadows
    shadows = np.zeros((H, W), dtype=bool)
    shadows[100:200, 100:200] = True
    
    generate_v7_panel(rendered, real, shadows, "v7_panel_metrics.json")
