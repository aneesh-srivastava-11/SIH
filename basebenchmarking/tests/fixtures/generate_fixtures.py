"""
Synthetic Test Fixture Generator for Development and Unit Testing.
Important: Synthetic images are kept strictly in tests/fixtures/ and NEVER placed in data/pairs/.
"""

from typing import Tuple, Dict, Any
import os
import json
import numpy as np
import cv2


def generate_synthetic_pair(output_dir: str, prefix: str = "synth1") -> Tuple[str, str, str]:
    """
    Generate synthetic reference image, target image warped by known homography H,
    and ground truth homography JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = 512, 512
    # Create checkerboard pattern with shapes and noise
    ref_img = np.zeros((h, w), dtype=np.uint8)

    # Checkerboard
    sq = 64
    for i in range(0, h, sq):
        for j in range(0, w, sq):
            if ((i // sq) + (j // sq)) % 2 == 0:
                ref_img[i : i + sq, j : j + sq] = 200

    # Draw shapes for distinctive keypoints
    cv2.circle(ref_img, (150, 150), 40, (255,), -1)
    cv2.rectangle(ref_img, (300, 100), (420, 220), (50,), -1)
    cv2.line(ref_img, (50, 400), (450, 400), (120,), 8)

    # Known Affine/Homography transformation: Rotation + Translation + Scale
    angle = 15.0  # degrees
    scale = 0.95
    tx, ty = 20.0, -15.0

    center = (w / 2.0, h / 2.0)
    M_affine = cv2.getRotationMatrix2D(center, angle, scale)
    M_affine[0, 2] += tx
    M_affine[1, 2] += ty

    H_gt = np.eye(3, dtype=np.float64)
    H_gt[:2, :] = M_affine

    # Warp target image using GT Homography
    tgt_img = cv2.warpPerspective(ref_img, H_gt, (w, h))

    ref_path = os.path.join(output_dir, f"reference_{prefix}.png")
    tgt_path = os.path.join(output_dir, f"target_{prefix}.png")
    gt_path = os.path.join(output_dir, f"pair_{prefix}.json")

    cv2.imwrite(ref_path, ref_img)
    cv2.imwrite(tgt_path, tgt_img)

    gt_data = {
        "label": "[SYNTHETIC TEST FIXTURE - NOT BENCHMARK DATA]",
        "pair_id": f"pair_{prefix}",
        "transform": H_gt.tolist(),
        "transform_type": "homography",
        "description": "Synthetic affine transform (15 deg rotation, 0.95 scale, [20, -15] shift)",
    }

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    return ref_path, tgt_path, gt_path


if __name__ == "__main__":
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    r, t, g = generate_synthetic_pair(fixtures_dir, "test1")
    print(f"Generated synthetic test fixtures:")
    print(f"  Reference: {r}")
    print(f"  Target:    {t}")
    print(f"  GT Matrix: {g}")
