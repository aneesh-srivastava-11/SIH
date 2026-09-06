import sys
import numpy as np
import cv2

# Add current path to sys.path
sys.path.append('.')

from methods.efficient_loftr import EfficientLoFTRMethod

class MockConfig:
    class Device:
        prefer_gpu = False
    class Matching:
        min_matches = 10
    class Ransac:
        reproj_threshold = 3.0
    class Evaluation:
        success_min_inliers = 10
        success_min_inlier_ratio = 0.1
    
    device = Device()
    matching = Matching()
    ransac = Ransac()
    evaluation = Evaluation()

if __name__ == "__main__":
    # Create random images
    ref_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tgt_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    print("Initializing EfficientLoFTRMethod...")
    method = EfficientLoFTRMethod()
    avail, reason = method.is_available()
    print(f"Available: {avail}, Reason: {reason}")
    
    if avail:
        print("Running EfficientLoFTR on dummy images...")
        result = method.run(ref_image, tgt_image, MockConfig(), pair_id="test_pair")
        
        print("\n--- Result ---")
        print(f"Success: {result.success}")
        print(f"Num matches: {result.num_matches}")
        print(f"Num inliers: {result.num_inliers}")
        print(f"Runtime (ms): {result.runtime_ms}")
        print(f"Error message: {result.error_message}")
        print(f"Method: {result.method}")
    else:
        print("Method not available, skipping run test.")
