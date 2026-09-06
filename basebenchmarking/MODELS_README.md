# Models Overview & Dashboard Guide

This README explains the models included in the image registration benchmark, how they work under the hood, how to resolve common setup issues (like with EfficientLoFTR), and how to run the evaluation dashboard.

---

## 1. How to Make EfficientLoFTR Work

EfficientLoFTR is a high-performance deep learning feature matcher. The benchmark provides two paths to run EfficientLoFTR:

### Method A: Hugging Face (Recommended & Automated)
The easiest way is via the official `transformers` integration. The benchmark handles everything for you if you have `transformers` installed.
- **Dependency:** `transformers>=4.30.0` (Included in `requirements.txt`).
- **How it works:** Under the hood, `methods/efficient_loftr.py` detects if `transformers` is installed. It automatically downloads the `zju-community/efficientloftr` model from the Hugging Face hub on its first run and uses `EfficientLoFTRForKeypointMatching` and `AutoImageProcessor` to compute matches.

### Method B: Local Fallback
If you are completely offline or prefer the original repository codebase:
1. Clone the repo: `git clone https://github.com/zju-3dv/EfficientLoFTR.git src`
2. Our adapter (`methods/efficient_loftr.py`) will detect `src.loftr` and fallback to the legacy model if `transformers` is unavailable or errors out.

**Note on RANSAC:**
EfficientLoFTR produces a *dense* set of matches. The `efficient_loftr.py` code applies OpenCV's `findHomography` with RANSAC (`cv2.RANSAC`) as a post-processing step to filter out outliers and generate the final transformation matrix. 

---

## 2. Explanation of Model Codes

All models inherit from `RegistrationMethod` defined in `methods/base.py`. They all implement the `run(reference_image, target_image, config, pair_id)` method and return a `RegistrationResult`.

### 1. SIFT (`methods/sift.py`)
- **Type:** Traditional handcrafted feature extractor.
- **Code Flow:** Uses `cv2.SIFT_create()` to detect keypoints and compute descriptors. It uses a FLANN-based KD-Tree matcher (or BFMatcher) with a Lowe's ratio test (usually 0.7 or 0.8) to find good matches. Finally, it uses `cv2.findHomography` with RANSAC to calculate the affine/perspective transform.

### 2. AKAZE (`methods/akaze.py`)
- **Type:** Traditional handcrafted feature extractor (Fast, nonlinear scale space).
- **Code Flow:** Uses `cv2.AKAZE_create()`. Similar to SIFT, but often paired with binary descriptor matchers like `cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING`. Followed by Ratio Test and RANSAC homography.

### 3. SuperPoint + LightGlue (`methods/superpoint_lightglue.py`)
- **Type:** Deep Learning (Sparse feature extraction + Attention-based matching).
- **Code Flow:** Uses the `kornia.feature` library or official `lightglue` repository. SuperPoint extracts sub-pixel keypoints and descriptors using a CNN. LightGlue acts as a Graph Neural Network (GNN) matcher, taking SuperPoint outputs and using self/cross-attention layers to identify matches. Requires PyTorch.

### 4. EfficientLoFTR (`methods/efficient_loftr.py`)
- **Type:** Deep Learning (Dense matching, detector-free).
- **Code Flow:** Doesn't extract keypoints first. It extracts coarse feature maps using a CNN/Transformer backbone, computes a correlation matrix between the two images, and refines matches to sub-pixel accuracy. Evaluates directly on images using Hugging Face's `EfficientLoFTRForKeypointMatching`. Requires PyTorch + Transformers.

### 5. RIFT2 (`methods/rift2.py`)
- **Type:** Traditional (Multi-modal focused).
- **Code Flow:** Designed for matching images from different sensors (e.g., optical to SAR). Uses Phase Congruency (PC) and a Maximum Index Map (MIM) to build robust descriptors invariant to non-linear intensity changes. Requires cloning the external repository.

### 6. AROSICS (`methods/arosics_method.py`)
- **Type:** Geospatial phase-correlation.
- **Code Flow:** Operates on geographic metadata if available, using phase correlation on global and local image levels to compute spatial shifts. Best for remote sensing images. Requires `arosics` and `rasterio` libraries.

---

## 3. How to Run the Dashboard

The project includes a web-based dashboard built on Flask to visualize the results of the models on your dataset.

### Step 1: Install Requirements
Make sure your environment is ready (includes `flask`):
```bash
pip install -r requirements.txt
```

### Step 2: Run the Benchmarks
Before the dashboard can display anything, you need to run the evaluation pipeline on your image pairs (place your images in `data/pairs/`):
```bash
python -m pipeline.runner
```
*This processes the images and saves the results to the `results/` directory.*

### Step 3: Launch the Dashboard
Run the Flask application:
```bash
python -m dashboard.app
```

### Step 4: View the Dashboard
Open your web browser and go to:
**http://127.0.0.1:5000**

- **Leaderboard Tab:** Shows average RMSE, success rates, and runtimes across the dataset for each method.
- **Detailed Results Tab:** View specific matching visualizations (inlier/outlier lines) for individual image pairs.
- **ISRO Benchmark Tab:** Compare your local run against published baseline data.
