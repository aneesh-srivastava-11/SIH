# Image Registration Benchmarking Platform

A modular, off-the-shelf benchmarking platform for comparing traditional and deep learning image registration algorithms. Designed specifically for satellite remote sensing and lunar imagery (e.g. Chandrayaan-2 OHRC/NAC, IIRS/WAC, DFSAR/SELENE).

---

## 1. Environment Configuration (`.env`)

Copy the example environment file to `.env`:

```bash
cd benchmarking
cp .env.example .env
```

Configuration variables available in `.env`:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Port for the Web Dashboard server |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID for PyTorch baseline methods |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `BENCHMARK_DATA_DIR` | `data/pairs` | Custom override for pair image directory |
| `BENCHMARK_OUTPUT_DIR` | `results` | Custom override for output results directory |

---

## 2. Installation

### Core Dependencies
```bash
cd benchmarking
pip install -r requirements.txt
```

### Optional Baseline Dependencies
To enable PyTorch deep learning models (**SuperPoint+LightGlue**, **EfficientLoFTR**):
```bash
pip install -r requirements-optional.txt
```

For **AROSICS** geospatial co-registration, install via Conda to resolve GDAL/PROJ dependencies:
```bash
conda install -c conda-forge arosics
```

For **RIFT2**, clone the official repository into the vendor directory:
```bash
git clone https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation vendor/rift2
```

---

## 3. Quick Start Guide

### Step 1: Run Baseline Benchmark
```bash
python -m pipeline.runner
```
*(If `data/pairs/` has no images, it reports zero pairs and exits cleanly without errors.)*

### Step 2: Launch Web Dashboard
```bash
python -m dashboard.app
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Step 3: Run Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 4. System Architecture

```text
                  REAL IMAGE PAIRS (data/pairs/)
                               │
                               ▼
                        DATASET LOADER
                               │
                               ▼
                        PREPROCESSING
                               │
          ┌──────────────┼──────────────┼──────────────┐
          ▼              ▼              ▼              ▼
        SIFT           AKAZE          RIFT2         AROSICS
          │              │              │              │
          ├──────────────┴──────────────┤              │
          ▼                             ▼              │
       SP+LG                         ELoFTR            │
          │                             │              │
          └──────────────┬──────────────┴──────────────┘
                         ▼
                COMMON EVALUATOR
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           Metrics     Results    Visualizations
             │           │           │
             └───────────┼───────────┘
                         ▼
                     DASHBOARD
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Method Comparison      Pair Inspection
              │
              ▼
       ISRO Benchmark Comparison (Makharia et al., arXiv:2509.04775)
              │
              ▼
      Future Custom Model (methods/my_custom_model.py)
              │
              ▼
        BASELINE BEAT?
```

---

## 5. Adding Image Pairs

Drop image pairs directly into `data/pairs/`.

### Flat Naming (Recommended)
```text
data/pairs/
├── reference1.png
├── target1.png
├── reference2.jpg
├── target2.jpg
```

### Subfolder Naming (Supported)
```text
data/pairs/
├── pair_001/
│   ├── reference.png
│   └── target.png
```

### Ground Truth Homography (Optional)
Add ground truth 3x3 matrices in `data/ground_truth/`:
```text
data/ground_truth/
├── pair_1.json
├── pair_2.json
```

Example `pair_1.json`:
```json
{
  "pair_id": "pair_1",
  "transform": [
    [0.9659, -0.2588, 20.0],
    [0.2588,  0.9659, -15.0],
    [0.0,     0.0,     1.0]
  ]
}
```

---

## 6. Directory Structure

```text
benchmarking/
├── .env.example                # Sample environment configuration file
├── .env                        # Active environment configuration file
├── data/
│   ├── raw/                    # Raw unprocessed image storage
│   ├── pairs/                  # Benchmark image pairs (reference1.png / target1.png)
│   └── ground_truth/           # Ground truth homography matrices (pair_1.json)
├── methods/
│   ├── base.py                 # Abstract base class & RegistrationResult schema
│   ├── sift.py                 # SIFT baseline adapter
│   ├── akaze.py                # AKAZE baseline adapter
│   ├── rift2.py                # RIFT2 baseline adapter
│   ├── superpoint_lightglue.py # SuperPoint + LightGlue baseline adapter
│   ├── efficient_loftr.py      # EfficientLoFTR baseline adapter
│   └── arosics_method.py       # AROSICS baseline adapter
├── evaluation/
│   ├── metrics.py              # RMSE and inlier ratio metric calculations
│   ├── evaluator.py            # Result enrichment & dataset summary aggregator
│   └── comparison.py           # Cross-method & ISRO paper comparison
├── pipeline/
│   ├── runner.py               # CLI benchmark runner
│   ├── config.py               # Configuration loader
│   ├── registry.py             # Algorithm registry
│   └── dataset.py              # Pair discoverer & image loader
├── results/
│   ├── raw/                    # Per-pair experiment JSON results
│   ├── aggregated/             # Summary.json & Leaderboard CSV
│   └── visualizations/         # Generated match lines and overlay plots
├── dashboard/
│   ├── app.py                  # Flask web server
│   ├── api.py                  # REST API endpoints
│   ├── templates/index.html    # Single Page Application HTML
│   └── static/                 # CSS theme and Chart.js frontend
├── configs/
│   ├── default.yaml            # Pipeline configuration
│   └── isro_benchmark.yaml     # ISRO paper benchmark numbers (arXiv:2509.04775)
└── tests/                      # Unit tests & synthetic fixture generators
```

---

## 7. Baseline Algorithms Reference

| Method | Key Libraries | Core/Optional | Default Device |
|---|---|---|---|
| **SIFT** | OpenCV (`cv2.SIFT_create`) | Core | CPU |
| **AKAZE** | OpenCV (`cv2.AKAZE_create`) | Core | CPU |
| **RIFT2** | NumPy, SciPy (`vendor/rift2`) | Optional (Git Repo) | CPU |
| **SuperPoint + LightGlue** | PyTorch, Kornia (`kornia.feature`) | Optional | CUDA (Default) / CPU |
| **EfficientLoFTR** | PyTorch, Hugging Face `transformers` | Optional | CUDA (Default) / CPU |
| **AROSICS** | AROSICS, GDAL, Rasterio | Optional (Conda) | CPU |

---

## 8. Evaluation Metrics

- **RMSE (Root Mean Square Error)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\|\mathbf{p}'_i - \mathbf{p}^{\text{gt}}_i\|^2)}$$
  Calculated over a grid of test points transformed by predicted $\mathbf{H}$ vs ground truth $\mathbf{H}_{\text{gt}}$.

- **Inlier Ratio**:
  $$\text{Ratio} = \frac{N_{\text{inliers}}}{N_{\text{matches}}}$$

- **Success Rate**: Percentage of dataset pairs meeting minimum inlier thresholds without exceptions.

---

## 9. Adding Your Team's Custom Model

To benchmark a new custom model:

1. Create a new file in `methods/` (e.g. `methods/my_custom_model.py`):
```python
from methods.base import RegistrationMethod, RegistrationResult
from pipeline.registry import MethodRegistry

class MyCustomModel(RegistrationMethod):
    @property
    def name(self) -> str:
        return "CustomModel"

    def run(self, reference_image, target_image, config, pair_id="") -> RegistrationResult:
        # Implement registration logic here
        return RegistrationResult(
            method=self.name,
            pair_id=pair_id,
            success=True,
            transform=H.tolist(),
            runtime_ms=45.0,
        )

    def get_matches(self): return self._matches
    def get_transform(self): return self._transform
    def get_visualization_data(self): return {}

MethodRegistry.register("custom_model", MyCustomModel)
```

2. Add `"custom_model"` to `methods.enabled` in `configs/default.yaml`.
3. Run `python -m pipeline.runner`. It automatically appears in the CLI leaderboard and Web Dashboard!

---

## 10. Reproducing ISRO Chandrayaan-2 Benchmark

The platform pre-loads reported benchmark figures from ISRO's paper:
> *Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data* (Makharia et al., arXiv:2509.04775)

View comparison tables directly in the **ISRO Paper Comparison** tab of the Web Dashboard.
