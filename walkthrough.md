# Global Data Folder Migration & Analysis Walkthrough

We've completed the implementation plan to consolidate the data paths, add `.npy` support to the benchmark pipeline, and analyze the data inconsistencies!

## 1. Global Data Folder Setup

- **Redundant Folders Removed**: Deleted the legacy `basebenchmarking/data` and `finetuningiirscrossmodal/data` directories. All data is now sourced from the global `C:\Users\ANEESH\Desktop\SIH\data` folder.
- **Pipeline Overhaul**: Modified `DatasetLoader` in `basebenchmarking/pipeline/dataset.py` to:
  - Recursively search sub-directories (like `p1` and `p3`) inside `data/cropped/`.
  - Intelligently pair files inside these folders (if exactly two supported files are found, they are automatically designated as the reference and target pair).
- **Array Support**: Extended `supported_formats` in `default.yaml` to include `npy` and modified `cv2.imread` logic to use `np.load()` for numpy arrays. This allows the benchmarking suite to load your `p1` and `p3` files directly.
- **Fine-Tuning Updates**: Updated paths in `finetuningiirscrossmodal/configs/training.yaml` and `training/__main__.py` to point to the unified `../data/rendered_pairs` folder.

## 2. CRS Verification

I updated and ran `C:\Users\ANEESH\Desktop\SIH\Preprocess\check_crs.py` against all the raw root data formats. Here are the results:

> [!WARNING]
> **Missing CRS in NAC**
> The NAC `M1529798616LC.IMG` file is a raw PDS image format and `rasterio` reads its CRS as `None`. To project this image accurately, the metadata (`nac_geometry.json` or PDS labels) must be used to manually construct and assign the Geotransform and CRS during preprocessing.

- **WAC**: Correctly georeferenced (`SimpleCylindrical Moon`).
- **NAC**: Missing CRS (`None`). 
- **IIR & OHR**: `.qub` and `.img` extensions are not directly supported by `rasterio` without a specialized GDAL driver or converting them to `.tif` via ISIS/GDAL tools first.

## 3. Why RANSAC Fails on `p3` (Cross-Modal Registration)

RANSAC (Random Sample Consensus) is failing on `p3` while succeeding on `p1` due to the fundamental differences in the datasets:

### The `p1` Dataset (NAC + OHRC)
- **Modality**: Both are high-resolution panchromatic optical cameras.
- **Features**: They share very similar visual features, textures, shadows, and gradients.
- **RANSAC Success**: SIFT can easily find corresponding keypoints. While some are false matches, there are enough *true* matches that RANSAC can lock onto a mathematically valid Homography matrix and successfully filter the outliers.

### The `p3` Dataset (IIRS + WAC)
- **Modality**: This is a **cross-modal** pair. IIRS is an infrared/hyperspectral spectrometer, while WAC is a wide-angle visible camera.
- **Features**: A crater in WAC (visible light) looks entirely different in IIRS (infrared heat signatures). 
- **RANSAC Failure**: Gradient-based descriptors like SIFT fail to find meaningful correspondences across modalities. Almost **100%** of the initial SIFT matches are false positives (noise). RANSAC works by finding a geometric consensus among matches; if there are virtually no true matches in the pool, RANSAC cannot find a valid homography, resulting in 0 inliers or an outright failure.

> [!TIP]
> **Solution for `p3`**
> Traditional methods (SIFT/AKAZE) will inherently fail on cross-modal pairs like `p3`. This is exactly why your deep learning models (`SuperPoint + LightGlue` and `EfficientLoFTR`) are needed! To fix `p3`, we need to use fine-tuned deep learning models that learn semantic features rather than relying on pixel gradients.
