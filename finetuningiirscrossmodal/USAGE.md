# How to Use the Fine-Tuning & Cross-Modal Module (`finetuningiirscrossmodal`)

This document provides step-by-step instructions for:
1. Preparing DEM-rendered image pair datasets.
2. Fine-tuning **EfficientLoFTR** and **RoMa** deep feature matchers.
3. Running **MatchAnything** for cross-modal (IIRS hyperspectral ↔ optical) registration.
4. Integrating fine-tuned checkpoints into evaluation workflows.

---

## 1. Quick Start

### Installation
Ensure core dependencies are installed in your Python environment:
```bash
pip install -r finetuningiirscrossmodal/requirements.txt
```

*(Optional)* Install PyTorch deep learning extensions for official pretrained model weights:
```bash
pip install -r finetuningiirscrossmodal/requirements-optional.txt
```

---

## 2. Dataset Preparation

The fine-tuning pipeline expects DEM-rendered synthetic image pairs placed inside `data/rendered_pairs/`.

### File Naming Convention
For pair `$N$` (e.g. `0`, `1`, `2`...):
* `reference{N}.png` (or `ref_{N}.png` / `reference_{N}.png`) — Reference image
* `target{N}.png` (or `tgt_{N}.png` / `target_{N}.png`) — Target image
* `pair_{N}.json` (or `homography_{N}.json`) — Metadata containing 3x3 Ground-Truth Homography matrix $H_{gt}$

### Ground Truth JSON Format (`pair_0.json`)
```json
{
  "pair_id": "0",
  "reference": "reference0.png",
  "target": "target0.png",
  "homography": [
    [1.0024, -0.0152, 12.4500],
    [0.0148,  0.9981, -8.3200],
    [0.0000,  0.0000,  1.0000]
  ]
}
```

> **Note on Image Dimensions**: The PyTorch DataLoader automatically resizes reference and target images to dimensions divisible by 8 (e.g. 480x640) and scales $H_{gt}$ accordingly, matching LoFTR and RoMa 1/8th feature map resolution requirements.

---

## 3. Fine-Tuning Matchers

Fine-tuning is triggered using Python's module execution syntax:

### Command Line Interface (CLI)

```bash
# Fine-tune EfficientLoFTR (50 epochs, batch size 2 for 6GB GPU safety)
python -m finetuningiirscrossmodal.training --model eloftr --data-dir data/rendered_pairs --epochs 50 --batch-size 2

# Fine-tune RoMa (50 epochs, batch size 2 for 6GB GPU safety)
python -m finetuningiirscrossmodal.training --model roma --data-dir data/rendered_pairs --epochs 50 --batch-size 2

# Fine-tune BOTH models sequentially
python -m finetuningiirscrossmodal.training --model both --data-dir data/rendered_pairs --epochs 50 --batch-size 2
```

### CLI Arguments Summary

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | *(Required)* | Target model to fine-tune: `eloftr`, `roma`, or `both`. |
| `--data-dir` | `data/rendered_pairs` | Path to rendered image pairs and ground truth JSON files. |
| `--checkpoint-dir` | `checkpoints` | Destination directory for `.pth` model checkpoints. |
| `--epochs` | `50` | Total training epochs (with early stopping patience = 10). |
| `--batch-size` | `2` | Batch size per GPU iteration. **Do not exceed 2 for 6GB VRAM GPUs**. |
| `--lr` | `1e-4` | Initial learning rate (uses Cosine Annealing scheduler). |
| `--device` | `None` | Computation device (`cuda` or `cpu`). Auto-detected if unassigned. |

---

## 4. Architectural Details of Fine-Tuned Models

### A. EfficientLoFTR Fine-Tuning Strategy
* **Frozen Backbone**: The CNN feature extraction backbone is kept frozen to preserve low-level spatial features.
* **Trainable Transformer**: The positional self-attention and cross-attention transformer layers are fine-tuned using Dual-Softmax loss against ground-truth correspondences $H_{gt}$.
* **Saved Checkpoint**: `checkpoints/finetuned_eloftr_best.pth` (~45 MB).

### B. RoMa (Rotation-Robust Matcher) Fine-Tuning Strategy
* **Frozen DINOv2 Backbone**: RoMa relies on a DINOv2 Vision Transformer backbone. DINOv2 parameters are kept **strictly frozen** during training. This retains powerful pre-trained visual representations and ensures low memory usage.
* **Trainable Decoder & ConvNet Head**: Only the coarse-to-fine dense decoder head and ConvNet refinement layers are updated using Huber/Smooth L1 regression loss against $H_{gt}$.
* **Saved Checkpoint**: `checkpoints/finetuned_roma_best.pth` (~90 MB).

---

## 5. Cross-Modal Registration with MatchAnything

### What is MatchAnything?
MatchAnything is a foundation feature matching model built on EfficientLoFTR architecture (`zju-community/matchanything_eloftr`) fine-tuned across diverse domain images.

### Why MatchAnything for IIRS?
IIRS (Imaging Infrared Spectrometer) hyperspectral images have significant radiometric, intensity, and spectral contrast differences when paired with optical cameras (WAC/NAC). Traditional descriptors (SIFT, AKAZE) often fail due to gradient inversion across bands. MatchAnything's cross-modal pre-training enables robust feature correspondences across different spectral domains.

### Specifications
* **Weights Size**: **~28 MB**
* **Target Sensor Pairs**: `IIRS ↔ WAC`, `DFSAR ↔ SELENE`
* **Benchmark Comparison**: Although MatchAnything is primarily designed for cross-modal tasks, it is also executed on same-modality pairs (`OHRC ↔ NAC`) in the benchmark to evaluate how foundation cross-modal matchers perform against single-modality baselines.

---

## 6. Using Fine-Tuned Models in Python Code

You can directly instantiate and use any method adapter in your own scripts:

```python
from finetuningiirscrossmodal.methods import MatchAnythingMethod, FineTunedLoFTRMethod, FineTunedRoMaMethod
from types import SimpleNamespace
import cv2

# Define benchmark configuration
config = SimpleNamespace(
    ransac_reproj_threshold=3.0,
    image_width=640,
    image_height=480
)

# Load reference and target images
ref_img = cv2.imread("data/rendered_pairs/reference0.png")
tgt_img = cv2.imread("data/rendered_pairs/target0.png")

# 1. Run MatchAnything Adapter
matcher_ma = MatchAnythingMethod()
result_ma = matcher_ma.run(ref_img, tgt_img, config, pair_id="pair_0")
print(f"MatchAnything Success: {result_ma.success}, Inliers: {result_ma.num_inliers}, RMSE: {result_ma.rmse}")

# 2. Run Fine-Tuned EfficientLoFTR Adapter
matcher_loftr = FineTunedLoFTRMethod(checkpoint_path="checkpoints/finetuned_eloftr_best.pth")
result_loftr = matcher_loftr.run(ref_img, tgt_img, config, pair_id="pair_0")
print(f"FineTuned LoFTR Success: {result_loftr.success}, Inliers: {result_loftr.num_inliers}")

# 3. Run Fine-Tuned RoMa Adapter
matcher_roma = FineTunedRoMaMethod(checkpoint_path="checkpoints/finetuned_roma_best.pth")
result_roma = matcher_roma.run(ref_img, tgt_img, config, pair_id="pair_0")
print(f"FineTuned RoMa Success: {result_roma.success}, Inliers: {result_roma.num_inliers}")
```

---

## 7. Troubleshooting & Hardware FAQs

#### Q1: "CUDA Out of Memory" during fine-tuning on 6GB GPU?
* **Solution**: Ensure `--batch-size 2` is used. If memory remains tight, reduce `--img-size` in `configs/training.yaml` from `[480, 640]` to `[384, 512]`.

#### Q2: What happens if fine-tuning is executed before rendered pairs exist?
* **Solution**: The trainer detects `len(dataset) == 0`, prints a warning with instructions to populate `data/rendered_pairs/`, and exits gracefully without throwing an exception.

#### Q3: How do I test the code without GPUs?
* **Solution**: The codebase includes CPU fallback logic. All unit tests (`pytest finetuningiirscrossmodal/tests`) run on CPU in under 3 seconds.
