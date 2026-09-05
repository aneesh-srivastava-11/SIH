# Fine-Tuning & Cross-Modal Module (`finetuningiirscrossmodal`)

A specialized modular pipeline for fine-tuning deep feature matchers (**EfficientLoFTR**, **RoMa**) on rendered lunar image pairs, and integrating **MatchAnything** for cross-modal (e.g., IIRS hyperspectral ↔ optical WAC) image registration.

This repository is built as a **sibling folder** to `basebenchmarking/` under the main workspace.

---

## Technical Overview & Architecture

### 1. Fine-Tuning Sub-System
* **Target Models**:
  * **EfficientLoFTR**: Fine-tunes Transformer self-attention and cross-attention positional layers while keeping the CNN backbone frozen.
  * **RoMa (Rotation-Robust Matcher)**: Keeps the **DINOv2** Vision Transformer backbone frozen (to preserve rich, scale/rotation invariant representation) and trains only the dense decoder network and fine ConvNet refinement layers.
* **Supervision**: Homography-based loss computed directly from rendered lunar pairs ($H_{gt}$ ground-truth 3x3 matrices).
* **Memory & Hardware Optimization**: Configured by default with `batch_size = 2` to run safely within **8GB RAM / 6GB GPU VRAM** hardware constraints.

### 2. MatchAnything Cross-Modal Sub-System
* **Model**: `zju-community/matchanything_eloftr` (HuggingFace).
* **Weight Size**: **~28MB** download size.
* **Primary Target**: IIRS (Imaging Infrared Spectrometer) cross-modal pairs (IIRS ↔ WAC, DFSAR ↔ SELENE).
* **Benchmarking Strategy**: MatchAnything is also enabled for same-modality pairs (OHRC ↔ NAC) to compare performance against traditional and single-modality deep learning baselines.

---

## File & Directory Structure

```
finetuningiirscrossmodal/
├── README.md                      # Architecture & overview (this file)
├── USAGE.md                       # Comprehensive step-by-step usage guide
├── requirements.txt               # Core dependencies
├── requirements-optional.txt      # Deep learning libraries (kornia, romatch, etc.)
├── configs/
│   ├── default.yaml               # Registration method registry settings
│   └── training.yaml              # Training hyperparameters (batch_size=2, lr=1e-4)
├── checkpoints/                   # Output directory for fine-tuned checkpoints (.pth)
├── training/
│   ├── __init__.py                # Package exports
│   ├── __main__.py                # CLI runner (python -m finetuningiirscrossmodal.training)
│   ├── dataset.py                 # PyTorch RenderedPairsDataset (auto-resizes to div by 8)
│   ├── trainer.py                 # Base Trainer with AdamW + Cosine Annealing & Early Stopping
│   ├── train_eloftr.py            # EfficientLoFTR fine-tuning module
│   └── train_roma.py              # RoMa fine-tuning module
├── methods/
│   ├── __init__.py                # Method exports
│   ├── base.py                    # RegistrationMethod ABC & RegistrationResult schema
│   ├── matchanything.py           # MatchAnything-ELoFTR adapter
│   ├── finetuned_loftr.py         # Fine-Tuned EfficientLoFTR adapter
│   └── finetuned_roma.py          # Fine-Tuned RoMa adapter
└── tests/
    ├── test_dataset.py            # Dataset unit tests
    ├── test_adapters.py           # Method adapter unit tests
    └── test_training.py           # Training step unit tests
```

---

## Hardware Requirements & Memory Specifications

| Component | Target System Spec | Recommended Setting |
| :--- | :--- | :--- |
| **System RAM** | 8 GB RAM | PyTorch dataloader with `num_workers = 0` |
| **GPU VRAM** | 6 GB VRAM (NVIDIA CUDA) | `batch_size = 2` |
| **Storage** | ~2 GB free disk space | Model checkpoints & rendered data |

---

## Model Weights & Checkpoint File Sizes

| Model | Weight File / Source | Size | Usage Context |
| :--- | :--- | :--- | :--- |
| **MatchAnything** | `zju-community/matchanything_eloftr` | **~28 MB** | Pretrained cross-modal matching |
| **Fine-Tuned EfficientLoFTR** | `checkpoints/finetuned_eloftr_best.pth` | **~45 MB** | Fine-tuned lunar checkpoint |
| **Fine-Tuned RoMa** | `checkpoints/finetuned_roma_best.pth` | **~90 MB** | Fine-tuned dense decoder checkpoint |

---

## Integration with Base Benchmarking

Method adapters in `methods/` adhere strictly to the `RegistrationMethod` abstract base class used in `basebenchmarking`. They return standard `RegistrationResult` dataclasses with metrics:
* Keypoints extracted & matched
* Inlier count & Inlier ratio
* Estimated 3x3 Homography Matrix
* RMSE & Reprojection Error (Mean, Median, Max)
* Execution runtime & Device details

To plug these methods into `basebenchmarking`, simply import the adapter classes from `finetuningiirscrossmodal.methods` into the registry.
