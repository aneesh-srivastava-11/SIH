"""
Rendered Pairs PyTorch Dataset for Lunar Image Matcher Fine-Tuning.

Expects image pairs generated from DEM rendering pipelines:
  - reference{n}.png / target{n}.png
  - pair_{n}.json (or homography_{n}.json / homography_{n}.npy) containing ground truth 3x3 homography matrix H_gt

Dimensions are automatically resized to be divisible by 8 (required for LoFTR 1/8th feature maps).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RenderedPairsDataset(Dataset):
    """
    PyTorch Dataset for rendered lunar image pairs with Ground Truth Homography.

    Args:
        data_dir: Directory containing pair images and ground truth JSON files.
        img_size: Tuple (height, width) to resize images. Default: (480, 640).
                  Both height and width must be divisible by 8.
        transform: Optional torchvision or custom augmentations.
        split: 'train', 'val', or 'all' split.
        split_ratio: Ratio for train split (e.g. 0.8).
        seed: Random seed for deterministic train/val splitting.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (480, 640),
        transform: Optional[Any] = None,
        split: str = "all",
        split_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        self.split = split

        # Ensure image dimensions are divisible by 8 (required for LoFTR/RoMa)
        h, w = self.img_size
        if h % 8 != 0 or w % 8 != 0:
            h_adj = ((h + 7) // 8) * 8
            w_adj = ((w + 7) // 8) * 8
            self.img_size = (h_adj, w_adj)

        self.pairs = self._discover_pairs()

        # Perform train/val split if requested
        if split in ("train", "val") and len(self.pairs) > 0:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(self.pairs))
            split_idx = int(len(indices) * split_ratio)

            if split == "train":
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]

            self.pairs = [self.pairs[i] for i in selected_indices]

    def _discover_pairs(self) -> List[Dict[str, Any]]:
        """
        Scans data_dir for rendered pairs matching patterns:
        - reference{n}.png / target{n}.png + pair_{n}.json
        - ref_{n}.png / tgt_{n}.png + pair_{n}.json
        - pair_{n}_ref.png / pair_{n}_tgt.png + pair_{n}.json
        """
        pairs = []
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            return pairs

        # Search for pair JSON metadata files
        json_files = list(self.data_dir.glob("pair_*.json")) + list(self.data_dir.glob("homography_*.json"))
        
        for jf in sorted(json_files):
            stem = jf.stem
            # Determine pair ID or index
            pair_id = stem.replace("pair_", "").replace("homography_", "")

            # Look for matching images
            ref_candidates = [
                self.data_dir / f"reference{pair_id}.png",
                self.data_dir / f"reference_{pair_id}.png",
                self.data_dir / f"ref_{pair_id}.png",
                self.data_dir / f"pair_{pair_id}_ref.png",
                self.data_dir / f"reference{pair_id}.jpg",
                self.data_dir / f"reference_{pair_id}.tif",
            ]
            tgt_candidates = [
                self.data_dir / f"target{pair_id}.png",
                self.data_dir / f"target_{pair_id}.png",
                self.data_dir / f"tgt_{pair_id}.png",
                self.data_dir / f"pair_{pair_id}_tgt.png",
                self.data_dir / f"target{pair_id}.jpg",
                self.data_dir / f"target_{pair_id}.tif",
            ]

            ref_path = next((p for p in ref_candidates if p.exists()), None)
            tgt_path = next((p for p in tgt_candidates if p.exists()), None)

            if ref_path and tgt_path:
                pairs.append({
                    "pair_id": pair_id,
                    "ref_path": ref_path,
                    "tgt_path": tgt_path,
                    "json_path": jf,
                })

        # Fallback: if no JSONs, pair up reference{n}.png and target{n}.png directly
        if not pairs:
            ref_imgs = sorted(list(self.data_dir.glob("reference*.png")) + list(self.data_dir.glob("ref_*.png")))
            for ref_p in ref_imgs:
                # Find matching target
                name = ref_p.name
                tgt_name = name.replace("reference", "target").replace("ref_", "tgt_")
                tgt_p = ref_p.parent / tgt_name
                if tgt_p.exists():
                    pair_id = ref_p.stem.replace("reference", "").replace("ref_", "")
                    json_p = ref_p.parent / f"pair_{pair_id}.json"
                    pairs.append({
                        "pair_id": pair_id,
                        "ref_path": ref_p,
                        "tgt_path": tgt_p,
                        "json_path": json_p if json_p.exists() else None,
                    })

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        ref_path = str(item["ref_path"])
        tgt_path = str(item["tgt_path"])

        # Load images (grayscale for LoFTR/RoMa feature extraction)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        tgt_img = cv2.imread(tgt_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or tgt_img is None:
            raise ValueError(f"Failed to load image pair: {ref_path}, {tgt_path}")

        orig_h_ref, orig_w_ref = ref_img.shape[:2]
        orig_h_tgt, orig_w_tgt = tgt_img.shape[:2]

        target_h, target_w = self.img_size

        # Resize images
        ref_resized = cv2.resize(ref_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tgt_resized = cv2.resize(tgt_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Scale factors for adjusting ground truth homography to resized dimensions
        scale_x_ref = target_w / orig_w_ref
        scale_y_ref = target_h / orig_h_ref
        scale_x_tgt = target_w / orig_w_tgt
        scale_y_tgt = target_h / orig_h_tgt

        # Load Ground Truth Homography Matrix (3x3)
        h_gt = np.eye(3, dtype=np.float32)
        if item["json_path"] and item["json_path"].exists():
            try:
                with open(item["json_path"], "r") as f:
                    meta = json.load(f)
                if "homography" in meta:
                    h_gt = np.array(meta["homography"], dtype=np.float32)
                elif "H" in meta:
                    h_gt = np.array(meta["H"], dtype=np.float32)
            except Exception:
                pass

        # Adjust homography for resized image dimensions
        # H_scaled = S_tgt * H_orig * S_ref^-1
        S_ref = np.array([
            [scale_x_ref, 0, 0],
            [0, scale_y_ref, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        S_tgt = np.array([
            [scale_x_tgt, 0, 0],
            [0, scale_y_tgt, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        h_gt_scaled = S_tgt @ h_gt @ np.linalg.inv(S_ref)

        # Convert to PyTorch Tensors [1, H, W] normalized to [0, 1]
        ref_tensor = torch.from_numpy(ref_resized).float().unsqueeze(0) / 255.0
        tgt_tensor = torch.from_numpy(tgt_resized).float().unsqueeze(0) / 255.0
        h_gt_tensor = torch.from_numpy(h_gt_scaled).float()

        return {
            "pair_id": item["pair_id"],
            "image0": ref_tensor,            # [1, H, W] reference image
            "image1": tgt_tensor,            # [1, H, W] target image
            "homography": h_gt_tensor,       # [3, 3] scaled GT homography
            "ref_path": ref_path,
            "tgt_path": tgt_path,
        }


def get_dataloader(
    data_dir: str,
    batch_size: int = 2,
    img_size: Tuple[int, int] = (480, 640),
    split: str = "train",
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Utility function to create a PyTorch DataLoader for rendered pairs."""
    dataset = RenderedPairsDataset(
        data_dir=data_dir,
        img_size=img_size,
        split=split,
        seed=seed,
    )
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
