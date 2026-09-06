"""
CLI Entry Point for Matcher Fine-Tuning.

Usage:
  python -m finetuningiirscrossmodal.training --model eloftr --data-dir data/rendered_pairs --epochs 50 --batch-size 2
  python -m finetuningiirscrossmodal.training --model roma --data-dir data/rendered_pairs --epochs 50 --batch-size 2
"""

import argparse
import sys
from pathlib import Path

from finetuningiirscrossmodal.training.train_eloftr import train_eloftr
from finetuningiirscrossmodal.training.train_roma import train_roma


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientLoFTR or RoMa on rendered lunar image pairs."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["eloftr", "roma", "both"],
        help="Target matcher model to fine-tune ('eloftr', 'roma', or 'both')."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/rendered_pairs",
        help="Path to directory containing rendered reference/target pairs and GT JSON homographies."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory where output .pth checkpoints will be saved."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU iteration (default: 2 for 6GB GPU memory target)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4 with Cosine Annealing)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda' or 'cpu'). Auto-detected if unspecified."
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"Created data directory at '{data_path}'.")
        print("Please place your rendered pair images (reference{n}.png, target{n}.png) and ground-truth JSONs here.")

    if args.model in ("eloftr", "both"):
        print("\n=== Initiating EfficientLoFTR Fine-Tuning Task ===")
        res_eloftr = train_eloftr(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        print(f"EfficientLoFTR Fine-Tuning Status: {res_eloftr.get('status')}")

    if args.model in ("roma", "both"):
        print("\n=== Initiating RoMa Fine-Tuning Task ===")
        res_roma = train_roma(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        print(f"RoMa Fine-Tuning Status: {res_roma.get('status')}")


if __name__ == "__main__":
    main()
