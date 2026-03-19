"""
split_dataset.py

The Kaggle chest x-ray dataset ships with only 8 validation images (4 per
class) — completely useless for measuring anything. This script moves 20% of
the training images into val, stratified by class so the ratio stays balanced.

Run once before agent.py. Safe to re-run — skips if val already looks right.

Usage:
    python split_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

from config import DATA_PATH

SPLIT_RATIO  = 0.20   # fraction of training data to move to val
RANDOM_SEED  = 42
CLASSES      = ["NORMAL", "PNEUMONIA"]


def count(split: str, cls: str) -> int:
    return len(list((DATA_PATH / split / cls).glob("*")))


def main() -> None:
    random.seed(RANDOM_SEED)

    print(f"Dataset root: {DATA_PATH}\n")

    # ── sanity check before doing anything ───────────────────────────────────
    for cls in CLASSES:
        train_dir = DATA_PATH / "train" / cls
        val_dir   = DATA_PATH / "val"   / cls
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing: {train_dir}")
        val_dir.mkdir(parents=True, exist_ok=True)

    print("Before split:")
    for cls in CLASSES:
        print(f"  train/{cls}: {count('train', cls)}  val/{cls}: {count('val', cls)}")

    # if val already has a reasonable number of images, don't touch anything
    min_val = min(count("val", cls) for cls in CLASSES)
    if min_val >= 100:
        print(f"\nVal set already looks reasonable ({min_val}+ images per class). Skipping.")
        return

    # ── move images ───────────────────────────────────────────────────────────
    total_moved = 0
    for cls in CLASSES:
        train_dir = DATA_PATH / "train" / cls
        val_dir   = DATA_PATH / "val"   / cls

        images = [
            p for p in train_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        random.shuffle(images)

        n_move = int(len(images) * SPLIT_RATIO)
        to_move = images[:n_move]

        for img in to_move:
            shutil.move(str(img), str(val_dir / img.name))

        print(f"\n  {cls}: moved {n_move} images to val")
        total_moved += n_move

    print(f"\nTotal moved: {total_moved}")

    print("\nAfter split:")
    for cls in CLASSES:
        print(f"  train/{cls}: {count('train', cls)}  val/{cls}: {count('val', cls)}")

    print("\nDone. You can now run python agent.py")


if __name__ == "__main__":
    main()