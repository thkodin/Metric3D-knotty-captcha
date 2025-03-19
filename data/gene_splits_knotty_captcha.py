"""Split a dataset of blender renders into train/val/test splits for fine-tuning models."""

import random
import shutil
from pathlib import Path

from natsort import natsorted

# The root data for images is the directory containing the images.
REPO_ROOT = Path(__file__).parents[1]
DIR_DATA_ROOT = REPO_ROOT / "data" / "knotty_captcha"

FP_ERROR_TOLERANCE = 1e-6

# Split settings.
SPLIT_PROPORTIONS = (0.8, 0.1, 0.1)
SHUFFLE = True
SEED = 42  # set to None to disable


def split_dataset(
    base_dir: Path,
    split_proportions: tuple[float, float, float],
    shuffle: bool = True,
    seed: int = None,
):
    """Split the dataset into train/val/test splits.

    Args:
        base_dir: The base directory containing the color, depth, and normal image folders.
        split_proportions: The proportions of the dataset to split into train/val/test.
        shuffle: Whether to shuffle the dataset.
        seed: The seed to use for shuffling the dataset. If None, no shuffling is done.
    """
    if seed is not None:
        random.seed(seed)

    def move_images(indices: list[int], split: str):
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            (split_dir / "color").mkdir(parents=True, exist_ok=True)
            (split_dir / "depth").mkdir(parents=True, exist_ok=True)
            shutil.move(color_images[idx], split_dir / "color" / color_images[idx].name)
            shutil.move(depth_images[idx], split_dir / "depth" / depth_images[idx].name)
            if normal_images:
                (split_dir / "normal").mkdir(parents=True, exist_ok=True)
                shutil.move(normal_images[idx], split_dir / "normal" / normal_images[idx].name)

    color_dir = base_dir / "color"
    depth_dir = base_dir / "depth"
    normal_dir = base_dir / "normal"

    if not color_dir.exists() or not color_dir.glob("*.png"):
        raise ValueError("Color directory does not exist or is empty.")
    if not depth_dir.exists() or not depth_dir.glob("*.png"):
        raise ValueError("Depth directory does not exist or is empty.")

    color_images = natsorted(color_dir.glob("*.png"))
    depth_images = natsorted(depth_dir.glob("*.png"))
    normal_images = natsorted(normal_dir.glob("*.png"))

    if len(color_images) != len(depth_images):
        raise ValueError("Color and depth images count mismatch.")
    if normal_images and len(color_images) != len(normal_images):
        raise ValueError("Color and normal images count mismatch.")

    total_images = len(color_images)
    indices = list(range(total_images))

    if shuffle:
        random.shuffle(indices)

    train_split = int(split_proportions[0] * total_images)
    val_split = train_split + int(split_proportions[1] * total_images)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    move_images(train_indices, "train")
    move_images(val_indices, "val")
    move_images(test_indices, "test")

    shutil.rmtree(color_dir)  # Remove the original color directory.
    shutil.rmtree(depth_dir)  # Remove the original depth directory.
    if normal_dir.exists():
        shutil.rmtree(normal_dir)  # Remove the original normal directory if it exists.


def main():
    # This is the directory containing the color, depth, and optionally normal image folders.
    if not abs(1.0 - sum(SPLIT_PROPORTIONS)) < FP_ERROR_TOLERANCE:
        raise ValueError(f"Split proportions must sum to 1 within float-point error +-{FP_ERROR_TOLERANCE}.")

    split_dataset(
        base_dir=DIR_DATA_ROOT,
        split_proportions=SPLIT_PROPORTIONS,
        shuffle=SHUFFLE,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
