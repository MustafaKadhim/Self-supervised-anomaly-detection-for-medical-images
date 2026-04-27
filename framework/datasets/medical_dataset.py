"""
Medical image dataset utilities for anomaly detection.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """Load an image from disk as a normalised float32 numpy array."""
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


class MedicalImageDataset(Dataset):
    """
    Generic dataset for medical images supporting train/test splits.

    Expected directory layout::

        root/
          train/
            normal/   ← healthy images used for training
          test/
            normal/   ← healthy test images  (label = 0)
            anomaly/  ← anomalous test images (label = 1)

    Args:
        root (str | Path): Path to the dataset root directory.
        split (str): One of ``'train'`` or ``'test'``.
        transform (callable, optional): Transforms applied to each image
            (should accept a PIL Image and return a ``torch.Tensor``).
        grayscale (bool): Load images as grayscale (single channel).
        image_size (int): Resize images to ``(image_size, image_size)``.
        extensions (tuple): Accepted file extensions.
    """

    SPLITS = ("train", "test")

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        grayscale: bool = True,
        image_size: int = 128,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".bmp"),
    ):
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.grayscale = grayscale
        self.image_size = image_size

        self.images: List[str] = []
        self.labels: List[int] = []
        self._load_index(extensions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_index(self, extensions: Tuple[str, ...]) -> None:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for label_name, label_int in [("normal", 0), ("anomaly", 1)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() in extensions:
                    self.images.append(str(f))
                    self.labels.append(label_int)

        if len(self.images) == 0:
            raise RuntimeError(f"No images found under {split_dir}")

    def _default_transform(self, img: np.ndarray) -> torch.Tensor:
        """Resize and convert to tensor when no custom transform is given."""
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor.permute(2, 0, 1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = _load_image(self.images[idx], grayscale=self.grayscale)
        label = self.labels[idx]

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = self._default_transform(img)

        return img_tensor, label

    @property
    def num_normal(self) -> int:
        return self.labels.count(0)

    @property
    def num_anomaly(self) -> int:
        return self.labels.count(1)

    def __repr__(self) -> str:
        return (
            f"MedicalImageDataset(split='{self.split}', "
            f"normal={self.num_normal}, anomaly={self.num_anomaly})"
        )
