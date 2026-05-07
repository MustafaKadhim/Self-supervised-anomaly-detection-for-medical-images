import os
from glob import glob
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    RandFlipD,
    RandRotateD,
    Resized,
    CenterSpatialCropd,
    ToTensorD,
)


class NpySliceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, file_list: Optional[Sequence[str]] = None):
        super().__init__()
        if file_list is None:
            self.files = sorted(glob(os.path.join(root_dir, "*.npy")))
        else:
            self.files = sorted(list(file_list))
        self.transform = transform
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy slices found in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim < 2:
            raise RuntimeError(f"Loaded array with ndim={arr.ndim} from {path}; expected >=2")
        # Rotate 90° CCW to match SOTA Stage 1 orientation; copy to avoid negative strides
        arr = np.rot90(arr, k=-1).copy()
        sample = {"MRI_image": arr}
        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as e:
                raise RuntimeError(f"Transform failed for {path} with shape {arr.shape}") from e
        image = sample["MRI_image"]
        if image.dim() == 2:
            image = image.unsqueeze(0)
        return {"image": image, "path": path}


class SliceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split: float = 0.1,
        augment: bool = False,
        seed: int = 42,
        image_size: Sequence[int] = (256, 256),
        resize_size: Sequence[int] = (320, 320),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augment = augment
        self.seed = seed
        self.image_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.resize_size = tuple(resize_size) if isinstance(resize_size, (list, tuple)) else (resize_size, resize_size)

    def setup(self, stage: Optional[str] = None):
        # Pipeline: rotation is done in __getitem__ via np.rot90 to avoid MONAI Rotate90d axis issues
        base_preproc = [
            EnsureChannelFirstD(keys=["MRI_image"], channel_dim="no_channel"),
            Resized(keys=["MRI_image"], spatial_size=self.resize_size, mode="area"),
            CenterSpatialCropd(keys=["MRI_image"], roi_size=self.image_size),
            ToTensorD(keys=["MRI_image"]),
        ]

        aug = []
        if self.augment:
            aug.extend([
                RandFlipD(keys=["MRI_image"], prob=0.5, spatial_axis=2),  # horizontal flip on (C,H,W)
                RandRotateD(keys=["MRI_image"], range_x=0.0873, prob=0.3, keep_size=True),  # ~5 degrees
            ])

        train_transform = Compose(base_preproc + aug)
        val_transform = Compose(base_preproc)

        full_files = sorted(glob(os.path.join(self.data_dir, "*.npy")))
        full_files = [f for f in full_files if "_slice_" in os.path.basename(f)]

        valid_files = []
        invalid_files = []

        for path in full_files:
            try:
                arr = np.load(path, mmap_mode="r")
            except Exception as e:
                invalid_files.append((path, f"load_error: {e}"))
                continue

            if arr.ndim < 2:
                invalid_files.append((path, f"ndim={arr.ndim}"))
                continue

            if 0 in arr.shape:
                invalid_files.append((path, f"empty_shape={arr.shape}"))
                continue

            valid_files.append(path)

        if invalid_files:
            print(f"Filtered out {len(invalid_files)} invalid slices before split")
            for path, reason in invalid_files[:10]:
                print(f"  - {path}: {reason}")
            if len(invalid_files) > 10:
                print(f"  ... {len(invalid_files) - 10} more invalid files suppressed")

        full_files = valid_files

        if len(full_files) == 0:
            raise RuntimeError(f"No valid .npy slices found in {self.data_dir} (expected *_slice_*.npy with ndim>=2)")

        val_size = int(len(full_files) * self.val_split)
        train_size = len(full_files) - val_size
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(full_files), generator=generator).tolist()
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        train_files = [full_files[i] for i in train_idx]
        val_files = [full_files[i] for i in val_idx]

        self.train_ds = NpySliceDataset(self.data_dir, transform=train_transform, file_list=train_files)
        self.val_ds = NpySliceDataset(self.data_dir, transform=val_transform, file_list=val_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

