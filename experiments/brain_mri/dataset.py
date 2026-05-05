import os
from glob import glob
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    RandAdjustContrastD,
    RandAffineD,
    RandFlipD,
    RandGaussianNoiseD,
    RandScaleIntensityD,
    ToTensorD,
)


class SliceDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        file_list: Optional[Sequence[str]] = None,
        file_ext: str = ".png",
    ):
        super().__init__()
        self.file_ext = file_ext
        if file_list is None:
            self.files = sorted(glob(os.path.join(root_dir, f"*{file_ext}")))
        else:
            self.files = sorted(list(file_list))
        self.transform = transform
        if len(self.files) == 0:
            raise RuntimeError(f"No {file_ext} slices found in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        try:
            if self.file_ext.lower() == ".npz":
                with np.load(path) as data:
                    if "arr" not in data:
                        raise KeyError(f"{path} missing key 'arr' (has {data.files})")
                    arr = data["arr"].astype(np.float32)
            else:
                with Image.open(path) as img:
                    arr = np.asarray(img.convert("F"), dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read slice from {path}") from e

        if arr.ndim < 2:
            raise RuntimeError(f"Loaded array with ndim={arr.ndim} from {path}; expected >=2")

        # Keep orientation as saved in PNGs (no rotation)
        arr = arr.copy()
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
        data_dir: Optional[str] = None,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        val_split: float = 0.1,
        augment: bool = False,
        seed: int = 42,
        file_ext: str = ".png",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augment = augment
        self.seed = seed
        self.file_ext = file_ext

    def setup(self, stage: Optional[str] = None):
        # Pipeline: rotation is done in __getitem__ via np.rot90 to avoid MONAI Rotate90d axis issues
        base_preproc = [
            EnsureChannelFirstD(keys=["MRI_image"], channel_dim="no_channel"),
            ToTensorD(keys=["MRI_image"]),
        ]

        aug = []
        if self.augment:
            rotation_rad = 0.2618  # ~15 degrees
            translate_pix = 15
            zoom_min, zoom_max = 0.8, 1.2
            aug.extend([
                RandScaleIntensityD(keys=["MRI_image"], factors=0.10, prob=0.33),
                RandAdjustContrastD(keys=["MRI_image"], gamma=(0.5, 1.5), prob=0.33),
                RandGaussianNoiseD(keys=["MRI_image"], prob=0.50, mean=0.0, std=0.30),
                RandAffineD(
                    keys=["MRI_image"],
                    prob=0.33,
                    rotate_range=(-rotation_rad, rotation_rad),
                    translate_range=(translate_pix, translate_pix),
                    scale_range=(zoom_min - 1.0, zoom_max - 1.0),
                    padding_mode="border",
                ),
                RandFlipD(keys=["MRI_image"], prob=0.5, spatial_axis=1),
            ])

        train_transform = Compose(base_preproc + aug)
        val_transform = Compose(base_preproc)

        if self.train_dir and self.val_dir:
            train_files = sorted(glob(os.path.join(self.train_dir, f"*{self.file_ext}")))
            val_files = sorted(glob(os.path.join(self.val_dir, f"*{self.file_ext}")))
        else:
            full_files = sorted(glob(os.path.join(self.data_dir, f"*{self.file_ext}")))
            full_files = [f for f in full_files if "_slice_" in os.path.basename(f)]

        def _filter_valid(paths: list[str]) -> list[str]:
            valid = []
            invalid = []
            for path in paths:
                try:
                    if self.file_ext.lower() == ".npz":
                        with np.load(path) as data:
                            arr = data["arr"]
                    else:
                        with Image.open(path) as img:
                            arr = np.asarray(img.convert("F"))
                except Exception as e:
                    invalid.append((path, f"load_error: {e}"))
                    continue

                if arr.ndim < 2:
                    invalid.append((path, f"ndim={arr.ndim}"))
                    continue

                if 0 in arr.shape:
                    invalid.append((path, f"empty_shape={arr.shape}"))
                    continue

                valid.append(path)

            if invalid:
                print(f"Filtered out {len(invalid)} invalid slices before split")
                for path, reason in invalid[:10]:
                    print(f"  - {path}: {reason}")
                if len(invalid) > 10:
                    print(f"  ... {len(invalid) - 10} more invalid files suppressed")
            return valid

        if self.train_dir and self.val_dir:
            train_files = _filter_valid(train_files)
            val_files = _filter_valid(val_files)
        else:
            full_files = _filter_valid(full_files)
            if len(full_files) == 0:
                raise RuntimeError(
                    f"No valid {self.file_ext} slices found in {self.data_dir} "
                    f"(expected *_slice_*{self.file_ext} with ndim>=2)"
                )

            val_size = int(len(full_files) * self.val_split)
            train_size = len(full_files) - val_size
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(full_files), generator=generator).tolist()
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]

            train_files = [full_files[i] for i in train_idx]
            val_files = [full_files[i] for i in val_idx]

        self.train_ds = SliceDataset(
            self.train_dir or self.data_dir,
            transform=train_transform,
            file_list=train_files,
            file_ext=self.file_ext,
        )
        self.val_ds = SliceDataset(
            self.val_dir or self.data_dir,
            transform=val_transform,
            file_list=val_files,
            file_ext=self.file_ext,
        )

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
