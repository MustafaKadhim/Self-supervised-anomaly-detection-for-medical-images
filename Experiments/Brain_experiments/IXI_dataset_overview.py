#!/usr/bin/env python3
"""IXI dataset overview utility.

Reads NIfTI files in a folder, reports image shapes, and exports axial PNG slices.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import nibabel as nib
import imageio.v3 as iio
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an overview of IXI NIfTI volumes and save axial PNG slices."
    )
    parser.add_argument("--input-dir", type=Path, default="/mnt/md2/Musti_Phd_Projects/IXI-TI/", help="Directory containing .nii or .nii.gz files.")
    parser.add_argument("--output-dir", type=Path, default="/home/mluser1/Musti_Anomaly_Detection/IXI_Sample_Work", help="Directory for output PNGs and CSV summary.")
    parser.add_argument("--slice-index", type=int, default=-1, help="Axial slice index to export. Use -1 for the middle slice.")
    parser.add_argument("--volume-index", type=int, default=0, help="Volume index for 4D images (default: 0).")
    parser.add_argument("--pattern", type=str, default="*.nii.gz", help="Glob pattern for NIfTI files (default: *.nii.gz).")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of files processed (0 = all).")
    parser.add_argument("--training-slice-start", type=int, default=128, help="Start slice index for training PNGs (inclusive).")
    parser.add_argument("--training-slice-end", type=int, default=188, help="End slice index for training PNGs (inclusive).")
    
    parser.add_argument("--plot-slice-start", type=int, default=128, help="Start slice index for montage plot (inclusive).")
    parser.add_argument("--plot-slice-end", type=int, default=188, help="End slice index for montage plot (inclusive).")
    parser.add_argument("--plot-slice-count", type=int, default=12, help="Number of slices to show in montage.")
    
    parser.add_argument("--training-ready", action="store_true", help="Export training PNGs for a slice range.")
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=True, help="Save PNGs for debugging")
    parser.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=True, help="Save float32 arrays for training")
    parser.add_argument("--output-npy-dir", type=Path, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Training_samples_FastMRI_IXI", help="Directory for output float32 arrays.")
    parser.add_argument("--png-pattern", type=str, default="{file}_slice_{slice:03d}.png", help="PNG filename pattern using {file} and {slice} placeholders")
    parser.add_argument("--npy-pattern", type=str, default="{file}_slice_{slice:03d}.npz", help="NPZ filename pattern using {file} and {slice} placeholders")
    parser.add_argument(
        "--intensity-scale",
        type=str,
        default="none",
        choices=["none", "minus1_1", "zero1"],
        help="Scale clipped z-score to [-1,1], [0,1], or leave as-is",
    )
    parser.add_argument(
        "--z-clip",
        type=str,
        default="-3,3",
        help="Z-score clip bounds as 'min,max' (e.g., -3,3).",
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="Search for files recursively (default: enabled).")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive search.")
    return parser.parse_args()


def iter_nifti_files(input_dir: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from input_dir.rglob(pattern)
    else:
        yield from input_dir.glob(pattern)


def pick_slice_index(slices: int, slice_index: int) -> int:
    if slices <= 0:
        return 0
    if slice_index < 0:
        return slices // 2
    return int(np.clip(slice_index, 0, slices - 1))


def normalize_to_uint8(img2d: np.ndarray) -> np.ndarray:
    img2d = np.asarray(img2d, dtype=np.float32)
    finite_mask = np.isfinite(img2d)
    if not np.any(finite_mask):
        return np.zeros_like(img2d, dtype=np.uint8)
    vmin = np.nanmin(img2d)
    vmax = np.nanmax(img2d)
    if vmax <= vmin:
        return np.zeros_like(img2d, dtype=np.uint8)
    scaled = (img2d - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def normalize_to_uint8_with_range(img2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    img2d = np.asarray(img2d, dtype=np.float32)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return normalize_to_uint8(img2d)
    scaled = (img2d - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def center_crop_or_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]

    if h > target_h:
        top = (h - target_h) // 2
        arr = arr[top:top + target_h, :]
    if w > target_w:
        left = (w - target_w) // 2
        arr = arr[:, left:left + target_w]

    h, w = arr.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )


def pad_volume_inplane(volume: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if volume.ndim < 3:
        return center_crop_or_pad(volume, target_h, target_w)
    padded_slices = [center_crop_or_pad(volume[:, :, idx], target_h, target_w) for idx in range(volume.shape[2])]
    return np.stack(padded_slices, axis=2)


def extract_axial_slice(img: nib.Nifti1Image, slice_index: int, volume_index: int) -> Tuple[np.ndarray, int]:
    canon_img = nib.as_closest_canonical(img)
    dataobj = canon_img.dataobj
    shape = canon_img.shape
    if len(shape) == 3:
        z = shape[2]
        idx = pick_slice_index(z, slice_index)
        slice2d = np.asanyarray(dataobj[:, :, idx])
        return slice2d, idx
    if len(shape) >= 4:
        z = shape[2]
        idx = pick_slice_index(z, slice_index)
        vol = int(np.clip(volume_index, 0, shape[3] - 1))
        slice2d = np.asanyarray(dataobj[:, :, idx, vol])
        return slice2d, idx
    raise ValueError(f"Unsupported NIfTI shape: {shape}")


def extract_axial_slice_at(img: nib.Nifti1Image, slice_index: int, volume_index: int) -> np.ndarray:
    canon_img = nib.as_closest_canonical(img)
    dataobj = canon_img.dataobj
    shape = canon_img.shape
    if len(shape) == 3:
        return np.asanyarray(dataobj[:, :, slice_index])
    if len(shape) >= 4:
        vol = int(np.clip(volume_index, 0, shape[3] - 1))
        return np.asanyarray(dataobj[:, :, slice_index, vol])
    raise ValueError(f"Unsupported NIfTI shape: {shape}")


def parse_z_clip(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) == 1:
        bound = float(parts[0])
        return -abs(bound), abs(bound)
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError("--z-clip must be 'min,max' or a single number")


def build_file_id(nifti_path: Path) -> str:
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return nifti_path.stem


def get_training_slice_indices(total_slices: int, start_arg: int, end_arg: int) -> list[int]:
    if total_slices <= 0:
        return []
    start = max(min(start_arg, end_arg), 0)
    end = min(max(start_arg, end_arg), total_slices - 1)
    if end < start:
        return []
    return list(range(start, end + 1))


def apply_intensity_mapping(volume: np.ndarray, clip_value: tuple[float, float], scale_mode: str) -> np.ndarray:
    vol = volume.astype(np.float32)
    mean = float(np.mean(vol)) if vol.size else 0.0
    std = float(np.std(vol)) if vol.size else 1.0
    if std == 0.0:
        std = 1.0
    vol = (vol - mean) / std
    clip_min, clip_max = clip_value
    vol = np.clip(vol, clip_min, clip_max)

    if scale_mode == "minus1_1":
        scale = max(abs(clip_min), abs(clip_max)) or 1.0
        vol = vol / scale
    elif scale_mode == "zero1":
        scale = max(abs(clip_min), abs(clip_max)) or 1.0
        vol = (vol / scale + 1.0) / 2.0
    return vol.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    z_clip = parse_z_clip(args.z_clip)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    save_png_enabled = bool(args.save_png and args.max_samples != 0)
    if args.save_png and args.max_samples == 0:
        logging.info("PNG export disabled because --max-samples is 0 (all files mode).")

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        args.output_npy_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(iter_nifti_files(input_dir, args.pattern, args.recursive))
    if not files and not args.recursive:
        logging.info("No files found with non-recursive search. Retrying recursively.")
        files = sorted(iter_nifti_files(input_dir, args.pattern, True))
    if not files and args.pattern == "*.nii.gz":
        files = sorted(iter_nifti_files(input_dir, "*.nii", args.recursive))
        if not files and not args.recursive:
            files = sorted(iter_nifti_files(input_dir, "*.nii", True))

    if not files:
        logging.warning("No NIfTI files found in %s", input_dir)
        return

    if args.max_samples and args.max_samples > 0:
        files = files[:args.max_samples]

    csv_path = output_dir / "ixi_overview.csv"
    training_dir = output_dir / "IXI_PNGs"
    if args.training_ready and save_png_enabled:
        training_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "shape", "x", "y", "z", "t", "voxel_x", "voxel_y", "voxel_z", "voxel_t"])

        for nifti_path in files:
            try:
                img = nib.load(str(nifti_path))
            except Exception as exc:
                logging.warning("Failed to read %s: %s", nifti_path, exc)
                continue

            shape = img.shape
            x = shape[0] if len(shape) > 0 else 0
            y = shape[1] if len(shape) > 1 else 0
            z = shape[2] if len(shape) > 2 else 0
            t = shape[3] if len(shape) > 3 else 1
            zooms = img.header.get_zooms() if img.header is not None else ()
            voxel_x = float(zooms[0]) if len(zooms) > 0 else 0.0
            voxel_y = float(zooms[1]) if len(zooms) > 1 else 0.0
            voxel_z = float(zooms[2]) if len(zooms) > 2 else 0.0
            voxel_t = float(zooms[3]) if len(zooms) > 3 else 0.0
            writer.writerow([str(nifti_path), str(shape), x, y, z, t, voxel_x, voxel_y, voxel_z, voxel_t])

            if args.training_ready:
                try:
                    canon_img = nib.as_closest_canonical(img)
                    data = np.asanyarray(canon_img.dataobj)
                    if data.ndim == 4:
                        vol = data[:, :, :, int(np.clip(args.volume_index, 0, data.shape[3] - 1))]
                    else:
                        vol = data
                    vol = pad_volume_inplane(vol, 256, 256)
                    vol = apply_intensity_mapping(vol, z_clip, args.intensity_scale)
                    total_slices = vol.shape[2] if vol.ndim > 2 else 0
                    slice_indices = get_training_slice_indices(
                        total_slices,
                        args.training_slice_start,
                        args.training_slice_end,
                    )
                    file_id = build_file_id(nifti_path)
                    if slice_indices:
                        logging.info(
                            "Training export %s: saving %d slices in range [%d, %d]",
                            file_id,
                            len(slice_indices),
                            slice_indices[0],
                            slice_indices[-1],
                        )
                        for slice_idx in slice_indices:
                            slice2d = vol[:, :, slice_idx]
                            slice2d = np.rot90(slice2d, k=1)
                            slice2d = center_crop_or_pad(slice2d, 256, 256)
                            if args.save_npy:
                                npy_name = args.npy_pattern.format(file=file_id, slice=slice_idx)
                                npy_path = args.output_npy_dir / npy_name
                                slice_array = slice2d.astype(np.float32, copy=False)
                                if npy_path.suffix.lower() == ".npy":
                                    np.save(npy_path, slice_array)
                                else:
                                    np.savez_compressed(npy_path, arr=slice_array)
                            if save_png_enabled:
                                png = normalize_to_uint8(slice2d)
                                out_name = args.png_pattern.format(file=file_id, slice=slice_idx)
                                out_path = training_dir / out_name
                                iio.imwrite(out_path, png)
                    else:
                        logging.warning(
                            "Skipping %s: no valid slices in requested training range [%d, %d] for volume with %d slices",
                            file_id,
                            args.training_slice_start,
                            args.training_slice_end,
                            total_slices,
                        )
                except Exception as exc:
                    logging.warning("Failed training-ready export for %s: %s", nifti_path, exc)

                continue

            try:
                canon_img = nib.as_closest_canonical(img)
                data = np.asanyarray(canon_img.dataobj)
                if data.ndim == 4:
                    vol = data[:, :, :, int(np.clip(args.volume_index, 0, data.shape[3] - 1))]
                else:
                    vol = data
                vol = pad_volume_inplane(vol, 256, 256)
                vol = apply_intensity_mapping(vol, z_clip, args.intensity_scale)
                used_idx = pick_slice_index(vol.shape[2], args.slice_index)
                slice2d = vol[:, :, used_idx]
                slice2d = np.rot90(slice2d, k=1)
                slice2d = center_crop_or_pad(slice2d, 256, 256)
                if args.save_npy:
                    total_slices = vol.shape[2] if vol.ndim > 2 else 0
                    slice_indices = get_training_slice_indices(
                        total_slices,
                        args.training_slice_start,
                        args.training_slice_end,
                    )
                    file_id = build_file_id(nifti_path)
                    if slice_indices:
                        logging.info(
                            "Array export %s: saving %d slices in range [%d, %d]",
                            file_id,
                            len(slice_indices),
                            slice_indices[0],
                            slice_indices[-1],
                        )
                    else:
                        logging.warning(
                            "Skipping array export for %s: no valid slices in requested training range [%d, %d] for volume with %d slices",
                            file_id,
                            args.training_slice_start,
                            args.training_slice_end,
                            total_slices,
                        )
                    for slice_idx in slice_indices:
                        slice2d_range = vol[:, :, slice_idx]
                        slice2d_range = np.rot90(slice2d_range, k=1)
                        slice2d_range = center_crop_or_pad(slice2d_range, 256, 256)
                        npy_name = args.npy_pattern.format(file=file_id, slice=slice_idx)
                        npy_path = args.output_npy_dir / npy_name
                        slice_array = slice2d_range.astype(np.float32, copy=False)
                        if npy_path.suffix.lower() == ".npy":
                            np.save(npy_path, slice_array)
                        else:
                            np.savez_compressed(npy_path, arr=slice_array)
                if save_png_enabled:
                    file_id = build_file_id(nifti_path)
                    png = normalize_to_uint8(slice2d)
                    out_name = f"{file_id}_z{used_idx:04d}.png"
                    out_path = output_dir / out_name
                    iio.imwrite(out_path, png)
            except Exception as exc:
                logging.warning("Failed to export slice for %s: %s", nifti_path, exc)
                continue

            try:
                canon_img = nib.as_closest_canonical(img)
                total_slices = canon_img.shape[2] if len(canon_img.shape) > 2 else 0
                if total_slices > 0:
                    start = max(args.plot_slice_start, 0)
                    end = min(args.plot_slice_end, total_slices - 1)
                    if end < start:
                        start, end = 0, total_slices - 1
                    count = max(args.plot_slice_count, 1)
                    slice_indices = np.linspace(start, end, num=count, dtype=int)
                    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
                    axes = axes.flatten()
                    for ax, idx in zip(axes, slice_indices):
                        s2d = vol[:, :, int(idx)]
                        s2d = np.rot90(s2d, k=1)
                        s2d = center_crop_or_pad(s2d, 256, 256)
                        s2d = normalize_to_uint8(s2d)
                        ax.imshow(s2d, cmap="gray")
                        ax.set_title(f"z={int(idx)}", fontsize=8)
                        ax.axis("off")
                    for ax in axes[len(slice_indices):]:
                        ax.axis("off")
                    fig.tight_layout()
                    file_id = build_file_id(nifti_path)
                    montage_name = f"{file_id}_montage_z{start:04d}-{end:04d}.png"
                if save_png_enabled:
                    montage_path = output_dir / montage_name
                    fig.savefig(montage_path, dpi=150)
                    plt.close(fig)
            except Exception as exc:
                logging.warning("Failed to export montage for %s: %s", nifti_path, exc)
                continue

    logging.info("Wrote CSV summary to %s", csv_path)
    if save_png_enabled:
        logging.info("Saved PNG slices to %s", output_dir)


if __name__ == "__main__":
    main()
