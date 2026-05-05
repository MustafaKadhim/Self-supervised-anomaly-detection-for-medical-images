#!/usr/bin/env python3
"""Render PNGs for slices listed in a normal_slices CSV."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PNGs for slices listed in a CSV."
    )
    parser.add_argument("--csv-path", type=Path, default=Path("/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/Anomaly_Label_Intraventricular_substance.csv"), help="Input CSV with file names or columns: file[,slice,reason]")
    
    Label = "Intraventricular substance"  # Example label to filter on (case-insensitive)
    parser.add_argument("--output-npy-dir", type=Path, default=f"/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_Local_Anomalies_ByLabel_BestSlice_with_Label/{Label}", help="Output folder for saved float32 arrays")
    parser.add_argument("--output-dir", type=Path, default=f"/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_Local_Anomalies_ByLabel_BestSlice_with_Label/{Label}", help="Output folder for rendered PNGs")
    #parser.add_argument("--target-label", type=str, default=Label, help="Select the first slice containing this label (case-insensitive).")
    parser.add_argument("--include-label", type=str, default=Label, help="Only render slices that contain this label (case-insensitive).")
    parser.add_argument("--best-box-only", action="store_true", help="When used with --include-label, render only the slice with the largest bounding box per patient.")
    

    parser.add_argument("--label-root", type=Path, default=None, help="Optional root folder with label/patient subfolders (e.g. FastMRI_Local_Anomalies_ByLabel). Each patient folder may contain slices.csv to render those slice indices.")
    parser.add_argument("--slice-start", type=int, default=0, help="Start slice index (inclusive)")
    parser.add_argument("--slice-end", type=int, default=10, help="End slice index (inclusive)")
    parser.add_argument("--data-root", type=Path, default=Path("/media/mluser1/MustiUSB/"), help="Root folder with fastMRI .h5 files")
    parser.add_argument("--series-type", type=str, default="AXT1", help="Series type to filter on (e.g., AXT1)")

    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV with updated png_path")
    parser.add_argument("--png-pattern", type=str, default="{file}_slice_{slice:03d}.png", help="PNG filename pattern using {file} and {slice} placeholders")
    parser.add_argument("--npy-pattern", type=str, default="{file}_slice_{slice:03d}.npz", help="NPY/NPZ filename pattern using {file} and {slice} placeholders")
    parser.add_argument("--exclude-dirs", type=str, default=".Trash-1001,lost+found", help="Comma-separated folder names to skip while scanning")
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=True, help="Save PNGs for debugging")
    parser.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=True, help="Save float32 arrays for training")
    parser.add_argument("--PNGs-only", dest="pngs_only", action="store_true", help="Only save PNGs (disable NPZ output)")
    parser.add_argument("--npy-only", dest="npy_only", action="store_true", help="Only save NPZs (disable PNG output)")
    parser.add_argument("--showcase-labels", action=argparse.BooleanOptionalAction, default=True,
                        help="Overlay CSV 'reason' labels on saved PNGs for visualization")
    parser.add_argument("--annotation-csv", type=Path,
                        default=Path("/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Annotated_fastMRI_Brains_Detailed.csv"),
                        help="CSV with bounding boxes/labels (fastMRI+ detailed annotations)")

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
    return parser.parse_args()


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def get_series_type(hf: h5py.File) -> str:
    for key in (
        "acquisition",
        "series",
        "series_description",
        "SeriesDescription",
        "protocol_name",
        "ProtocolName",
    ):
        if key in hf.attrs:
            return _as_str(hf.attrs[key]).strip()
    return ""


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
        constant_values=0,
    )


def pad_volume_inplane(volume: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if volume.ndim < 3:
        return center_crop_or_pad(volume, target_h, target_w)
    padded = [center_crop_or_pad(volume[idx], target_h, target_w) for idx in range(volume.shape[0])]
    return np.stack(padded, axis=0)


def resize_to_size(arr: np.ndarray, target_h: int = 256, target_w: int = 256) -> np.ndarray:
    if arr.size == 0:
        return arr
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


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


def load_csv_rows(csv_path: Path, slice_start: int, slice_end: int) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open(newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "," in sample and "file" in sample.lower()
        if has_header:
            reader = csv.DictReader(f)
            rows: list[dict[str, str]] = []
            for row in reader:
                file_val = (row.get("file") or row.get("File") or row.get("path") or row.get("Path") or "").strip()
                if not file_val:
                    continue
                slice_val = (row.get("slice") or row.get("Slice") or "").strip()
                reason = (row.get("reason") or row.get("Reason") or "normal")
                if slice_val:
                    rows.append({"file": Path(file_val).stem, "slice": slice_val, "reason": reason})
                else:
                    for slice_idx in range(slice_start, slice_end + 1):
                        rows.append({"file": Path(file_val).stem, "slice": str(slice_idx), "reason": reason})
            return rows

        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem
            for slice_idx in range(slice_start, slice_end + 1):
                rows.append({"file": stem, "slice": str(slice_idx), "reason": "normal"})
        return rows


def load_slices_csv(slices_csv: Path) -> list[int]:
    if not slices_csv.exists():
        return []
    slices: list[int] = []
    with slices_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if "slice" in (reader.fieldnames or []):
            for row in reader:
                value = (row.get("slice") or "").strip()
                if value == "":
                    continue
                try:
                    slices.append(int(value))
                except ValueError:
                    continue
            return slices

        f.seek(0)
        for line in f:
            value = line.strip()
            if not value:
                continue
            try:
                slices.append(int(value))
            except ValueError:
                continue
    return slices


def load_label_root_rows(
    label_root: Path,
    slice_start: int,
    slice_end: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not label_root.exists():
        raise FileNotFoundError(f"Label root not found: {label_root}")
    for label_dir in sorted([p for p in label_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        patient_dirs = [p for p in label_dir.iterdir() if p.is_dir()]
        patients_csv = label_dir / "patients.csv"
        patient_names: list[str] = []
        if patient_dirs:
            patient_names = [p.name for p in sorted(patient_dirs)]
        elif patients_csv.exists():
            with patients_csv.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    value = (row.get("patient_id") or row.get("patient") or row.get("file") or "").strip()
                    if value:
                        patient_names.append(value)

        for patient in patient_names:
            patient_dir = label_dir / patient
            slices_csv = patient_dir / "slices.csv"
            slice_list = load_slices_csv(slices_csv)
            if not slice_list and slice_end >= slice_start:
                slice_list = list(range(slice_start, slice_end + 1))
            if not slice_list:
                rows.append(
                    {
                        "label": label,
                        "patient": patient,
                        "file": patient,
                        "slice": "",
                        "reason": label,
                    }
                )
                continue
            for slice_idx in slice_list:
                rows.append(
                    {
                        "label": label,
                        "patient": patient,
                        "file": patient,
                        "slice": str(slice_idx),
                        "reason": label,
                    }
                )
    return rows


def build_h5_index(data_root: Path, series_type: str, exclude_dirs: set[str]) -> dict[str, Path]:
    h5_files = [
        path
        for path in data_root.rglob("*.h5")
        if not any(part in exclude_dirs for part in path.parts)
    ]
    h5_files.sort()
    index: dict[str, Path] = {}
    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as hf:
                series = get_series_type(hf)
                if series_type and series.strip().lower() != series_type.strip().lower():
                    continue
            index[file_path.stem] = file_path
        except OSError:
            continue
    return index


def build_png_path(output_dir: Path, png_pattern: str, file_name: str, slice_idx: int) -> Path:
    try:
        filename = png_pattern.format(file=file_name, slice=slice_idx)
    except KeyError as exc:
        raise ValueError("png-pattern must use {file} and/or {slice}") from exc
    return output_dir / filename


def build_npy_path(output_dir: Path, npy_pattern: str, file_name: str, slice_idx: int) -> Path:
    try:
        filename = npy_pattern.format(file=file_name, slice=slice_idx)
    except KeyError as exc:
        raise ValueError("npy-pattern must use {file} and/or {slice}") from exc
    return output_dir / filename


def parse_z_clip(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) == 1:
        bound = float(parts[0])
        return -abs(bound), abs(bound)
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError("--z-clip must be 'min,max' or a single number")


def apply_intensity_mapping(volume: np.ndarray, clip_value: tuple[float, float], scale_mode: str) -> np.ndarray:
    volume_mean = float(volume.mean()) if volume.size else 0.0
    volume_std = float(volume.std()) if volume.size else 1.0
    if volume_std == 0.0:
        volume_std = 1.0
    volume_z = (volume - volume_mean) / volume_std
    clip_min, clip_max = clip_value
    volume_z = np.clip(volume_z, clip_min, clip_max)

    if scale_mode == "minus1_1":
        scale = max(abs(clip_min), abs(clip_max)) or 1.0
        volume_z = volume_z / scale
    elif scale_mode == "zero1":
        scale = max(abs(clip_min), abs(clip_max)) or 1.0
        volume_z = (volume_z / scale + 1.0) / 2.0
    return volume_z.astype(np.float32, copy=False)


def _get_first(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def _to_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def load_annotation_boxes(annotation_csv: Path) -> dict[str, dict[int, list[dict]]]:
    """Load fastMRI+ annotation boxes keyed by file stem and slice index."""
    if not annotation_csv.exists():
        logging.warning("Annotation CSV not found: %s", annotation_csv)
        return {}

    with annotation_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        boxes_by_file: dict[str, dict[int, list[dict]]] = {}
        for row in reader:
            file_val = _get_first(row, ("file", "File", "filename", "Filename", "path", "Path"))
            slice_val = _get_first(row, ("slice", "Slice"))
            if not file_val or slice_val == "":
                continue

            file_stem = Path(file_val).stem
            slice_idx = _to_int(slice_val)
            if slice_idx is None:
                continue

            x = _to_int(_get_first(row, ("x", "X")))
            y = _to_int(_get_first(row, ("y", "Y")))
            w = _to_int(_get_first(row, ("width", "Width", "w", "W")))
            h = _to_int(_get_first(row, ("height", "Height", "h", "H")))
            label = _get_first(row, ("label", "Label"))
            study_level = _get_first(row, ("study_level", "StudyLevel", "study", "Study"))

            box = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "label": label,
                "study_level": study_level,
            }

            boxes_by_file.setdefault(file_stem, {}).setdefault(slice_idx, []).append(box)
        return boxes_by_file


def draw_annotation_boxes(
    ax: plt.Axes,
    boxes: list[dict],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[str]:
    labels: list[str] = []
    for box in boxes:
        if str(box.get("study_level", "")).strip().lower() == "yes":
            continue
        x = box.get("x")
        y = box.get("y")
        w = box.get("width")
        h = box.get("height")
        if None in (x, y, w, h):
            continue
        x_scaled = x * scale_x
        y_scaled = y * scale_y
        w_scaled = w * scale_x
        h_scaled = h * scale_y
        rect = patches.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled, linewidth=1, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
        label = str(box.get("label", "")).strip()
        if label:
            labels.append(label)
            ax.text(
                x_scaled,
                max(0, y_scaled - 8),
                label,
                color="white",
                fontsize=8,
                clip_on=True,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
            )
    return labels


def summarize_labels_from_boxes(boxes: list[dict]) -> str:
    labels = [str(b.get("label", "")).strip() for b in boxes]
    labels = [lbl for lbl in labels if lbl]
    if not labels:
        return ""
    unique_labels = list(dict.fromkeys(labels))
    label_text = " | ".join(unique_labels)
    max_len = 120
    if len(label_text) > max_len:
        label_text = f"{label_text[: max_len - 3]}..."
    return label_text


def find_first_slice_with_label(
    boxes_by_file: dict[str, dict[int, list[dict]]],
    file_name: str,
    target_label: str,
) -> int | None:
    if not target_label:
        return None
    target = target_label.strip().lower()
    if not target:
        return None
    slice_map = boxes_by_file.get(file_name, {})
    if not slice_map:
        return None
    for slice_idx in sorted(slice_map.keys()):
        boxes = slice_map.get(slice_idx, [])
        for box in boxes:
            label = str(box.get("label", "")).strip().lower()
            if label and target in label:
                return slice_idx
    return None


def slice_has_label(
    boxes_by_file: dict[str, dict[int, list[dict]]],
    file_name: str,
    slice_idx: int,
    include_label: str,
) -> bool:
    if not include_label:
        return True
    target = include_label.strip().lower()
    if not target:
        return True
    boxes = boxes_by_file.get(file_name, {}).get(slice_idx, [])
    for box in boxes:
        label = str(box.get("label", "")).strip().lower()
        if label and target in label:
            return True
    return False


def find_best_slice_with_label(
    boxes_by_file: dict[str, dict[int, list[dict]]],
    file_name: str,
    include_label: str,
) -> int | None:
    if not include_label:
        return None
    target = include_label.strip().lower()
    if not target:
        return None
    slice_map = boxes_by_file.get(file_name, {})
    if not slice_map:
        return None
    best_slice: int | None = None
    best_area = -1
    for slice_idx, boxes in slice_map.items():
        for box in boxes:
            label = str(box.get("label", "")).strip().lower()
            if not label or target not in label:
                continue
            w = box.get("width")
            h = box.get("height")
            if w is None or h is None:
                continue
            area = int(w) * int(h)
            if area > best_area:
                best_area = area
                best_slice = slice_idx
    return best_slice


def main() -> None:
    args = parse_args()
    z_clip = parse_z_clip(args.z_clip)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.pngs_only and args.npy_only:
        raise ValueError("Use only one of --PNGs-only or --npy-only")
    if args.pngs_only:
        args.save_png = True
        args.save_npy = False
    if args.npy_only:
        args.save_png = False
        args.save_npy = True

    if args.label_root:
        if args.output_dir is None:
            args.output_dir = Path(args.label_root)
        if args.output_npy_dir is None:
            args.output_npy_dir = Path(args.label_root)
    else:
        if args.output_dir is None:
            args.output_dir = Path(
                "/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/Anomaly_PNGs_Collection/"
            )
        if args.output_npy_dir is None:
            args.output_npy_dir = Path(
                "/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/AnomalyCollection_Local_npy"
            )

    exclude_dirs = {name.strip() for name in args.exclude_dirs.split(",") if name.strip()}
    slice_start = min(args.slice_start, args.slice_end)
    slice_end = max(args.slice_start, args.slice_end)
    if args.label_root:
        rows = load_label_root_rows(args.label_root, slice_start, slice_end)
    else:
        rows = load_csv_rows(args.csv_path, slice_start, slice_end)
    if not rows:
        source = args.label_root if args.label_root else args.csv_path
        logging.warning("No usable rows found in %s", source)
        return

    logging.info("Loaded %d rows (slice range %d-%d)", len(rows), slice_start, slice_end)
    logging.info("Save PNG: %s | Save NPY: %s", args.save_png, args.save_npy)

    annotation_boxes = load_annotation_boxes(args.annotation_csv)
    annotation_labels_by_file: dict[str, str] = {}
    for file_key, slices in annotation_boxes.items():
        all_boxes = []
        for boxes in slices.values():
            all_boxes.extend(boxes)
        annotation_labels_by_file[file_key] = summarize_labels_from_boxes(all_boxes)

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    if args.save_png:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        args.output_npy_dir.mkdir(parents=True, exist_ok=True)
    h5_index = build_h5_index(args.data_root, args.series_type, exclude_dirs)
    if not h5_index:
        raise FileNotFoundError("No matching .h5 files found under data root.")

    output_rows: list[dict[str, str]] = []
    rendered = 0
    missing = 0
    saved_npy = 0

    if args.label_root:
        processed_files: set[str] = set()
        for row in rows:
            file_name = (row.get("file") or "").strip()
            slice_raw = (row.get("slice") or "").strip()
            label = (row.get("label") or "").strip()
            patient = (row.get("patient") or "").strip()
            if not file_name:
                continue
            if args.best_box_only and args.include_label:
                if file_name in processed_files:
                    continue
                processed_files.add(file_name)
                slice_idx = find_best_slice_with_label(
                    annotation_boxes,
                    file_name,
                    args.include_label,
                )
                if slice_idx is None:
                    missing += 1
                    continue
            else:
                slice_idx: int | None = None
                if slice_raw != "":
                    try:
                        slice_idx = int(slice_raw)
                    except ValueError:
                        continue

            h5_path = h5_index.get(file_name)
            if h5_path is None:
                missing += 1
                continue

            try:
                with h5py.File(h5_path, "r") as hf:
                    if "reconstruction_rss" not in hf:
                        missing += 1
                        continue
                    volume = np.asarray(hf["reconstruction_rss"][:], dtype=np.float32)
            except OSError:
                missing += 1
                continue

            volume = pad_volume_inplane(volume, 320, 320)
            volume_z = apply_intensity_mapping(volume, z_clip, args.intensity_scale)

            if slice_idx is None:
                slice_idx = int(volume_z.shape[0] // 2) if volume_z.shape[0] else 0

            if slice_idx < 0 or slice_idx >= volume_z.shape[0]:
                missing += 1
                continue

            if args.include_label and not slice_has_label(
                annotation_boxes,
                file_name,
                slice_idx,
                args.include_label,
            ):
                continue

            arr = np.squeeze(volume_z[slice_idx])
            if arr.size:
                arr = np.flipud(arr)
            arr = center_crop_or_pad(arr, 320, 320)
            arr = resize_to_size(arr, 256, 256)
            png_path: Path | None = None
            npy_path: Path | None = None

            label_output_dir = args.output_dir / label / patient if label and patient else args.output_dir
            label_npy_dir = args.output_npy_dir / label / patient if label and patient else args.output_npy_dir
            if args.save_png:
                label_output_dir.mkdir(parents=True, exist_ok=True)
            if args.save_npy:
                label_npy_dir.mkdir(parents=True, exist_ok=True)

            if args.save_npy:
                npy_path = build_npy_path(label_npy_dir, args.npy_pattern, file_name, slice_idx)
                if npy_path.suffix.lower() == ".npz":
                    np.savez_compressed(npy_path, arr=arr.astype(np.float32, copy=False))
                else:
                    np.save(npy_path, arr.astype(np.float32, copy=False))

            if args.save_png:
                png_path = build_png_path(label_output_dir, args.png_pattern, file_name, slice_idx)
                height, width = arr.shape[:2]
                fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
                ax.imshow(normalize_to_uint8(arr), cmap="gray")
                ax.axis("off")
                if args.showcase_labels:
                    boxes = annotation_boxes.get(file_name, {}).get(slice_idx, [])
                    scale_x = arr.shape[1] / 320.0
                    scale_y = arr.shape[0] / 320.0
                    if boxes:
                        draw_annotation_boxes(ax, boxes, scale_x=scale_x, scale_y=scale_y)
                    label_text = summarize_labels_from_boxes(boxes)
                    if not label_text:
                        label_text = annotation_labels_by_file.get(file_name, "")
                    if not label_text and label:
                        label_text = label
                    if label_text:
                        ax.set_title(
                            f"{file_name} — {label_text}",
                            fontsize=8,
                            color="white",
                            loc="left",
                            pad=6,
                            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
                        )
                    reason = (row.get("reason") or "").strip()
                    if reason:
                        label_text = str(reason).strip()
                        if label_text and label_text.lower() != "normal":
                            ax.text(
                                0.02,
                                0.02,
                                label_text,
                                transform=ax.transAxes,
                                fontsize=10,
                                color="white",
                                ha="left",
                                va="bottom",
                                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3),
                            )
                fig.savefig(png_path, bbox_inches=None, pad_inches=0)
                plt.close(fig)
                rendered += 1

            output_rows.append(
                {
                    "label": label,
                    "patient": patient,
                    "file": file_name,
                    "slice": str(slice_idx),
                    "png_path": str(png_path) if png_path else "",
                    "npy_path": str(npy_path) if npy_path else "",
                    "reason": (row.get("reason") or "").strip(),
                }
            )
    else:
        rows_by_patient: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            file_name = (row.get("file") or "").strip()
            slice_raw = (row.get("slice") or "").strip()
            if not file_name or slice_raw == "":
                continue
            try:
                int(slice_raw)
            except ValueError:
                continue
            rows_by_patient.setdefault(file_name, []).append(row)

        for file_name, patient_rows in rows_by_patient.items():
            h5_path = h5_index.get(file_name)
            if h5_path is None:
                missing += len(patient_rows)
                continue

            try:
                with h5py.File(h5_path, "r") as hf:
                    if "reconstruction_rss" not in hf:
                        missing += len(patient_rows)
                        continue
                    volume = np.asarray(hf["reconstruction_rss"][:], dtype=np.float32)
            except OSError:
                missing += len(patient_rows)
                continue

            volume = pad_volume_inplane(volume, 320, 320)
            volume_z = apply_intensity_mapping(volume, z_clip, args.intensity_scale)
            def _render_slice(slice_idx: int, reason: str) -> None:
                nonlocal rendered, saved_npy
                if slice_idx < 0 or slice_idx >= volume_z.shape[0]:
                    return
                if args.include_label and not slice_has_label(
                    annotation_boxes,
                    file_name,
                    slice_idx,
                    args.include_label,
                ):
                    return

                arr = np.squeeze(volume_z[slice_idx])
                if arr.size:
                    arr = np.flipud(arr)
                arr = center_crop_or_pad(arr, 320, 320)
                arr = resize_to_size(arr, 256, 256)
                png_path: Path | None = None
                npy_path: Path | None = None

                if args.save_npy:
                    npy_path = build_npy_path(args.output_npy_dir, args.npy_pattern, file_name, slice_idx)
                    if npy_path.suffix.lower() == ".npz":
                        np.savez_compressed(npy_path, arr=arr.astype(np.float32, copy=False))
                    else:
                        np.save(npy_path, arr.astype(np.float32, copy=False))
                    saved_npy += 1

                if args.save_png:
                    png_path = build_png_path(args.output_dir, args.png_pattern, file_name, slice_idx)
                    height, width = arr.shape[:2]
                    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
                    ax.imshow(normalize_to_uint8(arr), cmap="gray")
                    ax.axis("off")
                    if args.showcase_labels:
                        boxes = annotation_boxes.get(file_name, {}).get(slice_idx, [])
                        scale_x = arr.shape[1] / 320.0
                        scale_y = arr.shape[0] / 320.0
                        if boxes:
                            draw_annotation_boxes(ax, boxes, scale_x=scale_x, scale_y=scale_y)
                        label_text = summarize_labels_from_boxes(boxes)
                        if not label_text:
                            label_text = annotation_labels_by_file.get(file_name, "")
                        if label_text:
                            ax.set_title(
                                f"{file_name} — {label_text}",
                                fontsize=8,
                                color="white",
                                loc="left",
                                pad=6,
                                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
                            )
                        if reason:
                            label_text = str(reason).strip()
                            if label_text and label_text.lower() != "normal":
                                ax.text(
                                    0.02,
                                    0.02,
                                    label_text,
                                    transform=ax.transAxes,
                                    fontsize=10,
                                    color="white",
                                    ha="left",
                                    va="bottom",
                                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3),
                                )
                    fig.savefig(png_path, bbox_inches=None, pad_inches=0)
                    plt.close(fig)
                    rendered += 1

                output_rows.append(
                    {
                        "file": file_name,
                        "slice": str(slice_idx),
                        "png_path": str(png_path) if png_path else "",
                        "npy_path": str(npy_path) if npy_path else "",
                        "reason": reason,
                    }
                )

            if args.best_box_only and args.include_label:
                best_slice_idx = find_best_slice_with_label(
                    annotation_boxes,
                    file_name,
                    args.include_label,
                )
                if best_slice_idx is None:
                    missing += len(patient_rows)
                    continue
                _render_slice(best_slice_idx, args.include_label)
            elif args.include_label:
                for row in patient_rows:
                    slice_raw = (row.get("slice") or "").strip()
                    slice_idx = _to_int(slice_raw)
                    if slice_idx is None:
                        continue
                    reason = (row.get("reason") or "").strip()
                    _render_slice(slice_idx, reason)
            elif args.target_label:
                target_slice_idx = find_first_slice_with_label(
                    annotation_boxes,
                    file_name,
                    args.target_label,
                )
                if target_slice_idx is None:
                    logging.warning(
                        "No slice found with label '%s' for %s; skipping.",
                        args.target_label,
                        file_name,
                    )
                    missing += len(patient_rows)
                    continue
                reason = args.target_label
                _render_slice(target_slice_idx, reason)
            else:
                for row in patient_rows:
                    slice_raw = (row.get("slice") or "").strip()
                    slice_idx = _to_int(slice_raw)
                    if slice_idx is None:
                        continue
                    reason = (row.get("reason") or "").strip()
                    _render_slice(slice_idx, reason)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["label", "patient", "file", "slice", "png_path", "npy_path", "reason"]
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    logging.info("Rendered %d PNGs", rendered)
    logging.info("Saved %d NPY/NPZ files", saved_npy)
    if missing:
        logging.warning("Skipped %d rows due to missing data", missing)
    if args.output_csv:
        logging.info("Wrote updated CSV to %s", args.output_csv)


if __name__ == "__main__":
    main()
