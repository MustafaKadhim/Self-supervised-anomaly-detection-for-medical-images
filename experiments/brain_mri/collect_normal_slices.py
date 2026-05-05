#!/usr/bin/env python3
"""Collect normal slices (Normal-labeled or unannotated) into a CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect normal slices (Normal-labeled or unannotated) into a CSV."
    )
    parser.add_argument("--annotation-csv", type=Path, default=Path(__file__).parent / ".annotation_cache" / "brain.csv", help="Annotation CSV with columns like file,slice,study_level,label")
    parser.add_argument("--patient-list", type=Path, default=Path(__file__).parent / "Annotated_FastMRI_Brains.csv", help="CSV with a single column of patient file stems")
    parser.add_argument("--series-type", type=str, default="AXT1", help="Series type to filter patient list (e.g., AXT1)")
    parser.add_argument("--slice-start", type=int, default=0, help="Start slice index (inclusive)")
    parser.add_argument("--slice-end", type=int, default=10, help="End slice index (inclusive)")
    parser.add_argument("--png-root", type=Path, default=Path("/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Normal_Brains_pngs"), help="Root directory containing PNGs")
    parser.add_argument("--png-pattern", type=str, default="{file}.png", help="PNG filename pattern using {file} and {slice} placeholders")
    parser.add_argument("--normal-label-keyword", type=str, default="Normal", help="Label keyword indicating normal annotations (case-insensitive)")
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).parent / "normal_slices_0_10.csv", help="Output CSV path")
    return parser.parse_args()


def load_patient_list(path: Path, series_type: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Patient list not found: {path}")
    patients: list[str] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if not name:
                continue
            if series_type and f"_{series_type}_" not in name:
                continue
            patients.append(name)
    return patients


def load_annotations(path: Path) -> dict[str, dict[int, list[dict[str, str]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")
    annotations: dict[str, dict[int, list[dict[str, str]]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = (row.get("file") or "").strip()
            if not file_name:
                continue
            slice_raw = row.get("slice")
            if slice_raw is None or slice_raw == "":
                continue
            try:
                slice_idx = int(slice_raw)
            except ValueError:
                continue
            annotations.setdefault(file_name, {}).setdefault(slice_idx, []).append(
                {k: ("" if v is None else str(v)) for k, v in row.items()}
            )
    return annotations


def is_normal_slice(
    rows_for_slice: list[dict[str, str]],
    normal_label_keyword: str,
) -> tuple[bool, str]:
    if not rows_for_slice:
        return True, "no_annotation"

    keyword = normal_label_keyword.strip().lower()
    for row in rows_for_slice:
        study_level = (row.get("study_level") or "").strip()
        label = (row.get("label") or "").strip()
        if study_level.lower() == "yes":
            return True, "study_level_yes"
        if keyword and keyword in label.lower():
            return True, "label_normal"

    return False, "has_non_normal_annotation"


def build_png_path(png_root: Path, png_pattern: str, file_name: str, slice_idx: int) -> Path:
    try:
        filename = png_pattern.format(file=file_name, slice=slice_idx)
    except KeyError as exc:
        raise ValueError("png-pattern must use {file} and/or {slice}") from exc
    return png_root / filename


def main() -> None:
    args = parse_args()

    patients = load_patient_list(args.patient_list, args.series_type)
    annotations = load_annotations(args.annotation_csv)

    slice_start = min(args.slice_start, args.slice_end)
    slice_end = max(args.slice_start, args.slice_end)

    rows_out: list[dict[str, str]] = []
    for file_name in patients:
        per_file = annotations.get(file_name, {})
        for slice_idx in range(slice_start, slice_end + 1):
            rows_for_slice = per_file.get(slice_idx, [])
            is_normal, reason = is_normal_slice(rows_for_slice, args.normal_label_keyword)
            if not is_normal:
                continue
            png_path = build_png_path(args.png_root, args.png_pattern, file_name, slice_idx)
            rows_out.append(
                {
                    "file": file_name,
                    "slice": str(slice_idx),
                    "png_path": str(png_path),
                    "reason": reason,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "slice", "png_path", "reason"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} normal slices to {args.output_csv}")


if __name__ == "__main__":
    main()
