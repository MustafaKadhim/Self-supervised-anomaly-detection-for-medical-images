import argparse
from bisect import bisect_right
import csv
import json
import logging
import re
from datetime import datetime
from math import inf, isinf
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt


FP_RATIO_IMAGE_SIZE = 256
FP_RATIO_SLICE_PIXELS = FP_RATIO_IMAGE_SIZE * FP_RATIO_IMAGE_SIZE
SLICE_STEM_RE = re.compile(r"_slice_(\d+)(.*)$")


def split_stem_slice_suffix(stem: str) -> Tuple[str, Optional[int], str]:
    """Split '<base>_slice_<idx><suffix>' into base, idx, and suffix.

    Suffix is preserved so variants like ' copy' remain unique patient cases.
    """
    match = SLICE_STEM_RE.search(stem)
    if not match:
        return stem, None, ""
    base = stem[: match.start()]
    slice_idx = int(match.group(1))
    suffix = match.group(2) or ""
    return base, slice_idx, suffix


def patient_label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    base, _, suffix = split_stem_slice_suffix(stem)
    return strip_unknown_prefix(base + suffix)


def slice_index_from_filename(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    _, idx, _ = split_stem_slice_suffix(stem)
    return idx


def compute_fp_pixel_ratio_percent(highlighted_fp_pixels: int, no_bbox_slices: int) -> float:
    """Compute FP pixel ratio in percent over no-bbox slice area."""
    if no_bbox_slices <= 0:
        return 0.0
    denom = float(no_bbox_slices * FP_RATIO_SLICE_PIXELS)
    return (float(highlighted_fp_pixels) / denom) * 100.0


def compute_paper_precision_f1(
    tp_detected: Optional[bool],
    inside_pixels: int,
    outside_pixels: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute FP ratio, precision, and F1 using the paper's TP-event definition.

    FP ratio = predicted_pixels_outside_healthy / predicted_pixels_inside_bbox
    Precision = TP / (TP + FP_ratio), where TP is a detection event in {0, 1}
    F1 = 2P/(P+1) when TP=1, else 0
    """
    if tp_detected is None:
        return None, None, None

    fp_ratio: Optional[float] = None
    if int(inside_pixels) > 0:
        fp_ratio = float(outside_pixels) / float(inside_pixels)

    tp_value = 1.0 if bool(tp_detected) else 0.0
    precision = 0.0
    if fp_ratio is not None:
        denom = tp_value + fp_ratio
        precision = float(tp_value / denom) if denom > 0 else 0.0

    f1_score = float((2.0 * precision) / (precision + 1.0)) if bool(tp_detected) else 0.0
    return fp_ratio, float(precision), float(f1_score)


def truncate_label(name: str, keep: int = 50) -> str:
    if len(name) <= keep:
        return name
    return "..." + name[-keep:]


def matches_case_folder(case_folder: Optional[str], filters: Optional[Set[str]]) -> bool:
    """Return True when the case_folder matches any filter substring (case-insensitive)."""
    if not filters:
        return True
    if not case_folder:
        return False
    cf_lower = str(case_folder).lower()
    return any(token in cf_lower for token in filters)


def matches_category(category: Optional[str], filters: Optional[Set[str]]) -> bool:
    """Return True when category matches any filter token (case-insensitive substring)."""
    if not filters:
        return True
    if not category:
        return False
    cat_lower = str(category).lower()
    return any(token in cat_lower for token in filters)


def filter_items_by_category(items: List[dict], categories: Optional[Set[str]]) -> List[dict]:
    """Filter result or patient-summary entries by category."""
    if not categories:
        return items
    return [item for item in items if matches_category(item.get("category"), categories)]


def filter_items_by_case_folder(items: List[dict], case_folders: Optional[Set[str]]) -> List[dict]:
    """Filter result or patient-summary entries by case_folder."""
    if not case_folders:
        return items
    return [item for item in items if matches_case_folder(item.get("case_folder"), case_folders)]


def load_payload(json_path: Path) -> Dict:
    with json_path.open("r") as f:
        return json.load(f)


def load_results(json_path: Path) -> List[dict]:
    payload = load_payload(json_path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("results field is missing or not a list")
    return results


def aggregate_patient_clamp_from_results(results: List[dict]) -> List[dict]:
    """Aggregate clamped pixel sums per patient from slice-level results.

    Uses filename stem before `_slice_` to derive patient ID (consistent with sharpness/binary_sum collection).
    Expects `clamped_pixel_sum` and `case_folder` fields per slice.
    """
    if not results:
        return []
    def derive_patient_id(item: dict) -> str:
        # Derive from filename stem first for consistency with other patient ID derivations
        filename = item.get("filename") or Path(item.get("path", "")).name
        if filename:
            stem = Path(filename).stem
            if "_slice_" in stem:
                return strip_unknown_prefix(stem.split("_slice_", 1)[0])
            return strip_unknown_prefix(stem)
        # Fallback to explicit patient_id or case_folder
        pid = item.get("patient_id")
        if pid:
            return strip_unknown_prefix(str(pid))
        case_folder = item.get("case_folder")
        if case_folder:
            return strip_unknown_prefix(str(case_folder))
        return "unknown"

    agg: Dict[str, Dict[str, object]] = {}
    for item in results:
        pid = derive_patient_id(item)
        case_folder = item.get("case_folder", "")
        clamp_val = item.get("clamped_pixel_sum")
        if clamp_val is None:
            clamp_val = item.get("clamped_sum", 0.0)
        clamp_sum = float(clamp_val)
        clamp_pixels = int(item.get("num_pixels_above_clamp_thresh") or item.get("anomaly_pixel_count") or 0)
        if pid not in agg:
            agg[pid] = {
                "patient_id": pid,
                "case_folder": case_folder,
                "category": item.get("category", ""),
                "total_clamped_pixel_sum": 0.0,
                "total_pixels_above_thresh": 0,
                "num_slices": 0,
                "slice_details": [],
            }
        agg_entry = agg[pid]
        agg_entry["total_clamped_pixel_sum"] += clamp_sum
        agg_entry["total_pixels_above_thresh"] += clamp_pixels
        agg_entry["num_slices"] += 1
        agg_entry["slice_details"].append({
            "filename": item.get("filename"),
            "clamped_sum": clamp_sum,
            "pixels_above_thresh": clamp_pixels,
        })

    # finalize mean per slice
    for entry in agg.values():
        ns = entry["num_slices"]
        entry["mean_clamped_sum_per_slice"] = entry["total_clamped_pixel_sum"] / ns if ns else 0.0
    return sorted(agg.values(), key=lambda x: x["total_clamped_pixel_sum"], reverse=True)


def extract_series(
    results: List[dict],
    metric: str,
    top_n: int | None,
    allowed_case_folders: Optional[Set[str]] = None,
) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    values: List[float] = []
    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        value = item.get(metric)
        if value is None:
            continue
        filename = item.get("filename") or Path(item.get("path", "")).name
        labels.append(truncate_label(filename))
        values.append(float(value))
        if top_n is not None and len(values) >= top_n:
            break
    if not values:
        raise ValueError(f"No values collected for metric '{metric}'")
    return labels, values


def aggregate_anomaly_counts(
    results: List[dict],
    metric: str,
    cutoff_y: float,
    allowed_case_folders: Optional[Set[str]] = None,
) -> Tuple[List[str], List[int]]:
    counts: Dict[str, int] = {}
    for item in results:
        case_folder = item.get("case_folder")
        if case_folder is None:
            continue
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        value = item.get(metric)
        if value is None:
            continue
        if float(value) > cutoff_y:
            counts[case_folder] = counts.get(case_folder, 0) + 1
    if not counts:
        raise ValueError("No anomaly counts found for given filters and cutoff")
    # Sort by descending count then name for stable, readable ordering
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [truncate_label(k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    return labels, values


def patient_id_from_item(item: dict) -> Optional[str]:
    filename = item.get("filename") or Path(item.get("path", "")).name
    if not filename:
        return None
    return patient_label_from_filename(filename)


def patient_display_label(p: dict) -> str:
    """Choose a display label for a patient summary entry.

    Prefers case_folder when informative; otherwise derives from first slice filename.
    For filenames like Unknown_Unknown_5_Stor_T2_till_sCT-motion_slice_039 copy.npy,
    preserves the suffix while stripping leading "Unknown_" prefixes.
    """
    case_folder = str(p.get("case_folder", ""))
    if case_folder and case_folder.lower() != "unknown":
        return case_folder

    slice_details = p.get("slice_details", []) or []
    if slice_details:
        fname = slice_details[0].get("filename") or ""
        base = patient_label_from_filename(fname) if fname else ""
        return base or case_folder or p.get("patient_id", "unknown")

    return case_folder or p.get("patient_id", "unknown")


def strip_unknown_prefix(name: str) -> str:
    """Remove leading "unknown_" segments (case-insensitive) from a name."""
    lowered = name.lower()
    result = name
    while lowered.startswith("unknown_"):
        if "_" in result:
            result = result.split("_", 1)[1]
        else:
            break
        lowered = result.lower()
    return result or name


def aggregate_patient_status(
    results: List[dict],
    metric: str,
    metric_cutoff: float,
    slice_count_cutoff: int,
    allowed_case_folders: Optional[Set[str]] = None,
) -> Tuple[List[str], List[int], List[bool], Dict[str, int]]:
    counts: Dict[str, int] = {}
    patient_is_orig: Dict[str, bool] = {}
    patient_case_folders: Dict[str, Set[str]] = {}
    for item in results:
        case_folder = item.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        value = item.get(metric)
        if value is None:
            continue
        if float(value) <= metric_cutoff:
            continue
        patient_id = patient_id_from_item(item)
        if patient_id is None:
            continue
        counts[patient_id] = counts.get(patient_id, 0) + 1
        if case_folder is not None:
            patient_is_orig[patient_id] = patient_is_orig.get(patient_id, False) or (case_folder == "orig")
            if patient_id not in patient_case_folders:
                patient_case_folders[patient_id] = set()
            patient_case_folders[patient_id].add(case_folder)
    if not counts:
        raise ValueError("No patient counts found for given filters and cutoff")
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [truncate_label(k) for k, _ in sorted_items]
    statuses = [1 if v > slice_count_cutoff else 0 for _, v in sorted_items]
    orig_flags = [patient_is_orig.get(k, False) for k, _ in sorted_items]
    # Count anomalies per case_folder
    anomaly_per_folder: Dict[str, int] = {}
    for patient_id, status in [(k, 1 if v > slice_count_cutoff else 0) for k, v in sorted_items]:
        if status == 1 and patient_id in patient_case_folders:
            for cf in patient_case_folders[patient_id]:
                anomaly_per_folder[cf] = anomaly_per_folder.get(cf, 0) + 1
    return labels, statuses, orig_flags, anomaly_per_folder


def plot_patient_clamp_sums(
    patient_summary: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    anomaly_threshold: float = 450.0,
) -> None:
    if not patient_summary:
        raise ValueError("patient_summary is empty in JSON")
    sorted_patients = sorted(
        patient_summary,
        key=lambda x: float(x.get("total_clamped_pixel_sum", 0.0)),
        reverse=True,
    )
    if top_n is not None:
        sorted_patients = sorted_patients[:top_n]

    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    orig_flags: List[bool] = []
    base_color = "#4C72B0"
    for p in sorted_patients:
        display_label = patient_display_label(p)
        labels.append(truncate_label(display_label, keep=80))
        val = float(p.get("total_clamped_pixel_sum", 0.0))
        values.append(val)
        is_orig = "orig" in display_label.lower() or p.get("case_folder", "").lower() == "orig"
        orig_flags.append(is_orig)
        # Color based on anomaly threshold: red if above threshold, green if orig and below, blue otherwise
        if val > anomaly_threshold:
            colors.append("red")
        elif is_orig:
            colors.append("green")
        else:
            colors.append(base_color)

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    
    # Color x-axis labels green for orig patients
    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")
    
    ax.set_ylabel("Total clamped pixel sum")
    ax.set_title(f"Patient total clamped pixel sum (anomaly threshold = {anomaly_threshold})")

    # Add threshold line
    ax.axhline(anomaly_threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={anomaly_threshold}")

    # Add legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="red", label=f"anomaly (>{anomaly_threshold})"),
        Patch(facecolor="green", label="orig (normal)"),
        Patch(facecolor=base_color, label="other (normal)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    # Count anomalies
    num_anomalies = sum(1 for v in values if v > anomaly_threshold)
    num_orig_anomalies = sum(1 for v, o in zip(values, orig_flags) if v > anomaly_threshold and o)
    summary_text = f"Total: {len(values)} patients\nAnomalies: {num_anomalies}\nOrig anomalies: {num_orig_anomalies}"
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_patient_first_heatmap_sums(
    patient_summary: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 500.0,
    slice_start: int = 38,
    slice_end: int = 49,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot total first-heatmap sums for each INDIVIDUAL patient (by unique filename base).

    Each unique patient (derived from filename) gets its own bar.
    Sums clamped_sum_first_heatmap for slices in [slice_start, slice_end].
    Bars are red if value exceeds threshold, blue otherwise.
    """
    if not patient_summary:
        raise ValueError("patient_summary is empty in JSON")

    def slice_idx_from_name(name: str) -> Optional[int]:
        return slice_index_from_filename(name)

    def extract_patient_base(filename: str) -> str:
        """Extract patient identifier while preserving suffix modifiers."""
        return patient_label_from_filename(filename)

    # Aggregate by individual patient (unique filename base)
    patient_sums: Dict[str, float] = {}
    patient_info: Dict[str, Dict[str, object]] = {}
    for p in patient_summary:
        case_folder = p.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        slice_details = p.get("slice_details", []) or []
        for sd in slice_details:
            fname = sd.get("filename", "")
            if not fname:
                continue
            idx = slice_idx_from_name(fname)
            if idx is None:
                continue
            if not (slice_start <= idx <= slice_end):
                continue
            patient_base = extract_patient_base(fname)
            first_heatmap_sum = float(sd.get("clamped_sum_first_heatmap", 0.0))
            patient_sums[patient_base] = patient_sums.get(patient_base, 0.0) + first_heatmap_sum
            if patient_base not in patient_info:
                is_orig = str(case_folder or "").lower() == "orig" or "orig" in patient_base.lower()
                patient_info[patient_base] = {
                    "case_folder": case_folder,
                    "is_orig": is_orig,
                }

    if not patient_sums:
        raise ValueError("No individual patients found with slices in range")

    # Sort by descending sum
    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(k, keep=60) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_info.get(k, {}).get("is_orig", False) for k, _ in sorted_items]
    base_color = "#1f77b4"  # blue
    colors = ["red" if v > threshold else base_color for v in values]

    width = max(12, min(0.3 * len(labels), 80))
    fig, ax = plt.subplots(figsize=(width, 7))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)

    # Color patient labels: purple for anomalies, green for orig normals
    for text, val, is_orig in zip(ax.get_xticklabels(), values, orig_flags):
        if val > threshold:
            text.set_color("purple")
        elif is_orig:
            text.set_color("green")

    ax.set_ylabel(f"Total first-heatmap pixel sum (slices {slice_start}-{slice_end})")
    ax.set_title(f"Individual patient first-heatmap sums (threshold = {threshold})")

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold}")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="red", label=f"anomaly (>{threshold})"),
        Patch(facecolor=base_color, label=f"normal (<= {threshold})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    # Summary stats
    num_patients = len(values)
    num_anomalies = sum(1 for v in values if v > threshold)
    summary_text = (
        f"Total patients: {num_patients}\n"
        f"Above threshold: {num_anomalies}"
    )
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_patient_pixels_above_thresh(
    patient_summary: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
) -> None:
    """Plot total pixels above threshold per patient.

    X-axis: patient name (green if contains 'orig')
    Y-axis: sum of pixels_above_thresh over all slices
    """
    if not patient_summary:
        raise ValueError("patient_summary is empty in JSON")
    sorted_patients = sorted(
        patient_summary,
        key=lambda x: int(x.get("total_pixels_above_thresh", 0)),
        reverse=True,
    )
    if top_n is not None:
        sorted_patients = sorted_patients[:top_n]

    labels: List[str] = []
    values: List[int] = []
    orig_flags: List[bool] = []
    base_color = "#4C72B0"
    for p in sorted_patients:
        display_label = patient_display_label(p)
        labels.append(truncate_label(display_label, keep=80))
        val = int(p.get("total_pixels_above_thresh", 0))
        values.append(val)
        is_orig = "orig" in display_label.lower() or p.get("case_folder", "").lower() == "orig"
        orig_flags.append(is_orig)

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=base_color)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    # Color x-axis labels green for orig patients
    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Total pixels above clamp threshold")
    ax.set_title("Patient total pixels above clamp threshold (summed over slices)")

    # Add summary box
    num_orig = sum(orig_flags)
    summary_text = f"Total: {len(values)} patients\nOrig patients: {num_orig}"
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_token_surprisal_hot_px(
    results: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 300.0,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot token_surprisal_hot_px for each slice/sample.

    Bars are red when value exceeds threshold; blue otherwise.
    """
    labels: List[str] = []
    values: List[float] = []
    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        value = item.get("token_surprisal_hot_px")
        if value is None:
            continue
        filename = item.get("filename") or Path(item.get("path", "")).name
        labels.append(truncate_label(filename))
        values.append(float(value))
        if top_n is not None and len(values) >= top_n:
            break
    if not values:
        raise ValueError("No token_surprisal_hot_px values found for given filters")

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    colors = ["red" if v > threshold else "#4C72B0" for v in values]
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Token surprisal hot pixels")
    ax.set_title(f"Token surprisal hot pixels per sample (threshold = {threshold})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.legend()

    num_anomalies = sum(1 for v in values if v > threshold)
    summary_text = f"Anomalies: {num_anomalies}/{len(values)}"
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_binary_sum_heatmap_per_sample(
    results: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 1500.0,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot Binary_Sum_Heatmap per slice/sample.

    Bars are red when value exceeds threshold; blue otherwise.
    """
    labels: List[str] = []
    values: List[float] = []
    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        value = item.get("Binary_Sum_Heatmap")
        if value is None:
            continue
        filename = item.get("filename") or Path(item.get("path", "")).name
        labels.append(truncate_label(filename))
        values.append(float(value))
        if top_n is not None and len(values) >= top_n:
            break
    if not values:
        raise ValueError("No Binary_Sum_Heatmap values found for given filters")

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    colors = ["red" if v > threshold else "#4C72B0" for v in values]
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Binary_Sum_Heatmap")
    ax.set_title(f"Binary_Sum_Heatmap per sample (threshold = {threshold})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.legend()

    num_anomalies = sum(1 for v in values if v > threshold)
    summary_text = f"Anomalies: {num_anomalies}/{len(values)}"
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_token_binary_per_sample(
    results: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 400.0,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot per-sample sum of token_surprisal_hot_px and Binary_Sum_Heatmap."""
    labels: List[str] = []
    values: List[float] = []
    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        token_val = item.get("token_surprisal_hot_px")
        binary_val = item.get("Binary_Sum_Heatmap")
        if token_val is None or binary_val is None:
            continue
        filename = item.get("filename") or Path(item.get("path", "")).name
        labels.append(truncate_label(filename))
        values.append(float(token_val) + float(binary_val))
        if top_n is not None and len(values) >= top_n:
            break
    if not values:
        raise ValueError("No combined token_surprisal_hot_px + Binary_Sum_Heatmap values found")

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    colors = ["red" if v > threshold else "#4C72B0" for v in values]
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Token surprisal hot px + Binary_Sum_Heatmap")
    ax.set_title(f"Combined token surprisal and binary sum per sample (threshold = {threshold})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.legend()

    num_anomalies = sum(1 for v in values if v > threshold)
    summary_text = f"Anomalies: {num_anomalies}/{len(values)}"
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_unique_anomaly_patients_counter(
    results: List[dict],
    output_path: Path,
    threshold: float,
    min_red_bars: int = 3,
    top_n: Optional[int] = None,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot distribution of unique patients by number of red bars.

    A red bar is a slice/sample where (token_surprisal_hot_px + Binary_Sum_Heatmap) > threshold.
    Patients with red-bar count strictly greater than min_red_bars are considered anomaly patients.
    """
    if min_red_bars < 0:
        raise ValueError("min_red_bars must be >= 0")

    def derive_patient_id(item: dict) -> str:
        pid = patient_id_from_item(item)
        if pid:
            return strip_unknown_prefix(pid)
        explicit_pid = item.get("patient_id")
        if explicit_pid:
            return strip_unknown_prefix(str(explicit_pid))
        case_folder = item.get("case_folder")
        if case_folder:
            return strip_unknown_prefix(str(case_folder))
        return "unknown"

    patient_total_counts: Dict[str, int] = {}
    patient_red_counts: Dict[str, int] = {}

    processed_samples = 0
    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        token_val = item.get("token_surprisal_hot_px")
        binary_val = item.get("Binary_Sum_Heatmap")
        if token_val is None or binary_val is None:
            continue

        value = float(token_val) + float(binary_val)
        patient_id = derive_patient_id(item)
        patient_total_counts[patient_id] = patient_total_counts.get(patient_id, 0) + 1
        if value > threshold:
            patient_red_counts[patient_id] = patient_red_counts.get(patient_id, 0) + 1

        processed_samples += 1
        if top_n is not None and processed_samples >= top_n:
            break

    if not patient_total_counts:
        raise ValueError("No combined token_surprisal_hot_px + Binary_Sum_Heatmap values found")

    # Count how many unique patients have exactly N red bars.
    patients_by_red_count: Dict[int, int] = {}
    for pid in patient_total_counts:
        red_count = patient_red_counts.get(pid, 0)
        patients_by_red_count[red_count] = patients_by_red_count.get(red_count, 0) + 1

    max_red = max(patients_by_red_count.keys())
    red_counts_axis = list(range(0, max_red + 1))
    unique_patient_counts = [patients_by_red_count.get(rc, 0) for rc in red_counts_axis]
    bar_colors = ["red" if rc > min_red_bars else "#4C72B0" for rc in red_counts_axis]

    width = max(8, min(0.9 * len(red_counts_axis), 20))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(red_counts_axis))
    ax.bar(x_positions, unique_patient_counts, color=bar_colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([str(rc) for rc in red_counts_axis], fontsize=9)
    ax.set_xlabel("Number of red bars per patient")
    ax.set_ylabel("Unique patient count")
    ax.set_title(f"Unique patients by red-bar count (red if count > {min_red_bars})")

    # Mark the anomaly cutoff location between min_red_bars and min_red_bars+1.
    cutoff_x = min_red_bars + 0.5
    if red_counts_axis:
        ax.axvline(cutoff_x, color="black", linestyle="--", linewidth=1.2, label=f"anomaly: > {min_red_bars} red bars")
        ax.legend(loc="upper right")

    for idx, val in enumerate(unique_patient_counts):
        if val > 0:
            ax.text(idx, val + 0.02, str(val), ha="center", va="bottom", fontsize=8)

    anomaly_patients = sum(1 for pid in patient_total_counts if patient_red_counts.get(pid, 0) > min_red_bars)
    total_unique_patients = len(patient_total_counts)
    total_red_bars = sum(patient_red_counts.values())
    summary_text = (
        f"Unique patients: {total_unique_patients}\n"
        f"Anomaly patients (> {min_red_bars} red bars): {anomaly_patients}\n"
        f"Total red bars: {total_red_bars}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_unique_patients_sum_of_all_bars(
    results: List[dict],
    output_path: Path,
    threshold: float,
    top_n: Optional[int] = None,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot per-patient sum of all combined bars.

    Combined bar per sample is: token_surprisal_hot_px + Binary_Sum_Heatmap.
    The figure shows, for each unique patient, the sum of all such bars.
    """

    def derive_patient_id(item: dict) -> str:
        pid = patient_id_from_item(item)
        if pid:
            return strip_unknown_prefix(pid)
        explicit_pid = item.get("patient_id")
        if explicit_pid:
            return strip_unknown_prefix(str(explicit_pid))
        case_folder = item.get("case_folder")
        if case_folder:
            return strip_unknown_prefix(str(case_folder))
        return "unknown"

    patient_sums: Dict[str, float] = {}
    patient_is_orig: Dict[str, bool] = {}

    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        token_val = item.get("token_surprisal_hot_px")
        binary_val = item.get("Binary_Sum_Heatmap")
        if token_val is None or binary_val is None:
            continue

        combined_val = float(token_val) + float(binary_val)
        pid = derive_patient_id(item)
        patient_sums[pid] = patient_sums.get(pid, 0.0) + combined_val

        if pid not in patient_is_orig:
            cf = str(item.get("case_folder", ""))
            patient_is_orig[pid] = cf.lower() == "orig" or "orig" in pid.lower()

    if not patient_sums:
        raise ValueError("No patient sums found for combined token_surprisal_hot_px + Binary_Sum_Heatmap")

    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(pid, keep=60) for pid, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_is_orig.get(pid, False) for pid, _ in sorted_items]

    base_color = "#4C72B0"
    colors = ["red" if v > threshold else base_color for v in values]

    width = max(10, min(0.3 * len(labels), 70))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Sum of all bars per patient")
    ax.set_title(f"Unique patient sum of all bars (threshold = {threshold})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.legend(loc="upper right")

    num_patients = len(values)
    num_anomalies = sum(1 for v in values if v > threshold)
    orig_anomalies = sum(1 for v, o in zip(values, orig_flags) if v > threshold and o)
    summary_text = (
        f"Unique patients: {num_patients}\n"
        f"Global anomalies: {num_anomalies}\n"
        f"Orig global anomalies: {orig_anomalies}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_unique_patients_sum_of_mask_scores(
    results: List[dict],
    output_path: Path,
    threshold: float = 5000.0,
    top_n: Optional[int] = None,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot per-patient sum of masked score (lpips_input_recon_sum_mask)."""

    def derive_patient_id(item: dict) -> str:
        pid = patient_id_from_item(item)
        if pid:
            return strip_unknown_prefix(pid)
        explicit_pid = item.get("patient_id")
        if explicit_pid:
            return strip_unknown_prefix(str(explicit_pid))
        case_folder = item.get("case_folder")
        if case_folder:
            return strip_unknown_prefix(str(case_folder))
        return "unknown"

    patient_sums: Dict[str, float] = {}
    patient_is_orig: Dict[str, bool] = {}

    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        masked_val = item.get("lpips_input_recon_sum_mask")
        if masked_val is None:
            continue

        pid = derive_patient_id(item)
        patient_sums[pid] = patient_sums.get(pid, 0.0) + float(masked_val)

        if pid not in patient_is_orig:
            cf = str(item.get("case_folder", ""))
            patient_is_orig[pid] = cf.lower() == "orig" or "orig" in pid.lower()

    if not patient_sums:
        raise ValueError("No lpips_input_recon_sum_mask values found for unique-patient mask-score plot")

    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(pid, keep=60) for pid, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_is_orig.get(pid, False) for pid, _ in sorted_items]

    base_color = "#4C72B0"
    colors = ["red" if v > threshold else base_color for v in values]

    width = max(10, min(0.3 * len(labels), 70))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Sum of masked score per patient")
    ax.set_title(f"Unique patient sum of mask scores (threshold = {threshold})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.legend(loc="upper right")

    num_patients = len(values)
    num_anomalies = sum(1 for v in values if v > threshold)
    orig_anomalies = sum(1 for v, o in zip(values, orig_flags) if v > threshold and o)
    summary_text = (
        f"Unique patients: {num_patients}\n"
        f"Anomalies: {num_anomalies}\n"
        f"Orig anomalies: {orig_anomalies}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_patient_clamp_sums_filename(
    patient_summary: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    anomaly_threshold: float = 450.0,
    slice_start: int = 38,
    slice_end: int = 49,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot clamped pixel sums per patient using filename-based labels.

    Sums clamped_sum across slice_details whose slice index is in [slice_start, slice_end].
    Labels use the base filename (before _slice_) truncated to the last 80 characters.
    """
    if not patient_summary:
        raise ValueError("patient_summary is empty in JSON")

    def slice_idx_from_name(name: str) -> Optional[int]:
        return slice_index_from_filename(name)

    entries: List[Tuple[str, float, bool]] = []  # (label, sum, is_orig)
    for p in patient_summary:
        case_folder = p.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        slice_details = p.get("slice_details", []) or []
        if not slice_details:
            continue
        base_label = slice_details[0].get("filename", "unknown")
        base_label = patient_label_from_filename(base_label)

        range_sum = 0.0
        for sd in slice_details:
            fname = sd.get("filename", "")
            idx = slice_idx_from_name(fname)
            if idx is None:
                continue
            if slice_start <= idx <= slice_end:
                range_sum += float(sd.get("clamped_sum", 0.0))
        # fallback to total if no slices in range
        if range_sum == 0.0:
            range_sum = float(p.get("total_clamped_pixel_sum", 0.0))

        is_orig = "orig" in base_label.lower() or str(p.get("case_folder", "")).lower() == "orig"
        entries.append((truncate_label(base_label, keep=80), range_sum, is_orig))

    sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_entries = sorted_entries[:top_n]

    labels = [e[0] for e in sorted_entries]
    values = [e[1] for e in sorted_entries]
    orig_flags = [e[2] for e in sorted_entries]
    base_color = "#1f77b4"  # blue for bars not exceeding threshold (non-orig)
    colors = ["red" if v > anomaly_threshold else ("green" if o else base_color) for v, o in zip(values, orig_flags)]

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Total clamped pixel sum (slices 38-49)")
    ax.set_title("Patient_clumped_sum_threshold")

    ax.axhline(anomaly_threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={anomaly_threshold}")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="red", label=f"anomaly (>{anomaly_threshold})"),
        Patch(facecolor="green", label="orig (normal)"),
        Patch(facecolor=base_color, label="other (<= threshold)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_individual_patient_clamp_sums(
    patient_summary: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    anomaly_threshold: float = 450.0,
    slice_start: int = 38,
    slice_end: int = 49,
    allowed_case_folders: Optional[Set[str]] = None,
    lpips_threshold: Optional[float] = None,
    allowed_patient_ids: Optional[Set[str]] = None,
    filter_note: Optional[str] = None,
) -> None:
    """Plot clamped pixel sums for each INDIVIDUAL patient (by unique filename base).

    Each unique patient (e.g., SyntheticVariations_RandomNoise_oldAcq_a93202c46cc57831_RandomNoise)
    gets its own bar. Sums clamped_sum for slices in [slice_start, slice_end].
    """
    if not patient_summary:
        raise ValueError("patient_summary is empty in JSON")

    def slice_idx_from_name(name: str) -> Optional[int]:
        return slice_index_from_filename(name)

    def extract_patient_base(filename: str) -> str:
        """Extract patient identifier while preserving suffix modifiers."""
        return patient_label_from_filename(filename)

    # Aggregate by individual patient (unique filename base)
    # NOW using anomaly_pixel_count (binary count) as PRIMARY metric
    patient_sums: Dict[str, float] = {}
    patient_info: Dict[str, Dict[str, object]] = {}
    for p in patient_summary:
        case_folder = p.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        slice_details = p.get("slice_details", []) or []
        for sd in slice_details:
            fname = sd.get("filename", "")
            if not fname:
                continue
            idx = slice_idx_from_name(fname)
            if idx is None:
                continue
            if not (slice_start <= idx <= slice_end):
                continue
            patient_base = extract_patient_base(fname)
            if allowed_patient_ids is not None and patient_base not in allowed_patient_ids:
                continue
            # PRIMARY: Use anomaly_pixel_count if available, else fall back to clamped_sum
            pixel_count = sd.get("anomaly_pixel_count", None)
            if pixel_count is not None:
                value = float(pixel_count)
            else:
                value = float(sd.get("clamped_sum", 0.0))
            patient_sums[patient_base] = patient_sums.get(patient_base, 0.0) + value
            if patient_base not in patient_info:
                is_orig = str(case_folder or "").lower() == "orig" or "orig" in patient_base.lower()
                patient_info[patient_base] = {
                    "case_folder": case_folder,
                    "is_orig": is_orig,
                    "lpips_sum": float(p.get("total_lpips_input_recon_sum_mask", 0.0)) if p.get("total_lpips_input_recon_sum_mask") is not None else None,
                }

    if not patient_sums:
        raise ValueError("No individual patients found with slices in range")

    # Sort by descending sum
    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(k, keep=60) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_info.get(k, {}).get("is_orig", False) for k, _ in sorted_items]
    lpips_sums = [patient_info.get(k, {}).get("lpips_sum") for k, _ in sorted_items]

    base_color = "#999999"  # neutral normal for non-orig, under-threshold
    orig_color = "#1f77b4"  # blue for orig patients

    def choose_color(value: float, is_orig: bool) -> str:
        if value > anomaly_threshold:
            return "red"
        if is_orig:
            return orig_color
        return base_color

    colors = [choose_color(v, o) for v, o in zip(values, orig_flags)]

    width = max(12, min(0.3 * len(labels), 80))
    fig, ax = plt.subplots(figsize=(width, 7))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)

    # Color patient labels: purple for anomalies, green for orig normals
    for text, val, is_orig in zip(ax.get_xticklabels(), values, orig_flags):
        if val > anomaly_threshold:
            text.set_color("purple")
        elif is_orig:
            text.set_color(orig_color)
        else:
            text.set_color("black")

    ax.set_ylabel(f"Anomaly pixel count - First heatmap (slices {slice_start}-{slice_end})")
    ax.set_title("individual_Patient_first_heatmap_pixel_count")

    ax.axhline(anomaly_threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={anomaly_threshold}")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="red", label=f"anomaly (>{anomaly_threshold})"),
        Patch(facecolor=orig_color, label="orig"),
        Patch(facecolor=base_color, label="other (<= threshold)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    # Summary stats
    num_patients = len(values)
    num_anomalies = sum(1 for v in values if v > anomaly_threshold)
    orig_anomalies = sum(1 for (label, v) in sorted_items if v > anomaly_threshold and patient_info.get(label, {}).get("is_orig", False))
    other_anomalies = num_anomalies - orig_anomalies
    num_orig = sum(orig_flags)
    summary_text = (
        f"Total patients: {num_patients}\n"
        f"Anomalies: {num_anomalies}\n"
        f"Orig anomalies: {orig_anomalies}\n"
        f"Other anomalies: {other_anomalies}"
    )
    if allowed_patient_ids is not None:
        summary_text += f"\nFiltered set: {num_patients} (Binary pass only)"
    if filter_note:
        summary_text += f"\n{filter_note}"
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_patient_lpips_in_rec_sum(
    patient_summary: List[dict],
    output_path: Path,
    threshold: float = 2000.0,
    top_n: Optional[int] = None,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot total_lpips_input_recon_sum_mask per patient.

    Bars red with purple labels when above threshold; blue otherwise.
    """
    if not patient_summary:
        raise ValueError("patient_summary is empty in JSON")

    entries: List[Tuple[str, float]] = []  # (label, value)
    for p in patient_summary:
        if not matches_case_folder(p.get("case_folder"), allowed_case_folders):
            continue
        val = p.get("total_lpips_input_recon_sum_mask")
        if val is None:
            continue
        label = truncate_label(patient_display_label(p), keep=80)
        entries.append((label, float(val)))

    if not entries:
        raise ValueError("No patients with total_lpips_input_recon_sum_mask available after filtering")

    entries.sort(key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        entries = entries[:top_n]

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    orig_flags = ["orig" in lbl.lower() for lbl in labels]

    # Bar colors remain anomaly-vs-normal; text colors highlight orig patients
    colors = ["red" if v > threshold else "blue" for v in values]

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    # Label coloring: green for orig normals, purple for anomalies (including orig anomalies)
    for text, val, is_orig in zip(ax.get_xticklabels(), values, orig_flags):
        if val > threshold:
            text.set_color("purple")
        elif is_orig:
            text.set_color("green")

    ax.set_ylabel("Total lpips_input_recon_sum_mask")
    ax.set_title("Individual patient LPIPS(in,recon) masked sums")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Threshold_LPIPS_in_rec={threshold}")
    # Summary box: total anomalies, orig anomalies, other anomalies
    num_anomalies = sum(1 for v in values if v > threshold)
    orig_anomalies = sum(1 for v, is_orig in zip(values, orig_flags) if v > threshold and is_orig)
    other_anomalies = num_anomalies - orig_anomalies
    summary_text = (
        f"Total patients: {len(values)}\n"
        f"Anomalies: {num_anomalies}\n"
        f"Orig anomalies: {orig_anomalies}\n"
        f"Other anomalies: {other_anomalies}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax.legend()
    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def collect_patient_binary_sums(
    results: List[dict],
    allowed_case_folders: Optional[Set[str]],
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Aggregate Binary_Sum_Heatmap per patient and mark orig patients."""
    if not results:
        raise ValueError("results list is empty in JSON")

    patient_sums: Dict[str, float] = {}
    patient_is_orig: Dict[str, bool] = {}

    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        val = item.get("Binary_Sum_Heatmap")
        if val is None:
            continue
        pid_raw = patient_id_from_item(item)
        if pid_raw is None:
            continue
        pid = strip_unknown_prefix(pid_raw)
        patient_sums[pid] = patient_sums.get(pid, 0.0) + float(val)
        if pid not in patient_is_orig:
            cf = str(item.get("case_folder", ""))
            patient_is_orig[pid] = cf.lower() == "orig" or "orig" in pid.lower()

    if not patient_sums:
        raise ValueError("No Binary_Sum_Heatmap values found for given filters")

    return patient_sums, patient_is_orig


def collect_patient_sharpness_totals(
    results: List[dict],
    allowed_case_folders: Optional[Set[str]],
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Aggregate total sharpness per patient from slice-level sharpness_score."""
    if not results:
        raise ValueError("results list is empty in JSON")

    patient_sums: Dict[str, float] = {}
    patient_is_orig: Dict[str, bool] = {}

    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        val = item.get("sharpness_score")
        if val is None:
            continue
        pid_raw = patient_id_from_item(item)
        if pid_raw is None:
            continue
        pid = strip_unknown_prefix(pid_raw)
        patient_sums[pid] = patient_sums.get(pid, 0.0) + float(val)
        if pid not in patient_is_orig:
            cf = str(item.get("case_folder", ""))
            patient_is_orig[pid] = cf.lower() == "orig" or "orig" in pid.lower()

    if not patient_sums:
        raise ValueError("No sharpness_score values found for given filters")

    return patient_sums, patient_is_orig


def collect_patient_clamp_totals_from_summary(
    patient_summary: List[dict],
    allowed_case_folders: Optional[Set[str]],
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Aggregate total clamped pixel sum per patient using patient_summary entries."""
    if not patient_summary:
        raise ValueError("patient_summary is empty; cannot compute clamped totals")

    def derive_pid(entry: dict) -> str:
        # First try to derive from slice_details filename (consistent with sharpness/binary_sum collection)
        slice_details = entry.get("slice_details") or []
        if slice_details:
            fname = slice_details[0].get("filename") or ""
            if fname:
                return patient_label_from_filename(fname)
        # Fallback to explicit patient_id or case_folder
        pid = entry.get("patient_id") or entry.get("case_folder")
        if pid:
            return strip_unknown_prefix(str(pid))
        return "unknown"

    totals: Dict[str, float] = {}
    is_orig: Dict[str, bool] = {}
    for entry in patient_summary:
        if not matches_case_folder(entry.get("case_folder"), allowed_case_folders):
            continue
        pid = derive_pid(entry)
        val_raw = entry.get("total_clamped_pixel_sum")
        if val_raw is None:
            val_raw = entry.get("total_clamped_sum") or entry.get("total_clamped")
        # Fallbacks when totals are missing/zero: use slice_details clamped sums, or anomaly pixel counts
        if val_raw is None or float(val_raw) == 0.0:
            slice_details = entry.get("slice_details") or []
            val_from_clamp = sum(
                float(sd.get("clamped_sum") or sd.get("clamped_pixel_sum") or 0.0)
                for sd in slice_details
            )
            if val_from_clamp == 0.0:
                val_from_clamp = sum(
                    float(sd.get("anomaly_pixel_count") or sd.get("num_pixels_above_clamp_thresh") or 0.0)
                    for sd in slice_details
                )
            if val_from_clamp == 0.0:
                val_from_clamp = float(entry.get("total_pixels_above_thresh") or 0.0)
            val_raw = val_from_clamp
        val = float(val_raw or 0.0)
        totals[pid] = val
        cf = str(entry.get("case_folder", ""))
        is_orig[pid] = cf.lower() == "orig" or "orig" in pid.lower()

    if not totals:
        raise ValueError("No clamped totals found after filtering")

    return totals, is_orig


def patients_below_binary_threshold(
    results: List[dict],
    threshold: float,
    allowed_case_folders: Optional[Set[str]],
) -> Set[str]:
    """Return patient ids whose Binary_Sum_Heatmap total is <= threshold."""
    patient_sums, _ = collect_patient_binary_sums(results, allowed_case_folders)
    return {pid for pid, total in patient_sums.items() if total <= threshold}


def compute_binary_token_patient_sensitivity(
    results: List[dict],
    threshold: float,
    allowed_case_folders: Optional[Set[str]] = None,
) -> Dict[str, object]:
    """Compute patient-level TP/FN and sensitivity for Binary+Token totals.

        Ground truth assumption:
            - `orig` patients are normal.
            - `Test_Samples_FastMRI` cohort is also normal.
                        - `Validation_samples` cohort is also normal.
            - Remaining patients are anomaly patients.
    Prediction rule:
      - patient total Binary_Sum_Heatmap > threshold => predicted anomaly.
    """
    if not results:
        raise ValueError("results list is empty in JSON")

    patient_sums: Dict[str, float] = {}
    patient_case_folders: Dict[str, Set[str]] = {}
    patient_categories: Dict[str, Dict[str, int]] = {}

    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        val = item.get("Binary_Sum_Heatmap")
        if val is None:
            continue

        pid_raw = patient_id_from_item(item)
        if pid_raw is None:
            continue
        pid = strip_unknown_prefix(pid_raw)
        patient_sums[pid] = patient_sums.get(pid, 0.0) + float(val)

        case_folder = str(item.get("case_folder", ""))
        if pid not in patient_case_folders:
            patient_case_folders[pid] = set()
        if case_folder:
            patient_case_folders[pid].add(case_folder)

        category = str(item.get("category", "")).strip()
        if category:
            if pid not in patient_categories:
                patient_categories[pid] = {}
            patient_categories[pid][category] = patient_categories[pid].get(category, 0) + 1

    if not patient_sums:
        raise ValueError("No Binary_Sum_Heatmap values found for sensitivity calculation")

    def patient_is_orig(pid: str) -> bool:
        folders = patient_case_folders.get(pid, set())
        if any(str(cf).lower() == "orig" for cf in folders):
            return True
        return "orig" in pid.lower()

    def patient_is_test_samples_normal(pid: str) -> bool:
        cats = patient_categories.get(pid, {})
        if not cats:
            return False
        return any("test_samples_fastmri" in str(cat).lower() for cat in cats.keys())

    def patient_is_validation_samples_normal(pid: str) -> bool:
        cats = patient_categories.get(pid, {})
        if any("validation_samples" in str(cat).lower() for cat in cats.keys()):
            return True
        folders = patient_case_folders.get(pid, set())
        if any("validation_samples" in str(cf).lower() for cf in folders):
            return True
        return "validation_samples" in pid.lower()

    def patient_category(
        pid: str,
        is_orig_patient: bool,
        is_test_samples_normal: bool,
        is_validation_samples_normal: bool,
    ) -> str:
        if is_test_samples_normal:
            return "Normal (Test_Samples_FastMRI)"
        if is_validation_samples_normal:
            return "Normal (Validation_samples)"
        if is_orig_patient:
            return "Normal (orig)"
        cats = patient_categories.get(pid, {})
        if cats:
            return sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        folders = sorted(cf for cf in patient_case_folders.get(pid, set()) if cf and str(cf).lower() != "orig")
        if folders:
            return folders[0]
        return "Unknown anomaly"

    per_category: Dict[str, Dict[str, int]] = {}
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_tn = 0

    for pid, total in patient_sums.items():
        is_orig_patient = patient_is_orig(pid)
        is_test_samples_normal = patient_is_test_samples_normal(pid)
        is_validation_samples_normal = patient_is_validation_samples_normal(pid)
        is_normal_gt = is_orig_patient or is_test_samples_normal or is_validation_samples_normal
        is_anomaly_gt = not is_normal_gt
        pred_anomaly = total > threshold
        category_name = patient_category(
            pid,
            is_orig_patient,
            is_test_samples_normal,
            is_validation_samples_normal,
        )

        if category_name not in per_category:
            per_category[category_name] = {
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "tn": 0,
                "support": 0,
                "total": 0,
            }
        bucket = per_category[category_name]
        bucket["total"] += 1

        if is_anomaly_gt:
            bucket["support"] += 1
            if pred_anomaly:
                bucket["tp"] += 1
                total_tp += 1
            else:
                bucket["fn"] += 1
                total_fn += 1
        else:
            if pred_anomaly:
                bucket["fp"] += 1
                total_fp += 1
            else:
                bucket["tn"] += 1
                total_tn += 1

    anomaly_support = total_tp + total_fn
    normal_support = total_tn + total_fp
    sensitivity = (total_tp / anomaly_support) if anomaly_support else 0.0
    specificity = (total_tn / normal_support) if normal_support else 0.0

    rows: List[Dict[str, object]] = []
    for name, counts in sorted(per_category.items(), key=lambda kv: (-kv[1]["support"], kv[0])):
        support = counts["support"]
        tp = counts["tp"]
        fn = counts["fn"]
        row_sens = (tp / support) if support else 0.0
        rows.append(
            {
                "category": name,
                "tp": tp,
                "fn": fn,
                "support": support,
                "sensitivity": row_sens,
                "fp": counts["fp"],
                "tn": counts["tn"],
                "total": counts["total"],
            }
        )

    return {
        "threshold": float(threshold),
        "total_patients": len(patient_sums),
        "anomaly_patients": anomaly_support,
        "normal_patients": normal_support,
        "detected_tp": total_tp,
        "missed_fn": total_fn,
        "false_positive": total_fp,
        "true_negative": total_tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "per_category": rows,
    }


def plot_binary_token_sensitivity_table(
    sensitivity_summary: Dict[str, object],
    output_path: Path,
) -> None:
    """Render a summary table figure with TP/FN/sensitivity values."""
    rows = sensitivity_summary.get("per_category", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No category rows available for sensitivity table")

    anomaly_support = int(sensitivity_summary.get("anomaly_patients", 0))
    tp = int(sensitivity_summary.get("detected_tp", 0))
    fn = int(sensitivity_summary.get("missed_fn", 0))
    sensitivity = float(sensitivity_summary.get("sensitivity", 0.0))

    header = ["Anomaly", "Detected (TP)", "Missed (FN)", "Sensitivity"]
    body: List[List[str]] = [
        [
            "Overall",
            f"{tp}/{anomaly_support}",
            f"{fn}/{anomaly_support}",
            f"{100.0 * sensitivity:.1f}%",
        ]
    ]
    for row in rows:
        support = int(row.get("support", 0))
        r_tp = int(row.get("tp", 0))
        r_fn = int(row.get("fn", 0))
        r_sens = float(row.get("sensitivity", 0.0))
        body.append(
            [
                str(row.get("category", "Unknown")),
                f"{r_tp}/{support}",
                f"{r_fn}/{support}",
                f"{100.0 * r_sens:.1f}%",
            ]
        )

    fig_h = max(3.5, min(0.45 * (len(body) + 1), 20.0))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=body,
        colLabels=header,
        colLoc="center",
        cellLoc="center",
        loc="center",
        colWidths=[0.48, 0.18, 0.18, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1F1F1F")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx == 1:
            cell.set_facecolor("#E9F2FF")
            if col_idx == 0:
                cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8F8F8" if row_idx % 2 == 0 else "#FFFFFF")

    threshold = float(sensitivity_summary.get("threshold", 0.0))
    ax.set_title(
        f"Binary+Token patient detection summary (threshold = {threshold})",
        fontsize=12,
        pad=16,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_binary_token_sensitivity_outputs(
    sensitivity_summary: Dict[str, object],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write sensitivity summary to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "BinaryToken_patient_sensitivity_summary.json"
    csv_path = output_dir / "BinaryToken_patient_sensitivity_summary.csv"

    with json_path.open("w") as f:
        json.dump(sensitivity_summary, f, indent=2)

    rows = sensitivity_summary.get("per_category", [])
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "tp", "fn", "support", "sensitivity_percent", "fp", "tn", "total"])
        writer.writerow(
            [
                "Overall",
                int(sensitivity_summary.get("detected_tp", 0)),
                int(sensitivity_summary.get("missed_fn", 0)),
                int(sensitivity_summary.get("anomaly_patients", 0)),
                100.0 * float(sensitivity_summary.get("sensitivity", 0.0)),
                int(sensitivity_summary.get("false_positive", 0)),
                int(sensitivity_summary.get("true_negative", 0)),
                int(sensitivity_summary.get("total_patients", 0)),
            ]
        )
        if isinstance(rows, list):
            for row in rows:
                writer.writerow(
                    [
                        row.get("category", "Unknown"),
                        int(row.get("tp", 0)),
                        int(row.get("fn", 0)),
                        int(row.get("support", 0)),
                        100.0 * float(row.get("sensitivity", 0.0)),
                        int(row.get("fp", 0)),
                        int(row.get("tn", 0)),
                        int(row.get("total", 0)),
                    ]
                )

    return json_path, csv_path


def build_patient_detection_overview_rows(
    results: List[dict],
    patient_summary: List[dict],
    top_n: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Build per-patient detection rows for table output.

    Preferred source is `patient_summary` from inference JSON (contains
    `detected_based_on_thresholds` and bbox-derived metrics). If unavailable,
    it falls back to slice-level aggregation from `results`.
    """
    def _cohort_label(case_folder: str, category: str, patient_id: str) -> str:
        joined = " ".join([str(case_folder), str(category), str(patient_id)]).lower()
        if "test_samples_fastmri" in joined or "test set" in joined:
            return "Test set"
        if "validation_samples" in joined or "validation set" in joined:
            return "Validation set"
        return "Other"

    def _build_rows_from_results(items: List[dict]) -> List[Dict[str, object]]:
        grouped: Dict[str, Dict[str, object]] = {}
        for item in items:
            pid = patient_id_from_item(item)
            if pid is None:
                continue
            key = strip_unknown_prefix(pid)
            if key not in grouped:
                grouped[key] = {
                    "patient_id": key,
                    "case_folder": str(item.get("case_folder", "") or ""),
                    "category": str(item.get("category", "") or ""),
                    "detected": False,
                    "num_slices": 0,
                    "num_slices_with_ground_truth_bbox": 0,
                    "num_true_positive_slices": 0,
                    "num_no_bbox_slices": 0,
                    "highlighted_anomaly_pixels_no_bbox": 0,
                    "highlighted_anomaly_pixels_binary_token_total": 0,
                    "sum_ground_truth_bbox_pixels": 0,
                    "sum_predicted_anomaly_pixels_inside_bbox": 0,
                    "sum_predicted_anomaly_pixels_outside_bbox": 0,
                    "fp_pixel_ratio_percent": 0.0,
                    "normal_no_bbox_cohort": "Other",
                    "precision": None,
                    "false_positive_ratio": None,
                }

            bucket = grouped[key]
            bucket["num_slices"] = int(bucket["num_slices"]) + 1
            has_gt = bool(item.get("has_ground_truth_bbox", False))
            tp = bool(item.get("true_positive", False))
            highlighted = int(item.get("highlighted_anomaly_pixels_binary_token", item.get("Binary_Sum_Heatmap", 0)) or 0)
            bucket["highlighted_anomaly_pixels_binary_token_total"] = int(bucket["highlighted_anomaly_pixels_binary_token_total"]) + highlighted
            if has_gt:
                bucket["num_slices_with_ground_truth_bbox"] = int(bucket["num_slices_with_ground_truth_bbox"]) + 1
                bucket["sum_ground_truth_bbox_pixels"] = int(bucket["sum_ground_truth_bbox_pixels"]) + int(item.get("ground_truth_bbox_pixels", 0) or 0)
                bucket["sum_predicted_anomaly_pixels_inside_bbox"] = int(bucket["sum_predicted_anomaly_pixels_inside_bbox"]) + int(item.get("predicted_anomaly_pixels_inside_bbox", 0) or 0)
                bucket["sum_predicted_anomaly_pixels_outside_bbox"] = int(bucket["sum_predicted_anomaly_pixels_outside_bbox"]) + int(item.get("predicted_anomaly_pixels_outside_bbox", 0) or 0)
            else:
                bucket["num_no_bbox_slices"] = int(bucket["num_no_bbox_slices"]) + 1
                bucket["highlighted_anomaly_pixels_no_bbox"] = int(bucket["highlighted_anomaly_pixels_no_bbox"]) + highlighted
            if tp:
                bucket["num_true_positive_slices"] = int(bucket["num_true_positive_slices"]) + 1
                bucket["detected"] = True

            bucket["normal_no_bbox_cohort"] = _cohort_label(
                str(bucket.get("case_folder", "") or ""),
                str(bucket.get("category", "") or ""),
                str(bucket.get("patient_id", "") or ""),
            )

        for bucket in grouped.values():
            no_bbox_slices = int(bucket.get("num_no_bbox_slices", 0) or 0)
            highlighted_no_bbox = int(bucket.get("highlighted_anomaly_pixels_no_bbox", 0) or 0)
            bucket["fp_pixel_ratio_percent"] = float(
                compute_fp_pixel_ratio_percent(highlighted_no_bbox, no_bbox_slices)
            )
            inside_px = int(bucket.get("sum_predicted_anomaly_pixels_inside_bbox", 0) or 0)
            outside_px = int(bucket.get("sum_predicted_anomaly_pixels_outside_bbox", 0) or 0)
            tp_event = bool(bucket.get("detected", False))
            fp_ratio, paper_precision, paper_f1 = compute_paper_precision_f1(tp_event, inside_px, outside_px)
            bucket["false_positive_ratio"] = fp_ratio
            bucket["precision"] = paper_precision
            bucket["pixel_precision"] = float(paper_precision or 0.0)
            bucket["pixel_tp_ratio"] = 1.0 if tp_event else 0.0
            bucket["pixel_f1_score"] = float(paper_f1 or 0.0)

        return list(grouped.values())

    rows: List[Dict[str, object]] = []

    for entry in patient_summary:
        pid = str(entry.get("patient_id") or patient_display_label(entry) or "unknown")
        detected = entry.get("detected_based_on_thresholds")
        if detected is None:
            tp_slices = int(entry.get("num_true_positive_slices", 0) or 0)
            detected = tp_slices > 0
        case_folder = str(entry.get("case_folder", "") or "")
        category = str(entry.get("category", "") or "")
        highlighted_no_bbox = int(entry.get("sum_highlighted_anomaly_pixels_no_bbox", 0) or 0)
        highlighted_all = int(entry.get("sum_highlighted_anomaly_pixels_binary_token", 0) or 0)
        no_bbox_slices = int(entry.get("num_slices", 0) or 0) - int(entry.get("num_slices_with_ground_truth_bbox", 0) or 0)
        no_bbox_slices = max(no_bbox_slices, 0)
        cohort = _cohort_label(case_folder, category, pid)
        rows.append(
            {
                "patient_id": pid,
                "case_folder": case_folder,
                "category": category,
                "detected": bool(detected),
                "num_slices": int(entry.get("num_slices", 0) or 0),
                "num_slices_with_ground_truth_bbox": int(entry.get("num_slices_with_ground_truth_bbox", 0) or 0),
                "num_true_positive_slices": int(entry.get("num_true_positive_slices", 0) or 0),
                "num_no_bbox_slices": no_bbox_slices,
                "highlighted_anomaly_pixels_no_bbox": highlighted_no_bbox,
                "highlighted_anomaly_pixels_binary_token_total": highlighted_all,
                "sum_ground_truth_bbox_pixels": int(entry.get("sum_ground_truth_bbox_pixels", 0) or 0),
                "sum_predicted_anomaly_pixels_inside_bbox": int(entry.get("sum_predicted_anomaly_pixels_inside_bbox", 0) or 0),
                "sum_predicted_anomaly_pixels_outside_bbox": int(entry.get("sum_predicted_anomaly_pixels_outside_bbox", 0) or 0),
                "fp_pixel_ratio_percent": float(
                    compute_fp_pixel_ratio_percent(highlighted_no_bbox, no_bbox_slices)
                ),
                "normal_no_bbox_cohort": cohort,
                "precision": entry.get("precision"),
                "false_positive_ratio": entry.get("false_positive_ratio"),
            }
        )

    for row in rows:
        inside_px = int(row.get("sum_predicted_anomaly_pixels_inside_bbox", 0) or 0)
        outside_px = int(row.get("sum_predicted_anomaly_pixels_outside_bbox", 0) or 0)
        tp_event = bool(row.get("detected", False))
        fp_ratio, paper_precision, paper_f1 = compute_paper_precision_f1(tp_event, inside_px, outside_px)
        row["false_positive_ratio"] = fp_ratio
        row["precision"] = paper_precision
        row["pixel_precision"] = float(paper_precision or 0.0)
        row["pixel_tp_ratio"] = 1.0 if tp_event else 0.0
        row["pixel_f1_score"] = float(paper_f1 or 0.0)

    if results:
        rows_from_results = _build_rows_from_results(results)
        unique_result_patients = len(rows_from_results)

        summary_collapsed = False
        if rows and len(rows) == 1 and unique_result_patients > 1:
            summary_num_slices = int(rows[0].get("num_slices", 0) or 0)
            summary_collapsed = summary_num_slices >= max(len(results) - 1, 1)

        if not rows or summary_collapsed:
            rows = rows_from_results

    rows = sorted(
        rows,
        key=lambda r: (
            0 if bool(r.get("detected", False)) else 1,
            -int(r.get("num_true_positive_slices", 0) or 0),
            str(r.get("patient_id", "")),
        ),
    )
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def build_normal_nobbox_fp_overview_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate highlighted no-bbox pixels for normal cohorts (Test/Validation)."""
    cohorts = ["Test set", "Validation set"]
    out: List[Dict[str, object]] = []
    for cohort in cohorts:
        selected = [r for r in rows if r.get("normal_no_bbox_cohort") == cohort]
        no_bbox_patients = [r for r in selected if int(r.get("num_no_bbox_slices", 0) or 0) > 0]
        total_fp_pixels = sum(int(r.get("highlighted_anomaly_pixels_no_bbox", 0) or 0) for r in selected)
        out.append(
            {
                "cohort": cohort,
                "num_patients": len(selected),
                "num_patients_with_no_bbox_slices": len(no_bbox_patients),
                "total_highlighted_fp_pixels_no_bbox": int(total_fp_pixels),
                "mean_highlighted_fp_pixels_per_patient": (float(total_fp_pixels) / float(len(selected))) if selected else 0.0,
            }
        )

    total_fp = sum(int(r.get("total_highlighted_fp_pixels_no_bbox", 0) or 0) for r in out)
    total_patients = sum(int(r.get("num_patients", 0) or 0) for r in out)
    total_no_bbox_patients = sum(int(r.get("num_patients_with_no_bbox_slices", 0) or 0) for r in out)
    out.insert(
        0,
        {
            "cohort": "OVERALL_NORMAL_NO_BBOX",
            "num_patients": int(total_patients),
            "num_patients_with_no_bbox_slices": int(total_no_bbox_patients),
            "total_highlighted_fp_pixels_no_bbox": int(total_fp),
            "mean_highlighted_fp_pixels_per_patient": (float(total_fp) / float(total_patients)) if total_patients else 0.0,
        },
    )
    return out


def plot_patient_detection_overview_table(
    rows: List[Dict[str, object]],
    output_path: Path,
    source_name: str,
) -> None:
    """Render a table listing which patients were detected."""
    if not rows:
        raise ValueError("No patient detection rows available for table")

    total_patients = len(rows)
    detected_patients = sum(1 for r in rows if bool(r.get("detected", False)))
    total_no_bbox_slices = sum(int(r.get("num_no_bbox_slices", 0) or 0) for r in rows)
    total_no_bbox_fp_px = sum(int(r.get("highlighted_anomaly_pixels_no_bbox", 0) or 0) for r in rows)
    avg_fp_pixel_ratio_percent = compute_fp_pixel_ratio_percent(
        highlighted_fp_pixels=total_no_bbox_fp_px,
        no_bbox_slices=total_no_bbox_slices,
    )

    header = [
        "Patient",
        "Cohort",
        "Detected",
        "TP slices",
        "GT-bbox slices",
        "No-bbox slices",
        "No-bbox FP px",
        "FP pixel ratio",
        "Total slices",
    ]
    body: List[List[str]] = []
    for row in rows:
        body.append(
            [
                truncate_label(str(row.get("patient_id", "unknown")), keep=70),
                str(row.get("normal_no_bbox_cohort", "Other")),
                "YES" if bool(row.get("detected", False)) else "NO",
                str(int(row.get("num_true_positive_slices", 0) or 0)),
                str(int(row.get("num_slices_with_ground_truth_bbox", 0) or 0)),
                str(int(row.get("num_no_bbox_slices", 0) or 0)),
                str(int(row.get("highlighted_anomaly_pixels_no_bbox", 0) or 0)),
                f"{float(row.get('fp_pixel_ratio_percent', 0.0) or 0.0):.2f}%",
                str(int(row.get("num_slices", 0) or 0)),
            ]
        )

    fig_h = max(4.5, min(0.38 * (len(body) + 2), 30.0))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=body,
        colLabels=header,
        colLoc="center",
        cellLoc="center",
        loc="center",
        colWidths=[0.31, 0.11, 0.08, 0.09, 0.10, 0.10, 0.12, 0.11, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1F1F1F")
            cell.set_text_props(color="white", weight="bold")
            continue

        if row_idx % 2 == 0:
            cell.set_facecolor("#F8F8F8")
        else:
            cell.set_facecolor("#FFFFFF")

        if col_idx == 2:
            txt = cell.get_text().get_text().strip().upper()
            if txt == "YES":
                cell.set_facecolor("#FDECEC")
                cell.set_text_props(color="#8B0000", weight="bold")
            else:
                cell.set_facecolor("#EDF7ED")
                cell.set_text_props(color="#1B5E20")

        if col_idx == 1:
            cohort_txt = cell.get_text().get_text().strip().lower()
            if cohort_txt == "test set":
                cell.set_facecolor("#E8F1FF")
            elif cohort_txt == "validation set":
                cell.set_facecolor("#FFF4E5")

        if col_idx == 6:
            try:
                fp_px = int(float(cell.get_text().get_text().strip()))
            except ValueError:
                fp_px = 0
            if fp_px > 0:
                cell.set_facecolor("#FFE6E6")
                cell.set_text_props(color="#8B0000", weight="bold")

        if col_idx == 7:
            try:
                fp_ratio = float(cell.get_text().get_text().strip().replace("%", ""))
            except ValueError:
                fp_ratio = 0.0
            if fp_ratio > 0.0:
                cell.set_facecolor("#FFF0F0")
                cell.set_text_props(color="#8B0000", weight="bold")

    ax.set_title(
        (
            f"Patient detection overview ({source_name})\n"
            f"Detected: {detected_patients}/{total_patients} | "
            f"Avg FP pixel ratio: {avg_fp_pixel_ratio_percent:.2f}%"
        ),
        fontsize=12,
        pad=16,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_normal_nobbox_fp_overview_table(
    overview_rows: List[Dict[str, object]],
    output_path: Path,
    source_name: str,
) -> None:
    """Render cohort-level overview of highlighted FP pixels in normal no-bbox cohorts."""
    if not overview_rows:
        raise ValueError("No normal no-bbox FP overview rows available")

    header = [
        "Cohort",
        "Patients",
        "Patients w/ no-bbox",
        "Total highlighted FP px",
        "Mean FP px / patient",
    ]
    body: List[List[str]] = []
    for row in overview_rows:
        body.append(
            [
                str(row.get("cohort", "Unknown")),
                str(int(row.get("num_patients", 0) or 0)),
                str(int(row.get("num_patients_with_no_bbox_slices", 0) or 0)),
                str(int(row.get("total_highlighted_fp_pixels_no_bbox", 0) or 0)),
                f"{float(row.get('mean_highlighted_fp_pixels_per_patient', 0.0)):.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, max(4.0, 1.2 + 0.8 * len(body))))
    ax.axis("off")
    table = ax.table(
        cellText=body,
        colLabels=header,
        colLoc="center",
        cellLoc="center",
        loc="center",
        colWidths=[0.30, 0.12, 0.18, 0.22, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1F1F1F")
            cell.set_text_props(color="white", weight="bold")
            continue
        if row_idx == 1:
            cell.set_facecolor("#E9F2FF")
            if col_idx == 0:
                cell.set_text_props(weight="bold")
            continue
        cell.set_facecolor("#F8F8F8" if row_idx % 2 == 0 else "#FFFFFF")
        if col_idx == 3:
            try:
                val = int(float(cell.get_text().get_text()))
            except ValueError:
                val = 0
            if val > 0:
                cell.set_facecolor("#FFE6E6")
                cell.set_text_props(color="#8B0000", weight="bold")

    ax.set_title(
        f"Normal no-bbox false-positive pixel overview ({source_name})",
        fontsize=12,
        pad=14,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_patient_detection_overview_outputs(
    rows: List[Dict[str, object]],
    output_dir: Path,
    source_name: str,
) -> Tuple[Path, Path]:
    """Write patient detection overview rows to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "Patient_detection_overview.json"
    csv_path = output_dir / "Patient_detection_overview.csv"

    total_patients = len(rows)
    detected_patients = sum(1 for r in rows if bool(r.get("detected", False)))

    with json_path.open("w") as f:
        json.dump(
            {
                "source": source_name,
                "summary": {
                    "total_patients": total_patients,
                    "detected_patients": detected_patients,
                    "detection_rate": (float(detected_patients) / float(total_patients)) if total_patients else 0.0,
                },
                "patients": rows,
            },
            f,
            indent=2,
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "case_folder",
            "category",
            "normal_no_bbox_cohort",
            "detected",
            "num_true_positive_slices",
            "num_slices_with_ground_truth_bbox",
            "num_no_bbox_slices",
            "highlighted_anomaly_pixels_no_bbox",
            "fp_pixel_ratio_percent",
            "highlighted_anomaly_pixels_binary_token_total",
            "num_slices",
            "precision",
            "false_positive_ratio",
        ])
        for row in rows:
            writer.writerow([
                row.get("patient_id", ""),
                row.get("case_folder", ""),
                row.get("category", ""),
                row.get("normal_no_bbox_cohort", "Other"),
                "YES" if bool(row.get("detected", False)) else "NO",
                int(row.get("num_true_positive_slices", 0) or 0),
                int(row.get("num_slices_with_ground_truth_bbox", 0) or 0),
                int(row.get("num_no_bbox_slices", 0) or 0),
                int(row.get("highlighted_anomaly_pixels_no_bbox", 0) or 0),
                float(row.get("fp_pixel_ratio_percent", 0.0) or 0.0),
                int(row.get("highlighted_anomaly_pixels_binary_token_total", 0) or 0),
                int(row.get("num_slices", 0) or 0),
                row.get("precision"),
                row.get("false_positive_ratio"),
            ])

    return json_path, csv_path


def write_normal_nobbox_fp_overview_outputs(
    overview_rows: List[Dict[str, object]],
    output_dir: Path,
    source_name: str,
) -> Tuple[Path, Path]:
    """Write normal no-bbox FP overview to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "NormalNoBBox_false_positive_pixels_overview.json"
    csv_path = output_dir / "NormalNoBBox_false_positive_pixels_overview.csv"

    with json_path.open("w") as f:
        json.dump({"source": source_name, "rows": overview_rows}, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cohort",
            "num_patients",
            "num_patients_with_no_bbox_slices",
            "total_highlighted_fp_pixels_no_bbox",
            "mean_highlighted_fp_pixels_per_patient",
        ])
        for row in overview_rows:
            writer.writerow([
                row.get("cohort", "Unknown"),
                int(row.get("num_patients", 0) or 0),
                int(row.get("num_patients_with_no_bbox_slices", 0) or 0),
                int(row.get("total_highlighted_fp_pixels_no_bbox", 0) or 0),
                float(row.get("mean_highlighted_fp_pixels_per_patient", 0.0) or 0.0),
            ])

    return json_path, csv_path


def build_combined_patient_detection_overview_rows(
    run_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build combined patient-detection overview rows with aggregate summaries."""
    if not run_rows:
        return []

    def _sum_rows(rows: List[Dict[str, object]], source_name: str) -> Dict[str, object]:
        total_patients = sum(int(r.get("total_patients", 0)) for r in rows)
        detected_patients = sum(int(r.get("detected_patients", 0)) for r in rows)
        total_tp_slices = sum(int(r.get("total_tp_slices", 0)) for r in rows)
        total_gt_bbox_slices = sum(int(r.get("total_gt_bbox_slices", 0)) for r in rows)
        total_no_bbox_slices = sum(int(r.get("total_no_bbox_slices", 0)) for r in rows)
        total_no_bbox_fp_px = sum(int(r.get("total_no_bbox_fp_px", 0)) for r in rows)
        total_slices = sum(int(r.get("total_slices", 0)) for r in rows)
        sum_ground_truth_bbox_pixels = sum(int(r.get("sum_ground_truth_bbox_pixels", 0)) for r in rows)
        sum_predicted_anomaly_pixels_inside_bbox = sum(int(r.get("sum_predicted_anomaly_pixels_inside_bbox", 0)) for r in rows)
        sum_predicted_anomaly_pixels_outside_bbox = sum(int(r.get("sum_predicted_anomaly_pixels_outside_bbox", 0)) for r in rows)
        tp = sum(int(r.get("tp", 0)) for r in rows)
        fp = sum(int(r.get("fp", 0)) for r in rows)
        fn = sum(int(r.get("fn", 0)) for r in rows)
        fp_ratio = None
        if sum_predicted_anomaly_pixels_inside_bbox > 0:
            fp_ratio = float(sum_predicted_anomaly_pixels_outside_bbox) / float(sum_predicted_anomaly_pixels_inside_bbox)

        weighted_precision_sum = 0.0
        weighted_f1_sum = 0.0
        weighted_tp_event_sum = 0.0
        for r in rows:
            n = int(r.get("total_patients", 0) or 0)
            if n <= 0:
                continue
            weighted_precision_sum += float(r.get("pixel_precision", 0.0) or 0.0) * float(n)
            weighted_f1_sum += float(r.get("f1_score", 0.0) or 0.0) * float(n)
            weighted_tp_event_sum += float(r.get("pixel_tp_ratio", 0.0) or 0.0) * float(n)

        paper_precision = (weighted_precision_sum / float(total_patients)) if total_patients > 0 else 0.0
        paper_f1 = (weighted_f1_sum / float(total_patients)) if total_patients > 0 else 0.0
        tp_event_rate = (weighted_tp_event_sum / float(total_patients)) if total_patients > 0 else 0.0

        return {
            "source": source_name,
            "total_patients": int(total_patients),
            "detected_patients": int(detected_patients),
            "detection_rate": (float(detected_patients) / float(total_patients)) if total_patients else 0.0,
            "sum_ground_truth_bbox_pixels": int(sum_ground_truth_bbox_pixels),
            "sum_predicted_anomaly_pixels_inside_bbox": int(sum_predicted_anomaly_pixels_inside_bbox),
            "sum_predicted_anomaly_pixels_outside_bbox": int(sum_predicted_anomaly_pixels_outside_bbox),
            "pixel_precision": float(paper_precision),
            "pixel_tp_ratio": float(tp_event_rate),
            "false_positive_ratio": fp_ratio,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "f1_score": float(paper_f1),
            "total_tp_slices": int(total_tp_slices),
            "total_gt_bbox_slices": int(total_gt_bbox_slices),
            "total_no_bbox_slices": int(total_no_bbox_slices),
            "total_no_bbox_fp_px": int(total_no_bbox_fp_px),
            "total_slices": int(total_slices),
        }

    def _is_test_samples_source(row: Dict[str, object]) -> bool:
        return "test_samples_fastmri" in str(row.get("source", "")).lower()

    def _is_validation_samples_source(row: Dict[str, object]) -> bool:
        return "validation_samples" in str(row.get("source", "")).lower()

    def _is_normal_only_source(row: Dict[str, object]) -> bool:
        return _is_test_samples_source(row) or _is_validation_samples_source(row)

    test_rows = [r for r in run_rows if _is_test_samples_source(r)]
    validation_rows = [r for r in run_rows if _is_validation_samples_source(r)]
    rest_rows = [r for r in run_rows if not _is_normal_only_source(r)]

    sorted_model_rows = sorted(
        run_rows,
        key=lambda r: (
            -float(r.get("detection_rate", 0.0)),
            -int(r.get("detected_patients", 0)),
            -int(r.get("total_patients", 0)),
            str(r.get("source", "")),
        ),
    )

    rows_out: List[Dict[str, object]] = [
        _sum_rows(run_rows, "OVERALL"),
        _sum_rows(rest_rows, "OVERALL_REST"),
    ]
    if test_rows:
        rows_out.append(_sum_rows(test_rows, "OVERALL_TEST_SAMPLES_FASTMRI"))
    if validation_rows:
        rows_out.append(_sum_rows(validation_rows, "OVERALL_VALIDATION_SAMPLES"))
    return rows_out + sorted_model_rows


def plot_combined_patient_detection_overview_table(
    run_rows: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """Render one combined table for patient-detection overview across processed JSONs."""
    rows = build_combined_patient_detection_overview_rows(run_rows)
    if not rows:
        raise ValueError("No rows available for combined patient detection overview table")

    header = [
        "Source",
        "Patients",
        "Detected",
        "Detection",
        "Pixel P",
        "Pixel TP",
        "Pixel F1",
        "TP slices",
        "GT-bbox slices",
        "No-bbox slices",
        "No-bbox FP px",
        "Total slices",
    ]

    body: List[List[str]] = []
    for row in rows:
        total_patients = int(row.get("total_patients", 0))
        detected = int(row.get("detected_patients", 0))
        body.append(
            [
                str(row.get("source", "Unknown")),
                str(total_patients),
                str(detected),
                f"{100.0 * float(row.get('detection_rate', 0.0)):.1f}%",
                f"{100.0 * float(row.get('pixel_precision', 0.0)):.1f}%",
                f"{100.0 * float(row.get('pixel_tp_ratio', 0.0)):.1f}%",
                f"{100.0 * float(row.get('f1_score', 0.0)):.1f}%",
                str(int(row.get("total_tp_slices", 0))),
                str(int(row.get("total_gt_bbox_slices", 0))),
                str(int(row.get("total_no_bbox_slices", 0))),
                str(int(row.get("total_no_bbox_fp_px", 0))),
                str(int(row.get("total_slices", 0))),
            ]
        )

    fig_h = max(4.0, min(0.42 * (len(body) + 1), 24.0))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=body,
        colLabels=header,
        colLoc="center",
        cellLoc="center",
        loc="center",
        colWidths=[0.24, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.08, 0.07],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1F1F1F")
            cell.set_text_props(color="white", weight="bold")
            continue

        if row_idx == 1:
            cell.set_facecolor("#E9F2FF")
            if col_idx == 0:
                cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8F8F8" if row_idx % 2 == 0 else "#FFFFFF")

        if col_idx in (3, 4, 5, 6):
            try:
                pct = float(cell.get_text().get_text().strip().replace("%", ""))
            except ValueError:
                pct = 0.0
            if pct >= 50.0:
                cell.set_facecolor("#EDF7ED")
                cell.set_text_props(color="#1B5E20", weight="bold")

        if col_idx == 10:
            try:
                fp_px = int(float(cell.get_text().get_text().strip()))
            except ValueError:
                fp_px = 0
            if fp_px > 0:
                cell.set_facecolor("#FFE6E6")
                cell.set_text_props(color="#8B0000", weight="bold")

    overall = rows[0]
    overall_avg_fp_pixel_ratio_percent = compute_fp_pixel_ratio_percent(
        highlighted_fp_pixels=int(overall.get("total_no_bbox_fp_px", 0) or 0),
        no_bbox_slices=int(overall.get("total_no_bbox_slices", 0) or 0),
    )

    validation_summary_row = next(
        (r for r in rows if str(r.get("source", "")).strip().upper() == "OVERALL_VALIDATION_SAMPLES"),
        None,
    )
    if validation_summary_row is not None:
        validation_avg_fp_pixel_ratio_percent = compute_fp_pixel_ratio_percent(
            highlighted_fp_pixels=int(validation_summary_row.get("total_no_bbox_fp_px", 0) or 0),
            no_bbox_slices=int(validation_summary_row.get("total_no_bbox_slices", 0) or 0),
        )
    else:
        validation_rows = [
            r
            for r in rows
            if "validation_samples" in str(r.get("source", "")).strip().lower()
        ]
        validation_avg_fp_pixel_ratio_percent = compute_fp_pixel_ratio_percent(
            highlighted_fp_pixels=sum(int(r.get("total_no_bbox_fp_px", 0) or 0) for r in validation_rows),
            no_bbox_slices=sum(int(r.get("total_no_bbox_slices", 0) or 0) for r in validation_rows),
        )

    test_summary_row = next(
        (r for r in rows if str(r.get("source", "")).strip().upper() == "OVERALL_TEST_SAMPLES_FASTMRI"),
        None,
    )
    if test_summary_row is not None:
        test_avg_fp_pixel_ratio_percent = compute_fp_pixel_ratio_percent(
            highlighted_fp_pixels=int(test_summary_row.get("total_no_bbox_fp_px", 0) or 0),
            no_bbox_slices=int(test_summary_row.get("total_no_bbox_slices", 0) or 0),
        )
    else:
        test_rows = [
            r
            for r in rows
            if "test_samples_fastmri" in str(r.get("source", "")).strip().lower()
        ]
        test_avg_fp_pixel_ratio_percent = compute_fp_pixel_ratio_percent(
            highlighted_fp_pixels=sum(int(r.get("total_no_bbox_fp_px", 0) or 0) for r in test_rows),
            no_bbox_slices=sum(int(r.get("total_no_bbox_slices", 0) or 0) for r in test_rows),
        )

    ax.set_title(
        (
            "Combined patient detection overview (pixel-based P/TP/F1)\n"
            f"Detected: {int(overall.get('detected_patients', 0))}/{int(overall.get('total_patients', 0))} | "
            f"Avg FP pixel ratio: {overall_avg_fp_pixel_ratio_percent:.2f}% | "
            f"Validation avg FP pixel ratio: {validation_avg_fp_pixel_ratio_percent:.2f}% | "
            f"Test avg FP pixel ratio: {test_avg_fp_pixel_ratio_percent:.2f}%"
        ),
        fontsize=12,
        pad=14,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_combined_patient_detection_overview_outputs(
    run_rows: List[Dict[str, object]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write one combined patient-detection overview JSON and CSV across all processed JSONs."""
    rows = build_combined_patient_detection_overview_rows(run_rows)
    if not rows:
        raise ValueError("No rows available for combined patient detection overview outputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "PatientDetection_combined_overview.json"
    csv_path = output_dir / "PatientDetection_combined_overview.csv"

    with json_path.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source",
            "total_patients",
            "detected_patients",
            "detection_rate_percent",
            "f1_score_percent",
            "pixel_precision_percent",
            "pixel_tp_ratio_percent",
            "sum_ground_truth_bbox_pixels",
            "sum_predicted_anomaly_pixels_inside_bbox",
            "sum_predicted_anomaly_pixels_outside_bbox",
            "tp",
            "fp",
            "fn",
            "total_tp_slices",
            "total_gt_bbox_slices",
            "total_no_bbox_slices",
            "total_no_bbox_fp_px",
            "total_slices",
        ])
        for row in rows:
            writer.writerow([
                row.get("source", "Unknown"),
                int(row.get("total_patients", 0)),
                int(row.get("detected_patients", 0)),
                100.0 * float(row.get("detection_rate", 0.0)),
                100.0 * float(row.get("f1_score", 0.0)),
                100.0 * float(row.get("pixel_precision", 0.0)),
                100.0 * float(row.get("pixel_tp_ratio", 0.0)),
                int(row.get("sum_ground_truth_bbox_pixels", 0)),
                int(row.get("sum_predicted_anomaly_pixels_inside_bbox", 0)),
                int(row.get("sum_predicted_anomaly_pixels_outside_bbox", 0)),
                int(row.get("tp", 0)),
                int(row.get("fp", 0)),
                int(row.get("fn", 0)),
                int(row.get("total_tp_slices", 0)),
                int(row.get("total_gt_bbox_slices", 0)),
                int(row.get("total_no_bbox_slices", 0)),
                int(row.get("total_no_bbox_fp_px", 0)),
                int(row.get("total_slices", 0)),
            ])

    return json_path, csv_path


def build_combined_binary_token_overview_rows(
    run_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build overview rows with an additional overall aggregate row."""
    if not run_rows:
        return []

    def _sum_rows(rows: List[Dict[str, object]], source_name: str, threshold: float) -> Dict[str, object]:
        total_patients = sum(int(r.get("total_patients", 0)) for r in rows)
        anomaly_patients = sum(int(r.get("anomaly_patients", 0)) for r in rows)
        normal_patients = sum(int(r.get("normal_patients", 0)) for r in rows)
        tp = sum(int(r.get("tp", 0)) for r in rows)
        fn = sum(int(r.get("fn", 0)) for r in rows)
        fp = sum(int(r.get("fp", 0)) for r in rows)
        tn = sum(int(r.get("tn", 0)) for r in rows)

        sensitivity = (tp / anomaly_patients) if anomaly_patients else 0.0
        specificity = (tn / normal_patients) if normal_patients else 0.0

        return {
            "source": source_name,
            "threshold": threshold,
            "total_patients": total_patients,
            "anomaly_patients": anomaly_patients,
            "normal_patients": normal_patients,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def _is_test_samples_source(row: Dict[str, object]) -> bool:
        return "test_samples_fastmri" in str(row.get("source", "")).lower()

    def _is_validation_samples_source(row: Dict[str, object]) -> bool:
        source = str(row.get("source", "")).lower()
        return "validation_samples" in source

    def _is_normal_only_source(row: Dict[str, object]) -> bool:
        return _is_test_samples_source(row) or _is_validation_samples_source(row)

    threshold = float(run_rows[0].get("threshold", 0.0))
    test_rows = [r for r in run_rows if _is_test_samples_source(r)]
    validation_rows = [r for r in run_rows if _is_validation_samples_source(r)]
    rest_rows = [r for r in run_rows if not _is_normal_only_source(r)]

    sorted_model_rows = sorted(
        run_rows,
        key=lambda r: (
            -float(r.get("sensitivity", 0.0)),
            -int(r.get("total_patients", 0)),
            str(r.get("source", "")),
        ),
    )

    rows_out: List[Dict[str, object]] = [
        _sum_rows(run_rows, "OVERALL", threshold),
        _sum_rows(rest_rows, "OVERALL_REST", threshold),
    ]
    if test_rows:
        rows_out.append(_sum_rows(test_rows, "OVERALL_TEST_SAMPLES_FASTMRI", threshold))
    if validation_rows:
        rows_out.append(_sum_rows(validation_rows, "OVERALL_VALIDATION_SAMPLES", threshold))
    return rows_out + sorted_model_rows


def plot_combined_binary_token_overview_table(
    run_rows: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """Render one combined table with metrics for all processed JSONs."""
    rows = build_combined_binary_token_overview_rows(run_rows)
    if not rows:
        raise ValueError("No rows available for combined overview table")

    header = [
        "Source",
        "Total",
        "Anom GT",
        "Norm GT",
        "TP",
        "FN",
        "FP",
        "TN",
        "Sensitivity",
        "Specificity",
    ]

    body: List[List[str]] = []
    for row in rows:
        body.append(
            [
                str(row.get("source", "Unknown")),
                str(int(row.get("total_patients", 0))),
                str(int(row.get("anomaly_patients", 0))),
                str(int(row.get("normal_patients", 0))),
                str(int(row.get("tp", 0))),
                str(int(row.get("fn", 0))),
                str(int(row.get("fp", 0))),
                str(int(row.get("tn", 0))),
                f"{100.0 * float(row.get('sensitivity', 0.0)):.1f}%",
                f"{100.0 * float(row.get('specificity', 0.0)):.1f}%",
            ]
        )

    fig_h = max(4.0, min(0.42 * (len(body) + 1), 24.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=body,
        colLabels=header,
        colLoc="center",
        cellLoc="center",
        loc="center",
        colWidths=[0.27, 0.08, 0.09, 0.09, 0.06, 0.06, 0.06, 0.06, 0.11, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1F1F1F")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx == 1:
            cell.set_facecolor("#E9F2FF")
            if col_idx == 0:
                cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8F8F8" if row_idx % 2 == 0 else "#FFFFFF")

    threshold = float(rows[0].get("threshold", 0.0))
    ax.set_title(
        f"Combined Binary+Token patient detection overview (threshold = {threshold})",
        fontsize=12,
        pad=14,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_combined_binary_token_overview_outputs(
    run_rows: List[Dict[str, object]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write one combined overview JSON and CSV across all processed JSONs."""
    rows = build_combined_binary_token_overview_rows(run_rows)
    if not rows:
        raise ValueError("No rows available for combined overview outputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "BinaryToken_combined_overview.json"
    csv_path = output_dir / "BinaryToken_combined_overview.csv"

    with json_path.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source",
            "threshold",
            "total_patients",
            "anomaly_patients",
            "normal_patients",
            "tp",
            "fn",
            "fp",
            "tn",
            "sensitivity_percent",
            "specificity_percent",
        ])
        for row in rows:
            writer.writerow([
                row.get("source", "Unknown"),
                float(row.get("threshold", 0.0)),
                int(row.get("total_patients", 0)),
                int(row.get("anomaly_patients", 0)),
                int(row.get("normal_patients", 0)),
                int(row.get("tp", 0)),
                int(row.get("fn", 0)),
                int(row.get("fp", 0)),
                int(row.get("tn", 0)),
                100.0 * float(row.get("sensitivity", 0.0)),
                100.0 * float(row.get("specificity", 0.0)),
            ])

    return json_path, csv_path


def write_json_file(payload: Dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return output_path


def format_threshold_value(threshold: float) -> str:
    if isinf(threshold):
        return "inf" if threshold > 0 else "-inf"
    return f"{threshold:.6g}"


def parse_threshold_json(threshold: str | float | int) -> float:
    if isinstance(threshold, str):
        lower = threshold.lower()
        if lower == "inf":
            return inf
        if lower == "-inf":
            return -inf
        return float(threshold)
    return float(threshold)


def is_test_samples_normal(tokens: Set[str], patient_id: str) -> bool:
    if "test_samples_fastmri" in patient_id.lower():
        return True
    return any("test_samples_fastmri" in token for token in tokens)


def is_validation_samples(tokens: Set[str], patient_id: str) -> bool:
    if "validation_samples" in patient_id.lower():
        return True
    return any("validation_samples" in token for token in tokens)


def aggregate_fastmri_binary_token_patient_scores(
    results: List[dict],
    source_json: Path,
    allowed_case_folders: Optional[Set[str]] = None,
    allowed_categories: Optional[Set[str]] = None,
    include_validation_normals: bool = False,
) -> List[Dict[str, object]]:
    """Aggregate per-patient binary-token score for FastMRI ROC analysis."""
    patient_map: Dict[str, Dict[str, object]] = {}

    for item in results:
        case_folder = item.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        if allowed_categories and not matches_category(item.get("category"), allowed_categories):
            continue

        binary_val = item.get("Binary_Sum_Heatmap")
        if binary_val is None:
            continue

        pid_raw = patient_id_from_item(item)
        if pid_raw is None:
            pid_raw = item.get("patient_id") or case_folder or "unknown"
        patient_id = strip_unknown_prefix(str(pid_raw)) or "unknown"
        key = f"{source_json.parent.name}::{patient_id}"

        if key not in patient_map:
            patient_map[key] = {
                "global_patient_key": key,
                "patient_id": patient_id,
                "source_json": str(source_json),
                "source_id": source_json.parent.name,
                "case_folders": set(),
                "categories": set(),
                "category_votes": {},
                "binary_token_score": 0.0,
                "num_slices": 0,
            }

        entry = patient_map[key]
        entry["binary_token_score"] += float(binary_val)
        entry["num_slices"] += 1

        cf = str(case_folder) if case_folder is not None else ""
        if cf:
            entry["case_folders"].add(cf)

        category = str(item.get("category") or "")
        if category:
            entry["categories"].add(category)
            votes = entry["category_votes"]
            votes[category] = int(votes.get(category, 0)) + 1

    rows: List[Dict[str, object]] = []
    for entry in patient_map.values():
        categories = sorted(entry["categories"])
        case_folders = sorted(entry["case_folders"])
        tokens = {str(v).lower() for v in categories + case_folders if str(v).strip()}

        patient_id = str(entry["patient_id"])
        is_test_normal = is_test_samples_normal(tokens, patient_id)
        is_validation = is_validation_samples(tokens, patient_id)

        if is_test_normal:
            roc_class = "normal_test"
            label: Optional[int] = 0
            include_in_roc = True
        elif is_validation and not include_validation_normals:
            roc_class = "normal_validation_excluded"
            label = None
            include_in_roc = False
        else:
            roc_class = "anomaly"
            label = 1
            include_in_roc = True

        dominant_category = None
        category_votes = entry["category_votes"]
        if category_votes:
            dominant_category = max(category_votes.items(), key=lambda kv: kv[1])[0]
        elif case_folders:
            dominant_category = case_folders[0]
        else:
            dominant_category = entry["source_id"]

        rows.append({
            "global_patient_key": entry["global_patient_key"],
            "patient_id": patient_id,
            "source_json": entry["source_json"],
            "source_id": entry["source_id"],
            "categories": categories,
            "case_folders": case_folders,
            "dominant_category": str(dominant_category),
            "binary_token_score": float(entry["binary_token_score"]),
            "num_slices": int(entry["num_slices"]),
            "roc_class": roc_class,
            "include_in_roc": include_in_roc,
            "is_test_normal": is_test_normal,
            "is_validation": is_validation,
            "label": label,
        })

    return sorted(rows, key=lambda row: row["binary_token_score"], reverse=True)


def merge_fastmri_json_payloads_for_roc(
    input_paths: List[Path],
    allowed_case_folders: Optional[Set[str]] = None,
    allowed_categories: Optional[Set[str]] = None,
    include_validation_normals: bool = False,
) -> Dict[str, object]:
    """Merge FastMRI results and build patient-level binary-token scores."""
    unique_paths: List[Path] = []
    seen: Set[Path] = set()
    for p in input_paths:
        resolved = p.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    merged_results: List[dict] = []
    merged_patient_summary: List[dict] = []
    merged_patient_scores: List[Dict[str, object]] = []
    source_summaries: List[Dict[str, object]] = []

    for json_path in unique_paths:
        if not json_path.exists():
            logging.warning("Skipping missing JSON in FastMRI ROC merge: %s", json_path)
            continue

        payload = load_payload(json_path)
        results = payload.get("results", [])
        patient_summary = payload.get("patient_summary", [])
        if not isinstance(results, list):
            results = []
        if not isinstance(patient_summary, list):
            patient_summary = []

        for item in results:
            row = dict(item)
            row["source_json"] = str(json_path)
            row["source_id"] = json_path.parent.name
            merged_results.append(row)

        for item in patient_summary:
            row = dict(item)
            row["source_json"] = str(json_path)
            row["source_id"] = json_path.parent.name
            merged_patient_summary.append(row)

        file_scores = aggregate_fastmri_binary_token_patient_scores(
            results,
            source_json=json_path,
            allowed_case_folders=allowed_case_folders,
            allowed_categories=allowed_categories,
            include_validation_normals=include_validation_normals,
        )
        merged_patient_scores.extend(file_scores)

        source_summaries.append({
            "source_json": str(json_path),
            "source_id": json_path.parent.name,
            "num_results": len(results),
            "num_patient_summary": len(patient_summary),
            "num_patient_scores": len(file_scores),
        })

    summary_counts = {
        "num_input_files": len(unique_paths),
        "num_merged_results": len(merged_results),
        "num_merged_patient_summary": len(merged_patient_summary),
        "num_merged_patient_scores": len(merged_patient_scores),
        "num_roc_included": sum(1 for row in merged_patient_scores if bool(row.get("include_in_roc"))),
        "num_test_normals": sum(1 for row in merged_patient_scores if bool(row.get("is_test_normal"))),
        "num_validation_excluded": sum(1 for row in merged_patient_scores if bool(row.get("is_validation")) and not bool(row.get("include_in_roc"))),
        "num_anomalies": sum(1 for row in merged_patient_scores if row.get("label") == 1),
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "input_files": [str(p) for p in unique_paths],
        "source_summaries": source_summaries,
        "summary": summary_counts,
        "merged_patient_scores_binary_token": merged_patient_scores,
        "merged_patient_summary": merged_patient_summary,
        "merged_results": merged_results,
    }


def compute_auc_trapezoid(fpr: List[float], tpr: List[float]) -> float:
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr length mismatch")
    if len(fpr) < 2:
        return 0.0
    pairs = sorted(zip(fpr, tpr), key=lambda item: (item[0], item[1]))
    area = 0.0
    for idx in range(1, len(pairs)):
        x1, y1 = pairs[idx - 1]
        x2, y2 = pairs[idx]
        if x2 < x1:
            continue
        area += (x2 - x1) * (y1 + y2) * 0.5
    return area


def percentile_from_sorted_values(sorted_values: List[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile from empty data")
    if quantile <= 0.0:
        return float(sorted_values[0])
    if quantile >= 1.0:
        return float(sorted_values[-1])
    rank = (len(sorted_values) - 1) * quantile
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = rank - lower_idx
    lower = float(sorted_values[lower_idx])
    upper = float(sorted_values[upper_idx])
    return lower + (upper - lower) * weight


def compute_roc_pairs(y_true: List[int], scores: List[float]) -> List[Tuple[float, float]]:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC pair computation requires both classes")

    unique_scores = sorted(set(scores), reverse=True)
    thresholds = [inf] + unique_scores + [-inf]
    pairs: List[Tuple[float, float]] = []
    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        pairs.append((fpr, tpr))
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def build_step_curve_from_pairs(pairs: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    if not pairs:
        return [0.0, 1.0], [0.0, 1.0]

    best_tpr_by_fpr: Dict[float, float] = {}
    for fpr, tpr in pairs:
        fpr_clamped = min(max(float(fpr), 0.0), 1.0)
        tpr_clamped = min(max(float(tpr), 0.0), 1.0)
        prev = best_tpr_by_fpr.get(fpr_clamped)
        if prev is None or tpr_clamped > prev:
            best_tpr_by_fpr[fpr_clamped] = tpr_clamped

    fpr_values = sorted(best_tpr_by_fpr.keys())
    tpr_values: List[float] = []
    running_max_tpr = 0.0
    for fpr in fpr_values:
        running_max_tpr = max(running_max_tpr, best_tpr_by_fpr[fpr])
        tpr_values.append(running_max_tpr)

    if fpr_values[0] > 0.0:
        fpr_values.insert(0, 0.0)
        tpr_values.insert(0, 0.0)

    if fpr_values[-1] < 1.0:
        fpr_values.append(1.0)
        tpr_values.append(max(tpr_values[-1], 1.0))
    else:
        tpr_values[-1] = max(tpr_values[-1], 1.0)

    return fpr_values, tpr_values


def sample_step_tpr_on_grid(
    fpr_values: List[float],
    tpr_values: List[float],
    fpr_grid: List[float],
) -> List[float]:
    sampled: List[float] = []
    for fpr in fpr_grid:
        idx = bisect_right(fpr_values, fpr) - 1
        if idx < 0:
            sampled.append(0.0)
        else:
            sampled.append(float(tpr_values[idx]))
    return sampled


def compute_bootstrap_roc_ci(
    y_true: List[int],
    scores: List[float],
    n_bootstrap_samples: int,
    confidence_level: float,
    random_seed: Optional[int],
    fpr_grid_size: int,
) -> Dict[str, object]:
    if n_bootstrap_samples <= 1:
        raise ValueError("n_bootstrap_samples must be > 1")
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if fpr_grid_size < 2:
        raise ValueError("fpr_grid_size must be >= 2")

    pos_indices = [idx for idx, label in enumerate(y_true) if int(label) == 1]
    neg_indices = [idx for idx, label in enumerate(y_true) if int(label) == 0]
    if not pos_indices or not neg_indices:
        raise ValueError("Bootstrap ROC CI requires both classes")

    rng = random.Random(random_seed)
    fpr_grid = [idx / (fpr_grid_size - 1) for idx in range(fpr_grid_size)]

    auc_samples: List[float] = []
    tpr_samples_per_grid: List[List[float]] = [[] for _ in fpr_grid]

    for _ in range(n_bootstrap_samples):
        sampled_indices = [
            pos_indices[rng.randrange(len(pos_indices))] for _ in range(len(pos_indices))
        ]
        sampled_indices.extend(
            neg_indices[rng.randrange(len(neg_indices))] for _ in range(len(neg_indices))
        )

        sampled_y_true = [int(y_true[idx]) for idx in sampled_indices]
        sampled_scores = [float(scores[idx]) for idx in sampled_indices]

        roc_pairs = compute_roc_pairs(sampled_y_true, sampled_scores)
        auc_samples.append(
            compute_auc_trapezoid(
                [pair[0] for pair in roc_pairs],
                [pair[1] for pair in roc_pairs],
            )
        )

        step_fpr, step_tpr = build_step_curve_from_pairs(roc_pairs)
        tpr_on_grid = sample_step_tpr_on_grid(step_fpr, step_tpr, fpr_grid)
        for grid_idx, tpr_value in enumerate(tpr_on_grid):
            tpr_samples_per_grid[grid_idx].append(float(tpr_value))

    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q

    auc_samples_sorted = sorted(auc_samples)
    auc_mean = sum(auc_samples) / len(auc_samples)
    auc_std = (
        (sum((sample - auc_mean) ** 2 for sample in auc_samples) / (len(auc_samples) - 1)) ** 0.5
        if len(auc_samples) > 1
        else 0.0
    )
    auc_ci_lower = percentile_from_sorted_values(auc_samples_sorted, lower_q)
    auc_ci_upper = percentile_from_sorted_values(auc_samples_sorted, upper_q)

    tpr_ci_lower: List[float] = []
    tpr_ci_upper: List[float] = []
    tpr_median: List[float] = []
    for values in tpr_samples_per_grid:
        sorted_values = sorted(values)
        tpr_ci_lower.append(percentile_from_sorted_values(sorted_values, lower_q))
        tpr_ci_upper.append(percentile_from_sorted_values(sorted_values, upper_q))
        tpr_median.append(percentile_from_sorted_values(sorted_values, 0.5))

    return {
        "method": "stratified_bootstrap",
        "n_bootstrap_samples": int(n_bootstrap_samples),
        "confidence_level": float(confidence_level),
        "random_seed": random_seed,
        "fpr_grid": fpr_grid,
        "tpr_ci_lower": tpr_ci_lower,
        "tpr_ci_upper": tpr_ci_upper,
        "tpr_median": tpr_median,
        "auc_mean": float(auc_mean),
        "auc_std": float(auc_std),
        "auc_ci_lower": float(auc_ci_lower),
        "auc_ci_upper": float(auc_ci_upper),
    }


def compute_fastmri_roc_and_auc(
    patient_scores: List[Dict[str, object]],
    expected_test_normals: Optional[int] = None,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_random_seed: Optional[int] = 42,
    ci_fpr_grid_size: int = 201,
) -> Dict[str, object]:
    """Compute FastMRI patient-level ROC using binary_token_score and Test normals only."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be >= 0")
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if ci_fpr_grid_size < 2:
        raise ValueError("ci_fpr_grid_size must be >= 2")

    roc_rows = [
        row for row in patient_scores
        if bool(row.get("include_in_roc")) and row.get("label") in (0, 1)
    ]
    if not roc_rows:
        raise ValueError("No ROC rows available after applying Test-only normal policy")

    y_true = [int(row["label"]) for row in roc_rows]
    scores = [float(row.get("binary_token_score", 0.0)) for row in roc_rows]

    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "ROC requires both anomalies and Test normals. Found anomalies=%d, test_normals=%d"
            % (positives, negatives)
        )

    unique_scores = sorted(set(scores), reverse=True)
    thresholds = [inf] + unique_scores + [-inf]

    roc_points: List[Dict[str, object]] = []
    best_idx = 0
    best_youden = float("-inf")
    for idx, threshold in enumerate(thresholds):
        y_pred = [1 if score > threshold else 0 for score in scores]

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        sensitivity = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        specificity = tn / negatives if negatives else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        youden_j = sensitivity - fpr

        roc_points.append({
            "threshold": "inf" if isinf(threshold) and threshold > 0 else ("-inf" if isinf(threshold) else float(threshold)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": sensitivity,
            "tpr": sensitivity,
            "fpr": fpr,
            "specificity": specificity,
            "precision": precision,
            "youden_j": youden_j,
        })

        if youden_j > best_youden:
            best_youden = youden_j
            best_idx = idx

    auc_value = compute_auc_trapezoid(
        [float(point["fpr"]) for point in roc_points],
        [float(point["tpr"]) for point in roc_points],
    )

    bootstrap_ci: Optional[Dict[str, object]] = None
    auc_ci_lower: Optional[float] = None
    auc_ci_upper: Optional[float] = None
    auc_std: Optional[float] = None
    if bootstrap_samples > 1:
        bootstrap_ci = compute_bootstrap_roc_ci(
            y_true=y_true,
            scores=scores,
            n_bootstrap_samples=int(bootstrap_samples),
            confidence_level=float(confidence_level),
            random_seed=bootstrap_random_seed,
            fpr_grid_size=int(ci_fpr_grid_size),
        )
        auc_ci_lower = float(bootstrap_ci.get("auc_ci_lower", 0.0))
        auc_ci_upper = float(bootstrap_ci.get("auc_ci_upper", 0.0))
        auc_std = float(bootstrap_ci.get("auc_std", 0.0))

    summary = {
        "num_patients_in_roc": len(roc_rows),
        "num_anomalies": positives,
        "num_test_normals": negatives,
        "num_validation_excluded": sum(1 for row in patient_scores if bool(row.get("is_validation")) and not bool(row.get("include_in_roc"))),
        "expected_test_normals": expected_test_normals,
        "expected_test_normal_match": (negatives == expected_test_normals) if expected_test_normals is not None else None,
        "auc": auc_value,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "auc_std": auc_std,
        "roc_confidence_level": confidence_level if bootstrap_ci else None,
        "roc_bootstrap_samples": int(bootstrap_samples) if bootstrap_ci else 0,
    }

    return {
        "summary": summary,
        "best_threshold_by_youden_j": dict(roc_points[best_idx]),
        "roc_points": roc_points,
        "roc_bootstrap_ci": bootstrap_ci,
    }


def evaluate_threshold_on_patient_scores(
    patient_scores: List[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    """Evaluate one fixed threshold on the ROC-eligible FastMRI patient set."""
    rows = [
        row for row in patient_scores
        if bool(row.get("include_in_roc")) and row.get("label") in (0, 1)
    ]
    y_true = [int(row["label"]) for row in rows]
    scores = [float(row.get("binary_token_score", 0.0)) for row in rows]

    positives = sum(y_true)
    negatives = len(y_true) - positives
    y_pred = [1 if score > threshold else 0 for score in scores]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    sensitivity = tp / positives if positives else 0.0
    fpr = fp / negatives if negatives else 0.0
    specificity = tn / negatives if negatives else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "tpr": sensitivity,
        "fpr": fpr,
        "specificity": specificity,
        "precision": precision,
        "youden_j": sensitivity - fpr,
    }


def select_threshold_for_target_fpr(roc_metrics: Dict[str, object], target_fpr: float) -> Dict[str, object]:
    """Pick threshold with highest sensitivity while keeping FPR <= target_fpr."""
    if target_fpr < 0.0 or target_fpr > 1.0:
        raise ValueError("target_fpr must be within [0, 1]")

    roc_points = roc_metrics.get("roc_points", [])
    if not isinstance(roc_points, list) or not roc_points:
        raise ValueError("ROC metrics missing roc_points")

    finite_points: List[Dict[str, object]] = []
    for point in roc_points:
        threshold = parse_threshold_json(point.get("threshold", "inf"))
        if isinf(threshold):
            continue
        finite_points.append(point)
    if not finite_points:
        raise ValueError("No finite threshold points in ROC")

    constrained = [point for point in finite_points if float(point.get("fpr", 1.0)) <= target_fpr]
    candidates = constrained if constrained else finite_points
    best = max(
        candidates,
        key=lambda p: (
            float(p.get("sensitivity", 0.0)),
            -float(p.get("fpr", 1.0)),
            float(p.get("precision", 0.0)),
            float(p.get("specificity", 0.0)),
        ),
    )
    return dict(best)


def plot_fastmri_roc_curve(
    roc_metrics: Dict[str, object],
    output_path: Path,
    show_best_threshold_marker: bool = False,
) -> Path:
    roc_points = roc_metrics.get("roc_points", [])
    if not isinstance(roc_points, list) or not roc_points:
        raise ValueError("ROC points not available for plotting")

    pairs = sorted(
        ((float(point["fpr"]), float(point["tpr"])) for point in roc_points),
        key=lambda item: (item[0], item[1]),
    )
    step_fpr, step_tpr = build_step_curve_from_pairs(pairs)

    summary = roc_metrics.get("summary", {})
    auc_value = float(summary.get("auc", 0.0))
    auc_ci_lower = summary.get("auc_ci_lower")
    auc_ci_upper = summary.get("auc_ci_upper")
    confidence_level = float(summary.get("roc_confidence_level", 0.95) or 0.95)
    confidence_pct = int(round(100.0 * confidence_level))

    if auc_ci_lower is not None and auc_ci_upper is not None:
        roc_label = (
            f"ROC (AUC={auc_value:.2f}, {confidence_pct}% CI={float(auc_ci_lower):.2f}-{float(auc_ci_upper):.2f})"
        )
    else:
        roc_label = f"ROC (AUC={auc_value:.2f})"

    fig, ax = plt.subplots(figsize=(7, 6))

    bootstrap_ci = roc_metrics.get("roc_bootstrap_ci", {})
    if isinstance(bootstrap_ci, dict):
        ci_fpr = bootstrap_ci.get("fpr_grid")
        ci_low = bootstrap_ci.get("tpr_ci_lower")
        ci_high = bootstrap_ci.get("tpr_ci_upper")
        if (
            isinstance(ci_fpr, list)
            and isinstance(ci_low, list)
            and isinstance(ci_high, list)
            and len(ci_fpr) == len(ci_low) == len(ci_high)
            and len(ci_fpr) > 1
        ):
            ax.fill_between(
                ci_fpr,
                ci_low,
                ci_high,
                step="post",
                color="#e7a46e",
                alpha=0.25,
                linewidth=0.0,
                label=f"{confidence_pct}% ROC CI band",
            )

    ax.step(
        step_fpr,
        step_tpr,
        where="post",
        linewidth=2.2,
        color="#e7a46e",
        label=roc_label,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Chance")

    best = roc_metrics.get("best_threshold_by_youden_j", {})
    if show_best_threshold_marker and isinstance(best, dict) and best:
        best_fpr = float(best.get("fpr", 0.0))
        best_tpr = float(best.get("tpr", 0.0))
        best_threshold = parse_threshold_json(best.get("threshold", "inf"))
        ax.scatter(
            [best_fpr],
            [best_tpr],
            color="red",
            s=40,
            zorder=3,
            label=f"Best J threshold={format_threshold_value(best_threshold)}",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("FastMRI ROC: Binary+Token patient sum")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_roc_threshold_table_rows(
    roc_metrics: Dict[str, object],
    include_infinite: bool = False,
) -> List[Dict[str, object]]:
    roc_points = roc_metrics.get("roc_points", [])
    if not isinstance(roc_points, list) or not roc_points:
        raise ValueError("ROC metrics missing roc_points for threshold table")

    rows: List[Dict[str, object]] = []
    for point in roc_points:
        threshold = parse_threshold_json(point.get("threshold", "inf"))
        if not include_infinite and isinf(threshold):
            continue
        rows.append({
            "threshold": format_threshold_value(threshold),
            "threshold_value": threshold,
            "sensitivity": float(point.get("sensitivity", 0.0)),
            "fpr": float(point.get("fpr", 0.0)),
            "specificity": float(point.get("specificity", 0.0)),
            "precision": float(point.get("precision", 0.0)),
            "tp": int(point.get("tp", 0)),
            "fp": int(point.get("fp", 0)),
            "tn": int(point.get("tn", 0)),
            "fn": int(point.get("fn", 0)),
            "youden_j": float(point.get("youden_j", 0.0)),
        })

    if not rows:
        raise ValueError("No threshold rows available for ROC table")
    return rows


def write_roc_threshold_table_csv(rows: List[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "sensitivity",
        "fpr",
        "specificity",
        "precision",
        "tp",
        "fp",
        "tn",
        "fn",
        "youden_j",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    return output_path


def roc_sensitivity_color(value: float) -> str:
    if value >= 0.9:
        return "#d8f3dc"
    if value >= 0.75:
        return "#e9f5db"
    if value >= 0.6:
        return "#fff3bf"
    if value >= 0.4:
        return "#ffe8cc"
    return "#ffd6d6"


def roc_fpr_color(value: float) -> str:
    if value <= 0.02:
        return "#d8f3dc"
    if value <= 0.05:
        return "#e9f5db"
    if value <= 0.10:
        return "#fff3bf"
    if value <= 0.20:
        return "#ffe8cc"
    return "#ffd6d6"


def plot_roc_threshold_table_figures(
    rows: List[Dict[str, object]],
    output_prefix: Path,
    rows_per_page: int = 40,
) -> List[Path]:
    if not rows:
        raise ValueError("rows is empty for ROC threshold figure generation")
    if rows_per_page < 1:
        raise ValueError("rows_per_page must be >= 1")

    prefix = output_prefix.with_suffix("") if output_prefix.suffix else output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    pages = [rows[idx: idx + rows_per_page] for idx in range(0, len(rows), rows_per_page)]
    total_pages = len(pages)
    outputs: List[Path] = []

    headers = ["Threshold", "Sensitivity", "FPR", "Specificity", "Precision", "TP", "FP", "TN", "FN", "Youden J"]
    for page_idx, page_rows in enumerate(pages, start=1):
        fig_h = max(4.0, 1.8 + 0.36 * len(page_rows))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")

        body: List[List[str]] = []
        for row in page_rows:
            body.append([
                str(row["threshold"]),
                f"{100.0 * float(row['sensitivity']):.2f}%",
                f"{100.0 * float(row['fpr']):.2f}%",
                f"{100.0 * float(row['specificity']):.2f}%",
                f"{100.0 * float(row['precision']):.2f}%",
                str(int(row["tp"])),
                str(int(row["fp"])),
                str(int(row["tn"])),
                str(int(row["fn"])),
                f"{float(row['youden_j']):.4f}",
            ])

        ax.set_title(f"ROC Threshold Performance Table (Page {page_idx}/{total_pages})", fontsize=12, pad=10)
        table = ax.table(cellText=body, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.18)

        for (r_idx, c_idx), cell in table.get_celld().items():
            if r_idx == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#264653")
                continue
            row = page_rows[r_idx - 1]
            if c_idx == 1:
                cell.set_facecolor(roc_sensitivity_color(float(row["sensitivity"])))
            elif c_idx == 2:
                cell.set_facecolor(roc_fpr_color(float(row["fpr"])))
            else:
                cell.set_facecolor("#f8f9fa")

        fig.tight_layout()
        if total_pages == 1:
            out = prefix.with_suffix(".png")
        else:
            out = prefix.parent / f"{prefix.name}_page_{page_idx:02d}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outputs.append(out)

    return outputs


def pairwise_auc_vs_normals(anomaly_scores: List[float], normal_scores: List[float]) -> float:
    total = len(anomaly_scores) * len(normal_scores)
    if total == 0:
        return 0.0
    wins = 0.0
    for a_score in anomaly_scores:
        for n_score in normal_scores:
            if a_score > n_score:
                wins += 1.0
            elif a_score == n_score:
                wins += 0.5
    return wins / total


def compute_fastmri_category_stratified_performance(
    patient_scores: List[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    """Compute per-anomaly-category metrics at a fixed threshold using Test normals as negatives."""
    normal_rows = [row for row in patient_scores if bool(row.get("include_in_roc")) and int(row.get("label", -1)) == 0]
    anomaly_rows = [row for row in patient_scores if bool(row.get("include_in_roc")) and int(row.get("label", -1)) == 1]

    if not normal_rows:
        raise ValueError("No Test normal rows available for category stratification")
    if not anomaly_rows:
        raise ValueError("No anomaly rows available for category stratification")

    normal_scores = [float(row.get("binary_token_score", 0.0)) for row in normal_rows]
    fp = sum(1 for score in normal_scores if score > threshold)
    tn = len(normal_scores) - fp
    fpr = fp / len(normal_scores)
    specificity = tn / len(normal_scores)

    by_category: Dict[str, List[float]] = {}
    for row in anomaly_rows:
        category = str(row.get("dominant_category") or row.get("source_id") or "Unknown")
        by_category.setdefault(category, []).append(float(row.get("binary_token_score", 0.0)))

    rows: List[Dict[str, object]] = []
    for category, scores in by_category.items():
        total = len(scores)
        tp = sum(1 for score in scores if score > threshold)
        fn = total - tp
        sensitivity = tp / total if total else 0.0
        auc_vs_normals = pairwise_auc_vs_normals(scores, normal_scores)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rows.append({
            "category": category,
            "num_patients": total,
            "threshold": threshold,
            "tp_detected": tp,
            "fn_missed": fn,
            "sensitivity": sensitivity,
            "auc_vs_test_normals": auc_vs_normals,
            "mean_score": sum(scores) / total,
            "min_score": min(scores),
            "max_score": max(scores),
            "precision_vs_test_normals_at_threshold": precision,
            "test_normal_count": len(normal_scores),
            "test_normal_fp_at_threshold": fp,
            "fpr_at_threshold": fpr,
            "specificity_at_threshold": specificity,
        })

    rows.sort(key=lambda row: (-float(row["sensitivity"]), -float(row["auc_vs_test_normals"]), -int(row["num_patients"]), str(row["category"])))
    return {
        "threshold": threshold,
        "test_normal_count": len(normal_scores),
        "test_normal_fp_at_threshold": fp,
        "fpr_at_threshold": fpr,
        "specificity_at_threshold": specificity,
        "num_categories": len(rows),
        "rows": rows,
    }


def write_fastmri_category_stratified_csv(rows: List[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "num_patients",
        "threshold",
        "tp_detected",
        "fn_missed",
        "sensitivity",
        "auc_vs_test_normals",
        "mean_score",
        "min_score",
        "max_score",
        "precision_vs_test_normals_at_threshold",
        "test_normal_count",
        "test_normal_fp_at_threshold",
        "fpr_at_threshold",
        "specificity_at_threshold",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_fastmri_category_stratified_table_figure(payload: Dict[str, object], output_path: Path) -> Path:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Category stratified rows missing for FastMRI table figure")

    threshold = float(payload.get("threshold", 0.0))
    fpr = float(payload.get("fpr_at_threshold", 0.0))
    headers = ["Category", "N", "TP/FN", "Sensitivity", "AUC vs Test", "Mean score", "FPR", "Threshold"]

    body: List[List[str]] = []
    for row in rows:
        body.append([
            str(row["category"]),
            str(int(row["num_patients"])),
            f"{int(row['tp_detected'])}/{int(row['fn_missed'])}",
            f"{100.0 * float(row['sensitivity']):.1f}%",
            f"{float(row['auc_vs_test_normals']):.3f}",
            f"{float(row['mean_score']):.1f}",
            f"{100.0 * float(row['fpr_at_threshold']):.1f}%",
            f"{float(row['threshold']):.1f}",
        ])

    fig_h = max(4.0, 1.8 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    ax.set_title(
        (
            "FastMRI Stratified Category Performance\n"
            f"Applied threshold={threshold:.1f}, achieved FPR={100.0 * fpr:.1f}% (Test normals only)"
        ),
        fontsize=12,
        pad=10,
    )

    table = ax.table(cellText=body, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.28)

    for (r_idx, c_idx), cell in table.get_celld().items():
        if r_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#264653")
            continue
        row = rows[r_idx - 1]
        if c_idx == 3:
            cell.set_facecolor(roc_sensitivity_color(float(row["sensitivity"])))
        elif c_idx == 4:
            cell.set_facecolor(roc_sensitivity_color(float(row["auc_vs_test_normals"])))
        else:
            cell.set_facecolor("#f8f9fa")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_patient_binary_heatmap_sum(
    results: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 500.0,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot total Binary_Sum_Heatmap per patient (sum of slice-level binary counts)."""
    patient_sums, patient_is_orig = collect_patient_binary_sums(results, allowed_case_folders)

    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(pid, keep=40) for pid, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_is_orig.get(pid, False) for pid, _ in sorted_items]
    base_color = "#4C72B0"
    colors = ["red" if v > threshold else base_color for v in values]

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Total binary heatmap pixels")
    ax.set_title(f"Patient total Binary_Sum_Heatmap (threshold = {threshold})")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"threshold={threshold}")
    secondary_level = 150000.0
    ax.axhline(secondary_level, color="gray", linestyle="--", linewidth=1.0, label=f"secondary={secondary_level}")
    ax.legend()

    num_anomalies = sum(1 for v in values if v > threshold)
    num_orig_anomalies = sum(1 for v, o in zip(values, orig_flags) if v > threshold and o)
    num_other_anomalies = num_anomalies - num_orig_anomalies
    summary_text = (
        f"Total patients: {len(values)}\n"
        f"Above threshold: {num_anomalies}\n"
        f"Orig above: {num_orig_anomalies}\n"
        f"Other above: {num_other_anomalies}"
    )
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_unique_patient_binary_token_sum(
    results: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 350.0,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot unique patient sum of Binary+Token values.

    Uses per-slice `Binary_Sum_Heatmap` (which already reflects Binary+Token when
    inference was run with --binary-include-token-surprisal).
    Patients are flagged as anomaly when total exceeds `threshold`.
    """
    patient_sums, patient_is_orig = collect_patient_binary_sums(results, allowed_case_folders)

    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(pid, keep=60) for pid, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_is_orig.get(pid, False) for pid, _ in sorted_items]

    base_color = "#4C72B0"
    colors = ["red" if v > threshold else base_color for v in values]

    width = max(10, min(0.3 * len(labels), 70))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Patient total Binary+Token sum")
    ax.set_title(f"Unique patient Binary+Token sum (anomaly if > {threshold})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.legend(loc="upper right")

    num_patients = len(values)
    num_anomalies = sum(1 for v in values if v > threshold)
    orig_anomalies = sum(1 for v, o in zip(values, orig_flags) if v > threshold and o)
    summary_text = (
        f"Unique patients: {num_patients}\n"
        f"Anomalies (> threshold): {num_anomalies}\n"
        f"Orig anomalies: {orig_anomalies}"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_patient_total_sharpness(
    results: List[dict],
    output_path: Path,
    top_n: Optional[int] = None,
    threshold: float = 8.0,
    allowed_case_folders: Optional[Set[str]] = None,
) -> None:
    """Plot total sharpness (sum of slice sharpness_score) per patient.

    Bars are red when total sharpness is below the threshold (flagged anomaly),
    blue otherwise. Patient labels containing "orig" are colored green.
    """
    patient_sums, patient_is_orig = collect_patient_sharpness_totals(results, allowed_case_folders)

    sorted_items = sorted(patient_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    labels = [truncate_label(pid, keep=60) for pid, _ in sorted_items]
    values = [v for _, v in sorted_items]
    orig_flags = [patient_is_orig.get(pid, False) for pid, _ in sorted_items]
    base_color = "#4C72B0"
    colors = ["red" if v < threshold else base_color for v in values]

    width = max(10, min(0.25 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")

    ax.set_ylabel("Total sharpness score (sum of slices)")
    ax.set_title(f"Patient total sharpness (anomaly if < {threshold})")

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")

    num_anomalies = sum(1 for v in values if v < threshold)
    num_orig_anomalies = sum(1 for v, o in zip(values, orig_flags) if v < threshold and o)
    num_other_anomalies = num_anomalies - num_orig_anomalies
    summary_text = (
        f"Total patients: {len(values)}\n"
        f"Below threshold: {num_anomalies}\n"
        f"Orig below: {num_orig_anomalies}\n"
        f"Other below: {num_other_anomalies}"
    )
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_stage_anomaly_counts(summary: Dict[str, Dict[str, int]], output_path: Path) -> Path:
    """Plot orig vs other anomaly counts for each pipeline stage."""
    stages = ["Global check 1", "Global check 2", "Local check"]
    stage_keys = ["stage1_sharpness", "stage2_binary_sum", "stage3_clamped_sum"]

    try:
        orig_counts = [summary[k]["orig"] for k in stage_keys]
        other_counts = [summary[k]["other"] for k in stage_keys]
    except KeyError as exc:
        raise ValueError(f"Missing stage summary key: {exc}") from exc

    x_pos = range(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(stages)), 5))
    ax.bar([x - width / 2 for x in x_pos], orig_counts, width, label="orig", color="green")
    ax.bar([x + width / 2 for x in x_pos], other_counts, width, label="other", color="#4C72B0")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Detected anomalies (count)")
    ax.set_title("Anomaly counts per stage")
    ax.legend()
    for idx, (o, r) in enumerate(zip(orig_counts, other_counts)):
        ax.text(idx - width / 2, o + 0.05, str(o), ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, r + 0.05, str(r), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_anomaly_pipeline(
    results: List[dict],
    patient_summary: List[dict],
    output_path: Path,
    sharpness_low: float = 5.0,
    sharpness_high: float = 15.0,
    binary_threshold: float = 18000.0,
    clamp_threshold: float = 160.0,
    allowed_case_folders: Optional[Set[str]] = None,
    top_n: Optional[int] = None,
) -> Dict[str, object]:
    """Pipeline plot: Global check 1 (sharpness), Global check 2 (Binary_Sum_Heatmap), Local check (clamped sum).

    Patients flagged in Global check 1 are still evaluated by later stages; only Global check 2 anomalies are skipped from Local check.
    Stage rules:
      - Global check 1 anomaly if sharpness < sharpness_low or sharpness > sharpness_high.
      - Global check 2 anomaly if Binary_Sum_Heatmap > binary_threshold.
      - Local check anomaly if total_clamped_pixel_sum > clamp_threshold.
    """
    sharp_sums, sharp_orig = collect_patient_sharpness_totals(results, allowed_case_folders)
    bin_sums, bin_orig = collect_patient_binary_sums(results, allowed_case_folders)
    clamp_sums, clamp_orig = collect_patient_clamp_totals_from_summary(patient_summary, allowed_case_folders)
    # Fill any missing patients using slice-level aggregation as a fallback
    if results:
        try:
            fallback_summary = aggregate_patient_clamp_from_results(results)
            fallback_totals, fallback_orig = collect_patient_clamp_totals_from_summary(fallback_summary, allowed_case_folders)
            for pid, val in fallback_totals.items():
                if pid not in clamp_sums:
                    clamp_sums[pid] = val
                    clamp_orig[pid] = fallback_orig.get(pid, False)
        except ValueError:
            pass

    # Log patient ID coverage for debugging
    sharp_pids = set(sharp_sums.keys())
    clamp_pids = set(clamp_sums.keys())
    missing_clamp = sharp_pids - clamp_pids
    if missing_clamp:
        logging.warning(
            "Pipeline: %d patients have sharpness but missing clamped sums (will use 0.0): %s",
            len(missing_clamp),
            list(missing_clamp)[:5] if len(missing_clamp) > 5 else list(missing_clamp),
        )
    logging.info(
        "Pipeline patient counts - sharpness: %d, binary: %d, clamped: %d",
        len(sharp_sums), len(bin_sums), len(clamp_sums),
    )

    def is_orig(pid: str) -> bool:
        return sharp_orig.get(pid) or bin_orig.get(pid) or clamp_orig.get(pid, False)

    # Use patient order based on descending sharpness to keep alignment across rows
    ordered_patients = sorted(sharp_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        ordered_patients = ordered_patients[:top_n]
    patient_ids = [pid for pid, _ in ordered_patients]

    stage1_vals = []
    stage1_colors = []
    stage1_anomaly = set()

    stage2_vals = []
    stage2_colors = []
    stage2_anomaly = set()

    stage3_vals = []
    stage3_colors = []

    base_color = "#4C72B0"
    gray_skip = "#B0B0B0"

    for pid in patient_ids:
        s_val = sharp_sums.get(pid, 0.0)
        s_anom = (s_val < sharpness_low) or (s_val > sharpness_high)
        stage1_vals.append(s_val)
        stage1_colors.append("red" if s_anom else base_color)
        if s_anom:
            stage1_anomaly.add(pid)

        # Stage 2: Skip patients already flagged in Stage 1
        if pid in stage1_anomaly:
            stage2_vals.append(0.0)
            stage2_colors.append(gray_skip)
        else:
            b_val = bin_sums.get(pid, 0.0)
            b_anom = b_val > binary_threshold
            stage2_vals.append(b_val)
            stage2_colors.append("red" if b_anom else base_color)
            if b_anom:
                stage2_anomaly.add(pid)

        # Stage 3: Skip patients flagged in Stage 1 OR Stage 2
        if pid in stage1_anomaly or pid in stage2_anomaly:
            stage3_vals.append(0.0)
            stage3_colors.append(gray_skip)
        else:
            c_val = clamp_sums.get(pid, 0.0)
            c_anom = c_val > clamp_threshold
            stage3_vals.append(c_val)
            stage3_colors.append("red" if c_anom else base_color)

    # Summaries per stage: count orig vs other anomalies
    def count_stage(anom_set: set[str]) -> Dict[str, int]:
        orig_cnt = sum(1 for pid in anom_set if is_orig(pid))
        return {
            "orig": orig_cnt,
            "other": len(anom_set) - orig_cnt,
            "total": len(anom_set),
        }

    stage1_summary = count_stage(stage1_anomaly)
    stage2_summary = count_stage(stage2_anomaly)
    # Stage 3 anomalies are patients that pass Stage 1 and Stage 2 but exceed the Local check threshold
    stage3_anomaly = {pid for pid, val, color in zip(patient_ids, stage3_vals, stage3_colors) if color == "red"}
    stage3_summary = count_stage(stage3_anomaly)

    summary = {
        "stage1_sharpness": stage1_summary,
        "stage2_binary_sum": stage2_summary,
        "stage3_clamped_sum": stage3_summary,
        "config": {
            "sharpness_low": sharpness_low,
            "sharpness_high": sharpness_high,
            "binary_threshold": binary_threshold,
            "clamp_threshold": clamp_threshold,
            "top_n": top_n,
        },
    }

    width = max(10, min(0.35 * len(patient_ids), 70))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(width, 12), sharex=True)
    x_pos = range(len(patient_ids))

    # Stage 1
    ax = axes[0]
    ax.bar(x_pos, stage1_vals, color=stage1_colors)
    ax.axhline(sharpness_low, color="red", linestyle="--", linewidth=1.2, label=f"low={sharpness_low}")
    ax.axhline(sharpness_high, color="red", linestyle=":", linewidth=1.2, label=f"high={sharpness_high}")
    ax.set_ylabel("Total sharpness")
    ax.set_title("Global check 1: Sharpness (anomaly if < low or > high)")
    ax.legend(loc="upper right")

    # Stage 2
    ax = axes[1]
    ax.bar(x_pos, stage2_vals, color=stage2_colors)
    ax.axhline(binary_threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={binary_threshold}")
    ax.set_ylabel("Binary_Sum_Heatmap")
    ax.set_title("Global check 2: Binary sum (feeds Local check)")
    ax.legend(loc="upper right")

    # Stage 3
    ax = axes[2]
    ax.bar(x_pos, stage3_vals, color=stage3_colors)
    ax.axhline(clamp_threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={clamp_threshold}")
    ax.set_ylabel("Total clamped sum")
    ax.set_title("Local check: Clamped sum (after Global checks)")
    ax.legend(loc="upper right")

    # Shared x labels
    axes[-1].set_xticks(list(x_pos))
    axes[-1].set_xticklabels([truncate_label(pid, keep=60) for pid in patient_ids], rotation=65, ha="right", fontsize=8)

    # Color patient labels green if they passed all checks
    all_anomalies = stage1_anomaly | stage2_anomaly | stage3_anomaly
    for text, pid in zip(axes[-1].get_xticklabels(), patient_ids):
        if pid not in all_anomalies:
            text.set_color("green")
            text.set_fontweight("bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    counts_path = output_path.with_name(output_path.stem + "_stage_counts.png")
    plot_stage_anomaly_counts(summary, counts_path)
    return summary


def plot_final_patient_anomaly_overview(
    results: List[dict],
    patient_summary: List[dict],
    output_path: Path,
    sharpness_low: float,
    sharpness_high: float,
    binary_threshold: float,
    clamp_threshold: float,
    allowed_case_folders: Optional[Set[str]] = None,
    top_n: Optional[int] = None,
) -> Dict[str, object]:
    """Generate a final per-patient anomaly vs normal overview.

    A patient is marked as anomaly if any stage triggers:
      - Stage 1: sharpness outside [sharpness_low, sharpness_high]
      - Stage 2: Binary_Sum_Heatmap > binary_threshold (evaluated only if Stage 1 passes)
      - Stage 3: total_clamped_pixel_sum > clamp_threshold (evaluated only if Stages 1 and 2 pass)
    """
    sharp_sums, sharp_orig = collect_patient_sharpness_totals(results, allowed_case_folders)
    bin_sums, bin_orig = collect_patient_binary_sums(results, allowed_case_folders)
    clamp_sums, clamp_orig = collect_patient_clamp_totals_from_summary(patient_summary, allowed_case_folders)

    # Fill missing clamped sums from slice-level aggregation
    if results:
        try:
            fallback_summary = aggregate_patient_clamp_from_results(results)
            fallback_totals, fallback_orig = collect_patient_clamp_totals_from_summary(fallback_summary, allowed_case_folders)
            for pid, val in fallback_totals.items():
                if pid not in clamp_sums:
                    clamp_sums[pid] = val
                    clamp_orig[pid] = fallback_orig.get(pid, False)
        except ValueError:
            pass

    # Map patient -> case_folders for summary counts
    patient_case_folders: Dict[str, Set[str]] = {}
    for item in results:
        if not matches_case_folder(item.get("case_folder"), allowed_case_folders):
            continue
        pid_raw = patient_id_from_item(item)
        if pid_raw is None:
            continue
        pid = strip_unknown_prefix(pid_raw)
        cf = str(item.get("case_folder", ""))
        if pid not in patient_case_folders:
            patient_case_folders[pid] = set()
        if cf:
            patient_case_folders[pid].add(cf)

    # Use patient order based on descending sharpness for stable alignment
    ordered_patients = sorted(sharp_sums.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        ordered_patients = ordered_patients[:top_n]
    patient_ids = [pid for pid, _ in ordered_patients]

    stage1_anomaly: Set[str] = set()
    stage2_anomaly: Set[str] = set()
    stage3_anomaly: Set[str] = set()

    final_flags: List[int] = []
    orig_flags: List[bool] = []

    def is_orig(pid: str) -> bool:
        return sharp_orig.get(pid) or bin_orig.get(pid) or clamp_orig.get(pid, False)

    for pid in patient_ids:
        s_val = sharp_sums.get(pid, 0.0)
        s_anom = (s_val < sharpness_low) or (s_val > sharpness_high)
        if s_anom:
            stage1_anomaly.add(pid)

        if pid in stage1_anomaly:
            b_anom = False
        else:
            b_val = bin_sums.get(pid, 0.0)
            b_anom = b_val > binary_threshold
            if b_anom:
                stage2_anomaly.add(pid)

        if pid in stage1_anomaly or pid in stage2_anomaly:
            c_anom = False
        else:
            c_val = clamp_sums.get(pid, 0.0)
            c_anom = c_val > clamp_threshold
            if c_anom:
                stage3_anomaly.add(pid)

        is_anom = pid in stage1_anomaly or pid in stage2_anomaly or pid in stage3_anomaly
        final_flags.append(1 if is_anom else 0)
        orig_flags.append(is_orig(pid))

    # Count anomalies per case_folder for context box
    anomaly_per_folder: Dict[str, int] = {}
    for pid, flag in zip(patient_ids, final_flags):
        if flag == 0:
            continue
        for cf in patient_case_folders.get(pid, []):
            anomaly_per_folder[cf] = anomaly_per_folder.get(cf, 0) + 1

    labels = [truncate_label(pid, keep=60) for pid in patient_ids]
    base_color = "#4C72B0"
    colors = ["red" if f == 1 else base_color for f in final_flags]

    width = max(10, min(0.3 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    ax.bar(x_positions, final_flags, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    for text, is_orig_flag, flag in zip(ax.get_xticklabels(), orig_flags, final_flags):
        if flag == 0 and is_orig_flag:
            text.set_color("green")
        elif flag == 1 and is_orig_flag:
            text.set_color("purple")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("Final status")
    ax.set_title("Patient anomaly pipeline - Final status")

    summary_counts = {
        "total_patients": len(final_flags),
        "anomalies": sum(final_flags),
        "orig_anomalies": sum(1 for flag, orig in zip(final_flags, orig_flags) if flag == 1 and orig),
        "other_anomalies": sum(final_flags) - sum(1 for flag, orig in zip(final_flags, orig_flags) if flag == 1 and orig),
    }

    if anomaly_per_folder:
        summary_text = (
            f"Total patients: {summary_counts['total_patients']}\n"
            f"Anomalies: {summary_counts['anomalies']}\n"
            f"Orig anomalies: {summary_counts['orig_anomalies']}\n"
            f"Other anomalies: {summary_counts['other_anomalies']}\n"
            "Anomalies per folder:\n" + "\n".join(f"{k}: {v}" for k, v in sorted(anomaly_per_folder.items()))
        )
    else:
        summary_text = (
            f"Total patients: {summary_counts['total_patients']}\n"
            f"Anomalies: {summary_counts['anomalies']}\n"
            f"Orig anomalies: {summary_counts['orig_anomalies']}\n"
            f"Other anomalies: {summary_counts['other_anomalies']}"
        )

    ax.text(
        0.98,
        0.97,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "summary": summary_counts,
        "anomaly_per_folder": anomaly_per_folder,
        "config": {
            "sharpness_low": sharpness_low,
            "sharpness_high": sharpness_high,
            "binary_threshold": binary_threshold,
            "clamp_threshold": clamp_threshold,
            "top_n": top_n,
        },
    }


def plot_bars(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    output_path: Path,
    cutoff_y: float | None = None,
) -> None:
    width = max(10, min(0.2 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 6))
    x_positions = range(len(labels))
    base_color = "#4C72B0"
    if cutoff_y is not None:
        colors = ["red" if v > cutoff_y else base_color for v in values]
    else:
        colors = base_color
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if cutoff_y is not None:
        ax.axhline(cutoff_y, color="red", linestyle="--", linewidth=1.2, label=f"cutoff={cutoff_y}")
        ax.legend()
    ax.margins(x=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_anomaly_patients(
    labels: List[str],
    values: List[int],
    orig_flags: List[bool],
    anomaly_per_folder: Dict[str, int],
    cutoff_count: int,
    output_path: Path,
) -> None:
    width = max(8, min(0.3 * len(labels), 60))
    fig, ax = plt.subplots(figsize=(width, 5))
    x_positions = range(len(labels))
    colors = ["red" if v == 1 else "#4C72B0" for v in values]
    ax.bar(x_positions, values, color=colors)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    # Color patient labels green if case_folder == "orig"
    for text, is_orig in zip(ax.get_xticklabels(), orig_flags):
        if is_orig:
            text.set_color("green")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("status")
    ax.set_title(f"Anomaly patients (> {cutoff_count} slices above cutoff)")
    # Add summary text box
    if anomaly_per_folder:
        summary_text = "Anomaly count per case_folder:\n" + "\n".join(
            f"{k}: {v}" for k, v in sorted(anomaly_per_folder.items())
        )
        ax.text(
            0.98,
            0.97,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot perceptual bar charts from a results JSON.")

    default_json = "/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_FastMRI/Inference_FastMRI_Results_SOTA_Rec_Heal__Automatic_ALL_Anomalies_full_val/results_v4_zscore.json"
    
    
    parser.add_argument(
        "--input",
        type=Path,
        default=default_json,
        help="Path to results_v4_zscore.json OR root folder containing many such JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save plots. Default: input folder "
            "(or input file parent for single-file mode)."
        ),
    )
    parser.add_argument("--top-n", type=int, default=None, help="Limit to first N items (default: all)")
    parser.add_argument(
        "--case-folder",
        dest="case_folders",
        action="append",
        help="Keep entries whose case_folder contains this substring (case-insensitive). Repeat for multiple filters.",
    )
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        help="Keep entries whose category contains this token (case-insensitive). Repeat for multiple categories.",
    )
    parser.add_argument(
        "--catagory",
        dest="categories",
        action="append",
        help="Alias for --category.",
    )
    parser.add_argument("--slice-count-cutoff", type=int, default=5, help="Threshold for number of slices above metric cutoff to mark patient as anomaly (default: 5)")
    parser.add_argument("--first-heatmap-threshold", type=float, default=10000.0, help="Threshold for total first-heatmap pixel sum to color patient bars (default: 500)")
    parser.add_argument("--threshold-lpips-in-rec", type=float, default=5000.0, help="Threshold for per-patient lpips_input_recon_sum_mask plot")
    
    parser.add_argument("--clamp-sum-threshold", type=float, default=140.0, help="Threshold for total clamped pixel sum to mark patient as anomaly (default: 450)")
    parser.add_argument("--binary-heatmap-threshold", type=float, default=700.0, help="Threshold for per-patient Binary_Sum_Heatmap plot")
    parser.add_argument("--sharpness-threshold", type=float, default=5.0, help="Threshold for per-patient total sharpness plot (anomaly if below)")
    parser.add_argument("--sharpness-low-threshold", type=float, default=7.0, help="Stage-1 sharpness lower bound (anomaly if below)")
    parser.add_argument("--sharpness-high-threshold", type=float, default=20.0, help="Stage-1 sharpness upper bound (anomaly if above)")
    parser.add_argument(
        "--combined-threshold",
        type=float,
        default=150.0,
        help="Threshold for combined token_surprisal_hot_px + Binary_Sum_Heatmap (used for red bars)",
    )
    parser.add_argument(
        "--min-red-bars-per-patient",
        type=int,
        default=0,
        help="Patient is counted as anomaly when red-bar count is strictly greater than this value (default: 2)",
    )
    parser.add_argument(
        "--sum-all-bars-threshold",
        type=float,
        default=80.0,
        help="Threshold for per-patient sum of all combined bars (token_surprisal_hot_px + Binary_Sum_Heatmap)",
    )
    parser.add_argument(
        "--binary-token-patient-threshold",
        type=float,
        default=1019.0,
        help="Threshold for unique-patient Binary+Token sum plot (anomaly if above)",
    )
    parser.add_argument(
        "--disable-fastmri-roc",
        action="store_true",
        help="Disable dataset-level FastMRI ROC/AUC and threshold-table generation.",
    )
    parser.add_argument(
        "--include-validation-in-roc",
        action="store_true",
        help="Include Validation_samples in ROC normal class (default: excluded; Test-only normals).",
    )
    parser.add_argument(
        "--expected-test-normal-cases",
        type=int,
        default=30,
        help="Expected number of Test normal patients in ROC analysis.",
    )
    parser.add_argument(
        "--roc-target-fpr",
        type=float,
        default=0.20,
        help="Target max FPR for recommended threshold in ROC outputs.",
    )
    parser.add_argument(
        "--roc-merged-json-output",
        type=Path,
        default=None,
        help="Output JSON path for merged FastMRI ROC payload.",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=None,
        help="Output image path for FastMRI ROC curve.",
    )
    parser.add_argument(
        "--show-best-j-marker",
        action="store_true",
        help="Show Best-Youden marker on ROC curve (default hidden).",
    )
    parser.add_argument(
        "--roc-metrics-output",
        type=Path,
        default=None,
        help="Output JSON path for FastMRI ROC metrics.",
    )
    parser.add_argument(
        "--roc-ci-bootstrap-samples",
        type=int,
        default=2000,
        help="Number of stratified bootstrap samples for ROC/AUC confidence intervals (set 0 to disable).",
    )
    parser.add_argument(
        "--roc-ci-confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for ROC/AUC intervals (default: 0.95).",
    )
    parser.add_argument(
        "--roc-ci-random-seed",
        type=int,
        default=42,
        help="Random seed for ROC bootstrap CI (set negative for non-deterministic seed).",
    )
    parser.add_argument(
        "--roc-ci-fpr-grid-size",
        type=int,
        default=201,
        help="Number of FPR grid points used when constructing ROC CI band.",
    )
    parser.add_argument(
        "--roc-threshold-table-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for threshold-by-threshold ROC table.",
    )
    parser.add_argument(
        "--roc-threshold-table-figure-prefix",
        type=Path,
        default=None,
        help="Output prefix for paginated ROC threshold table figures.",
    )
    parser.add_argument(
        "--roc-table-rows-per-figure",
        type=int,
        default=40,
        help="Number of rows per ROC threshold table figure page.",
    )
    parser.add_argument(
        "--include-infinite-threshold-rows",
        action="store_true",
        help="Include inf/-inf sentinel thresholds in ROC threshold table outputs.",
    )
    parser.add_argument(
        "--category-table-json-output",
        type=Path,
        default=None,
        help="Output JSON path for FastMRI category stratified table payload.",
    )
    parser.add_argument(
        "--category-table-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for FastMRI category stratified table.",
    )
    parser.add_argument(
        "--category-table-figure-output",
        type=Path,
        default=None,
        help="Output PNG path for FastMRI category stratified table figure.",
    )
    return parser.parse_args()


def find_result_json_files(input_path: Path) -> List[Path]:
    """Resolve one or many results_v4_zscore.json files from input path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    json_files = sorted(p for p in input_path.rglob("results_v4_zscore.json") if p.is_file())
    if not json_files:
        raise ValueError(f"No results_v4_zscore.json files found under: {input_path}")
    return json_files


def resolve_processing_json_files(input_path: Path, discovered: List[Path]) -> Tuple[List[Path], Path, bool]:
    """Expand single-file root input to sibling dataset JSONs when available."""
    if input_path.is_file():
        parent = input_path.parent
        sibling_jsons = sorted(p for p in parent.rglob("results_v4_zscore.json") if p.is_file())
        nested_jsons = [p for p in sibling_jsons if p.parent != parent]
        if nested_jsons:
            logging.info(
                "Input is one root JSON; using %d sibling dataset JSON files under %s",
                len(nested_jsons),
                parent,
            )
            return nested_jsons, parent, True
        return discovered, input_path.parent, False

    return discovered, input_path, True


def run_plots_for_payload(
    json_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    allowed_case_folders: Optional[Set[str]],
    allowed_categories: Optional[Set[str]],
) -> Optional[Dict[str, object]]:
    """Run selected plots for one JSON payload. Returns per-run summary row on success."""
    logging.info("Loading results from %s", json_path)
    payload = load_payload(json_path)
    results = payload.get("results", [])
    patient_summary = payload.get("patient_summary", [])

    if not patient_summary:
        logging.info("patient_summary missing/empty for %s; aggregating from slice-level", json_path.name)
        patient_summary = aggregate_patient_clamp_from_results(results)

    if allowed_categories:
        results = filter_items_by_category(results, allowed_categories)
        patient_summary = filter_items_by_category(patient_summary, allowed_categories)
        logging.info(
            "After category filtering (%s): %d slices, %d patient-summary entries",
            json_path.parent.name,
            len(results),
            len(patient_summary),
        )

    if allowed_case_folders:
        results = filter_items_by_case_folder(results, allowed_case_folders)
        patient_summary = filter_items_by_case_folder(patient_summary, allowed_case_folders)
        logging.info(
            "After case-folder filtering (%s): %d slices, %d patient-summary entries",
            json_path.parent.name,
            len(results),
            len(patient_summary),
        )

    try:
        if not results:
            raise ValueError("No slices remain after applying filters (category/case-folder).")

        source_label = json_path.parent.name
        if json_path.name:
            source_label = f"{source_label}/{json_path.name}"

        patient_detection_rows = build_patient_detection_overview_rows(
            results=results,
            patient_summary=patient_summary,
            top_n=args.top_n,
        )
        patient_detection_table_path = output_dir / "Patient_detection_overview_table.png"
        plot_patient_detection_overview_table(
            rows=patient_detection_rows,
            output_path=patient_detection_table_path,
            source_name=source_label,
        )
        patient_detection_json_path, patient_detection_csv_path = write_patient_detection_overview_outputs(
            rows=patient_detection_rows,
            output_dir=output_dir,
            source_name=source_label,
        )

        normal_nobbox_rows = build_normal_nobbox_fp_overview_rows(patient_detection_rows)
        normal_nobbox_table_path = output_dir / "NormalNoBBox_false_positive_pixels_overview_table.png"
        plot_normal_nobbox_fp_overview_table(
            overview_rows=normal_nobbox_rows,
            output_path=normal_nobbox_table_path,
            source_name=source_label,
        )
        normal_nobbox_json_path, normal_nobbox_csv_path = write_normal_nobbox_fp_overview_outputs(
            overview_rows=normal_nobbox_rows,
            output_dir=output_dir,
            source_name=source_label,
        )
        logging.info(
            "Saved patient detection overview -> %s, %s, %s | no-bbox FP overview -> %s, %s, %s",
            patient_detection_table_path,
            patient_detection_json_path,
            patient_detection_csv_path,
            normal_nobbox_table_path,
            normal_nobbox_json_path,
            normal_nobbox_csv_path,
        )

        unique_binary_token_sum_path = output_dir / "Unique_patient_BinaryPlusToken_sum.png"
        plot_unique_patient_binary_token_sum(
            results,
            unique_binary_token_sum_path,
            threshold=float(args.binary_token_patient_threshold),
            top_n=args.top_n,
            allowed_case_folders=allowed_case_folders,
        )

        sensitivity_summary = compute_binary_token_patient_sensitivity(
            results=results,
            threshold=float(args.binary_token_patient_threshold),
            allowed_case_folders=allowed_case_folders,
        )

        logging.info(
            "Binary+Token sensitivity @ %.3f -> TP=%d, FN=%d, sensitivity=%.2f%%",
            float(args.binary_token_patient_threshold),
            int(sensitivity_summary.get("detected_tp", 0)),
            int(sensitivity_summary.get("missed_fn", 0)),
            100.0 * float(sensitivity_summary.get("sensitivity", 0.0)),
        )

        per_run_row: Dict[str, object] = {
            "source": source_label,
            "threshold": float(sensitivity_summary.get("threshold", 0.0)),
            "total_patients": int(sensitivity_summary.get("total_patients", 0)),
            "anomaly_patients": int(sensitivity_summary.get("anomaly_patients", 0)),
            "normal_patients": int(sensitivity_summary.get("normal_patients", 0)),
            "tp": int(sensitivity_summary.get("detected_tp", 0)),
            "fn": int(sensitivity_summary.get("missed_fn", 0)),
            "fp": int(sensitivity_summary.get("false_positive", 0)),
            "tn": int(sensitivity_summary.get("true_negative", 0)),
            "sensitivity": float(sensitivity_summary.get("sensitivity", 0.0)),
            "specificity": float(sensitivity_summary.get("specificity", 0.0)),
        }

        per_run_patient_overview_row: Dict[str, object] = {
            # Patient-level confusion (aligned with patient_detection_rows / bbox-based detection):
            # GT positive => patient has at least one GT-bbox slice.
            # GT negative => patient has zero GT-bbox slices.
            # Pred positive =>
            #   - GT-positive patients: detected=True (has TP evidence)
            #   - GT-negative patients: highlighted no-bbox FP pixels > 0
            "tp": int(
                sum(
                    1
                    for r in patient_detection_rows
                    if (
                        int(r.get("num_slices_with_ground_truth_bbox", 0) or 0) > 0
                        and bool(r.get("detected", False))
                    )
                )
            ),
            "fp": int(
                sum(
                    1
                    for r in patient_detection_rows
                    if (
                        int(r.get("num_slices_with_ground_truth_bbox", 0) or 0) == 0
                        and int(r.get("highlighted_anomaly_pixels_no_bbox", 0) or 0) > 0
                    )
                )
            ),
            "fn": int(
                sum(
                    1
                    for r in patient_detection_rows
                    if (
                        int(r.get("num_slices_with_ground_truth_bbox", 0) or 0) > 0
                        and (not bool(r.get("detected", False)))
                    )
                )
            ),
            "source": source_label,
            "total_patients": int(len(patient_detection_rows)),
            "detected_patients": int(sum(1 for r in patient_detection_rows if bool(r.get("detected", False)))),
            "detection_rate": (
                float(sum(1 for r in patient_detection_rows if bool(r.get("detected", False))))
                / float(len(patient_detection_rows))
            ) if patient_detection_rows else 0.0,
            "total_tp_slices": int(sum(int(r.get("num_true_positive_slices", 0) or 0) for r in patient_detection_rows)),
            "total_gt_bbox_slices": int(sum(int(r.get("num_slices_with_ground_truth_bbox", 0) or 0) for r in patient_detection_rows)),
            "total_no_bbox_slices": int(sum(int(r.get("num_no_bbox_slices", 0) or 0) for r in patient_detection_rows)),
            "total_no_bbox_fp_px": int(sum(int(r.get("highlighted_anomaly_pixels_no_bbox", 0) or 0) for r in patient_detection_rows)),
            "total_slices": int(sum(int(r.get("num_slices", 0) or 0) for r in patient_detection_rows)),
            "sum_ground_truth_bbox_pixels": int(sum(int(r.get("sum_ground_truth_bbox_pixels", 0) or 0) for r in patient_detection_rows)),
            "sum_predicted_anomaly_pixels_inside_bbox": int(sum(int(r.get("sum_predicted_anomaly_pixels_inside_bbox", 0) or 0) for r in patient_detection_rows)),
            "sum_predicted_anomaly_pixels_outside_bbox": int(sum(int(r.get("sum_predicted_anomaly_pixels_outside_bbox", 0) or 0) for r in patient_detection_rows)),
        }

        inside_px = int(per_run_patient_overview_row["sum_predicted_anomaly_pixels_inside_bbox"])
        outside_px = int(per_run_patient_overview_row["sum_predicted_anomaly_pixels_outside_bbox"])
        detected_patients = int(per_run_patient_overview_row["detected_patients"])
        total_patients_for_metrics = int(per_run_patient_overview_row["total_patients"])
        fp_ratio = None
        if inside_px > 0:
            fp_ratio = float(outside_px) / float(inside_px)

        paper_precision = (
            float(sum(float(r.get("pixel_precision", 0.0) or 0.0) for r in patient_detection_rows)) / float(total_patients_for_metrics)
        ) if total_patients_for_metrics > 0 else 0.0
        paper_f1 = (
            float(sum(float(r.get("pixel_f1_score", 0.0) or 0.0) for r in patient_detection_rows)) / float(total_patients_for_metrics)
        ) if total_patients_for_metrics > 0 else 0.0
        tp_event_rate = (float(detected_patients) / float(total_patients_for_metrics)) if total_patients_for_metrics > 0 else 0.0

        per_run_patient_overview_row["false_positive_ratio"] = fp_ratio
        per_run_patient_overview_row["pixel_precision"] = float(paper_precision)
        per_run_patient_overview_row["pixel_tp_ratio"] = float(tp_event_rate)
        per_run_patient_overview_row["f1_score"] = float(paper_f1)
        per_run_patient_overview_row["precision"] = float(paper_precision)
        per_run_patient_overview_row["recall"] = float(tp_event_rate)

        denom_precision = int(per_run_patient_overview_row["tp"]) + int(per_run_patient_overview_row["fp"])
        per_run_patient_overview_row["classification_precision"] = (
            float(int(per_run_patient_overview_row["tp"])) / float(denom_precision)
        ) if denom_precision > 0 else 0.0

        denom_recall = int(per_run_patient_overview_row["tp"]) + int(per_run_patient_overview_row["fn"])
        per_run_patient_overview_row["classification_recall"] = (
            float(int(per_run_patient_overview_row["tp"])) / float(denom_recall)
        ) if denom_recall > 0 else 0.0
    except ValueError as exc:
        logging.warning("Skipping plots for %s: %s", json_path, exc)
        return None

    logging.info("Saved plots -> %s", output_dir)
    return {
        "binary_token_overview": per_run_row,
        "patient_detection_overview": per_run_patient_overview_row,
    }


def run_fastmri_roc_analysis(
    json_files: List[Path],
    output_root: Path,
    args: argparse.Namespace,
    allowed_case_folders: Optional[Set[str]],
    allowed_categories: Optional[Set[str]],
) -> None:
    """Run dataset-level FastMRI ROC/AUC and table generation."""
    merged_payload = merge_fastmri_json_payloads_for_roc(
        input_paths=json_files,
        allowed_case_folders=allowed_case_folders,
        allowed_categories=allowed_categories,
        include_validation_normals=bool(args.include_validation_in_roc),
    )

    merged_json_output = args.roc_merged_json_output or (output_root / "FastMRI_ROC_Merged_BinaryToken.json")
    write_json_file(merged_payload, merged_json_output)

    patient_scores = merged_payload.get("merged_patient_scores_binary_token", [])
    if not isinstance(patient_scores, list) or not patient_scores:
        raise ValueError("FastMRI merged patient scores are empty; cannot run ROC analysis")

    roc_metrics = compute_fastmri_roc_and_auc(
        patient_scores=patient_scores,
        expected_test_normals=int(args.expected_test_normal_cases),
        bootstrap_samples=max(0, int(args.roc_ci_bootstrap_samples)),
        confidence_level=float(args.roc_ci_confidence_level),
        bootstrap_random_seed=(None if int(args.roc_ci_random_seed) < 0 else int(args.roc_ci_random_seed)),
        ci_fpr_grid_size=max(2, int(args.roc_ci_fpr_grid_size)),
    )

    fixed_threshold = float(args.binary_token_patient_threshold)
    fixed_eval = evaluate_threshold_on_patient_scores(patient_scores, fixed_threshold)
    roc_metrics["fixed_threshold_evaluation_binary_token_patient_threshold"] = fixed_eval

    target_fpr = float(args.roc_target_fpr)
    selected_for_target_fpr = select_threshold_for_target_fpr(roc_metrics, target_fpr)
    roc_metrics["recommended_threshold_for_target_fpr"] = {
        "target_fpr": target_fpr,
        **selected_for_target_fpr,
    }
    two_pct_reference = select_threshold_for_target_fpr(roc_metrics, 0.02)
    roc_metrics["reference_threshold_for_2pct_fpr"] = {
        "target_fpr": 0.02,
        **two_pct_reference,
    }

    roc_metrics_output = args.roc_metrics_output or (output_root / "FastMRI_ROC_binary_token_metrics.json")
    write_json_file(roc_metrics, roc_metrics_output)

    roc_output = args.roc_output or (output_root / "FastMRI_ROC_binary_token_curve.png")
    plot_fastmri_roc_curve(
        roc_metrics,
        roc_output,
        show_best_threshold_marker=bool(args.show_best_j_marker),
    )

    roc_threshold_rows = build_roc_threshold_table_rows(
        roc_metrics,
        include_infinite=bool(args.include_infinite_threshold_rows),
    )
    roc_threshold_table_csv_output = args.roc_threshold_table_csv_output or (output_root / "FastMRI_ROC_Threshold_Table.csv")
    write_roc_threshold_table_csv(roc_threshold_rows, roc_threshold_table_csv_output)

    roc_threshold_table_prefix = args.roc_threshold_table_figure_prefix or (output_root / "FastMRI_ROC_Threshold_Table")
    roc_threshold_table_figures = plot_roc_threshold_table_figures(
        rows=roc_threshold_rows,
        output_prefix=roc_threshold_table_prefix,
        rows_per_page=max(1, int(args.roc_table_rows_per_figure)),
    )

    category_payload = compute_fastmri_category_stratified_performance(
        patient_scores=patient_scores,
        threshold=fixed_threshold,
    )
    category_payload["target_fpr"] = target_fpr
    category_payload["recommended_threshold_for_target_fpr"] = selected_for_target_fpr
    category_payload["reference_threshold_for_2pct_fpr"] = two_pct_reference

    category_json_output = args.category_table_json_output or (output_root / "FastMRI_Category_Stratified_Performance_Table.json")
    write_json_file(category_payload, category_json_output)

    category_csv_output = args.category_table_csv_output or (output_root / "FastMRI_Category_Stratified_Performance_Table.csv")
    write_fastmri_category_stratified_csv(category_payload.get("rows", []), category_csv_output)

    category_figure_output = args.category_table_figure_output or (output_root / "FastMRI_Category_Stratified_Performance_Table.png")
    plot_fastmri_category_stratified_table_figure(category_payload, category_figure_output)

    summary = roc_metrics.get("summary", {})
    if summary.get("auc_ci_lower") is not None and summary.get("auc_ci_upper") is not None:
        logging.info(
            "FastMRI ROC complete: AUC=%.4f (%d%% CI %.4f-%.4f), anomalies=%s, test_normals=%s, validation_excluded=%s",
            float(summary.get("auc", 0.0)),
            int(round(float(summary.get("roc_confidence_level", 0.95)) * 100.0)),
            float(summary.get("auc_ci_lower", 0.0)),
            float(summary.get("auc_ci_upper", 0.0)),
            summary.get("num_anomalies"),
            summary.get("num_test_normals"),
            summary.get("num_validation_excluded"),
        )
    else:
        logging.info(
            "FastMRI ROC complete: AUC=%.4f, anomalies=%s, test_normals=%s, validation_excluded=%s",
            float(summary.get("auc", 0.0)),
            summary.get("num_anomalies"),
            summary.get("num_test_normals"),
            summary.get("num_validation_excluded"),
        )
    logging.info(
        "Fixed threshold (--binary-token-patient-threshold=%.3f): FPR=%.2f%%, sensitivity=%.2f%%",
        fixed_threshold,
        100.0 * float(fixed_eval.get("fpr", 0.0)),
        100.0 * float(fixed_eval.get("sensitivity", 0.0)),
    )
    logging.info(
        "Recommended threshold for target FPR<=%.1f%%: %s (actual FPR=%.2f%%, sensitivity=%.2f%%)",
        100.0 * target_fpr,
        format_threshold_value(parse_threshold_json(selected_for_target_fpr.get("threshold", "inf"))),
        100.0 * float(selected_for_target_fpr.get("fpr", 0.0)),
        100.0 * float(selected_for_target_fpr.get("sensitivity", 0.0)),
    )
    logging.info(
        "Reference threshold for FPR<=2%%: %s (actual FPR=%.2f%%, sensitivity=%.2f%%)",
        format_threshold_value(parse_threshold_json(two_pct_reference.get("threshold", "inf"))),
        100.0 * float(two_pct_reference.get("fpr", 0.0)),
        100.0 * float(two_pct_reference.get("sensitivity", 0.0)),
    )
    logging.info("FastMRI merged ROC JSON -> %s", merged_json_output)
    logging.info("FastMRI ROC metrics JSON -> %s", roc_metrics_output)
    logging.info("FastMRI ROC curve -> %s", roc_output)
    logging.info("FastMRI ROC threshold table CSV -> %s", roc_threshold_table_csv_output)
    logging.info(
        "FastMRI ROC threshold table figure pages generated: %d (prefix: %s)",
        len(roc_threshold_table_figures),
        roc_threshold_table_prefix,
    )
    logging.info("FastMRI category stratified table JSON -> %s", category_json_output)
    logging.info("FastMRI category stratified table CSV -> %s", category_csv_output)
    logging.info("FastMRI category stratified table figure -> %s", category_figure_output)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    allowed_case_folders: Optional[Set[str]] = None
    if args.case_folders:
        allowed_case_folders = {cf.strip().lower() for cf in args.case_folders if cf and cf.strip()}
        if allowed_case_folders:
            logging.info(
                "Filtering to case_folders containing: %s (case-insensitive substring match)",
                ", ".join(sorted(allowed_case_folders)),
            )

    allowed_categories: Optional[Set[str]] = None
    if args.categories:
        allowed_categories = {c.strip().lower() for c in args.categories if c and c.strip()}
        if allowed_categories:
            logging.info(
                "Filtering to categories containing: %s (case-insensitive substring match)",
                ", ".join(sorted(allowed_categories)),
            )

    discovered_json_files = find_result_json_files(args.input)
    json_files, processing_root, batch_mode = resolve_processing_json_files(args.input, discovered_json_files)
    logging.info("Discovered %d JSON file(s)", len(json_files))

    output_root = args.output_dir if args.output_dir is not None else processing_root
    if args.output_dir is None:
        logging.info("Using input-root output directory: %s", output_root)
    else:
        logging.info("Using explicit output directory: %s", output_root)

    ok = 0
    combined_rows: List[Dict[str, object]] = []
    combined_patient_detection_rows: List[Dict[str, object]] = []
    for json_path in json_files:
        if batch_mode:
            if args.output_dir is None:
                target_output_dir = json_path.parent
            else:
                relative_parent = json_path.parent.relative_to(processing_root)
                target_output_dir = output_root / relative_parent
        else:
            target_output_dir = output_root

        per_run_row = run_plots_for_payload(
            json_path=json_path,
            output_dir=target_output_dir,
            args=args,
            allowed_case_folders=allowed_case_folders,
            allowed_categories=allowed_categories,
        )
        if per_run_row is not None:
            ok += 1
            binary_row = per_run_row.get("binary_token_overview") if isinstance(per_run_row, dict) else None
            patient_row = per_run_row.get("patient_detection_overview") if isinstance(per_run_row, dict) else None
            if isinstance(binary_row, dict):
                combined_rows.append(binary_row)
            if isinstance(patient_row, dict):
                combined_patient_detection_rows.append(patient_row)

    if combined_rows:
        combined_table_path = output_root / "BinaryToken_combined_overview_table.png"
        plot_combined_binary_token_overview_table(
            run_rows=combined_rows,
            output_path=combined_table_path,
        )

        combined_json_path, combined_csv_path = write_combined_binary_token_overview_outputs(
            run_rows=combined_rows,
            output_dir=output_root,
        )
        logging.info(
            "Saved combined overview outputs -> %s, %s, %s",
            combined_table_path,
            combined_json_path,
            combined_csv_path,
        )

    if combined_patient_detection_rows:
        combined_patient_detection_table_path = output_root / "PatientDetection_combined_overview_table.png"
        plot_combined_patient_detection_overview_table(
            run_rows=combined_patient_detection_rows,
            output_path=combined_patient_detection_table_path,
        )

        combined_patient_detection_json_path, combined_patient_detection_csv_path = write_combined_patient_detection_overview_outputs(
            run_rows=combined_patient_detection_rows,
            output_dir=output_root,
        )
        logging.info(
            "Saved combined patient-detection overview outputs -> %s, %s, %s",
            combined_patient_detection_table_path,
            combined_patient_detection_json_path,
            combined_patient_detection_csv_path,
        )

    if not args.disable_fastmri_roc:
        try:
            run_fastmri_roc_analysis(
                json_files=json_files,
                output_root=output_root,
                args=args,
                allowed_case_folders=allowed_case_folders,
                allowed_categories=allowed_categories,
            )
        except ValueError as exc:
            logging.warning("Skipping FastMRI ROC analysis: %s", exc)

    logging.info("Completed plot generation: %d/%d JSON files succeeded", ok, len(json_files))
    return



if __name__ == "__main__":
    main()
