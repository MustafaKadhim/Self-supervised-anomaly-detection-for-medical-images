import argparse
from bisect import bisect_right
import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from math import inf, isinf
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt

DEFAULT_ROC_INPUT_PATHS: Tuple[Path, ...] = (
    Path("/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_Volunteer_Clinical/results_v4_zscore.json"),
    Path("/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_ESTROTestdata_CervixBrachy/results_v4_zscore.json"),
    Path("/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_Global_Local_Clinical/results_v4_zscore.json"),
    Path("/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_test_LUND_PROBE_extended_npy/results_v4_zscore.json"),
    Path("/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_SpacerResampled/results_v4_zscore.json"),
)

SYNTHETIC_GLOBAL_FOCUS_CATEGORIES: Tuple[str, ...] = (
    "RandomGhosting",
    "RandomNoise",
    "RandomSpike",
    "RandomMotion",
    "WholeImageGaussian",
)

CLINICAL_FOCUS_CATEGORIES: Tuple[str, ...] = (
    "Unknown",
    "Spacer",
    "Stor_T2_till_sCT",
    "Stor_T2_to_sCT",
    "ClinicalVariations",
)

CANONICAL_CATEGORY_RULES: Tuple[Tuple[str, str], ...] = (
    ("WholeImageGaussian", "wholeimagegaussian"),
    ("RandomGhosting", "randomghosting"),
    ("RandomNoise", "randomnoise"),
    ("RandomSpike", "randomspike"),
    ("RandomMotion", "randommotion"),
    ("RandomCTVInsertion", "randomctvinsertion"),
    ("CTVAverage", "ctvaverage"),
    ("CTVBlur", "ctvblur"),
    ("Stor_T2_till_sCT", "stor_t2_till_sct"),
    ("Spacer", "spacer"),
    ("ClinicalVariations", "clinicalvariations"),
    ("Unknown", "unknown"),
    ("SyntheticVariations", "syntheticvariations"),
)

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

def load_payload(json_path: Path) -> Dict:
    with json_path.open("r") as f:
        return json.load(f)

def load_results(json_path: Path) -> List[dict]:
    payload = load_payload(json_path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("results field is missing or not a list")
    return results

def write_json_file(payload: Dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return output_path

def format_threshold_value(threshold: float) -> str:
    if isinf(threshold):
        return "inf" if threshold > 0 else "-inf"
    return f"{threshold:.6g}"

def threshold_json_value(threshold: float) -> str | float:
    if isinf(threshold):
        return "inf" if threshold > 0 else "-inf"
    return float(threshold)

def parse_threshold_json(threshold: str | float | int) -> float:
    if isinstance(threshold, str):
        lower = threshold.lower()
        if lower == "inf":
            return inf
        if lower == "-inf":
            return -inf
        return float(threshold)
    return float(threshold)

def patient_id_from_item(item: dict) -> Optional[str]:
    filename = item.get("filename") or Path(item.get("path", "")).name
    if not filename:
        return None
    stem = Path(filename).stem
    if "_slice_" in stem:
        return stem.split("_slice_", 1)[0]
    return stem

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

def is_orig_from_identifiers(patient_id: str, case_folders: Set[str]) -> bool:
    pid_lower = patient_id.lower()
    if "orig" in pid_lower:
        return True
    return any((cf.lower() == "orig") or ("orig" in cf.lower()) for cf in case_folders)

def patient_key_from_merged_result(item: dict) -> str:
    source_id = str(item.get("source_id") or "unknown_source")
    pid_raw = patient_id_from_item(item)
    if pid_raw is None:
        pid_raw = item.get("patient_id") or item.get("case_folder") or "unknown"
    pid = strip_unknown_prefix(str(pid_raw)) or "unknown"
    return f"{source_id}::{pid}"

def aggregate_patient_sum_of_all_bars(
    results: List[dict],
    source_json: Path,
    allowed_case_folders: Optional[Set[str]] = None,
) -> List[dict]:
    """Aggregate per-patient sum(token_surprisal_hot_px + Binary_Sum_Heatmap)."""
    patient_map: Dict[str, Dict[str, object]] = {}

    for item in results:
        case_folder = item.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue

        token_val = item.get("token_surprisal_hot_px")
        binary_val = item.get("Binary_Sum_Heatmap")
        if token_val is None and binary_val is None:
            continue

        pid_raw = patient_id_from_item(item)
        if pid_raw is None:
            pid_raw = item.get("patient_id") or case_folder or "unknown"
        patient_id = strip_unknown_prefix(str(pid_raw)) or "unknown"

        if patient_id not in patient_map:
            patient_map[patient_id] = {
                "patient_id": patient_id,
                "source_json": str(source_json),
                "source_id": source_json.parent.name,
                "case_folders": set(),
                "sum_all_bars_score": 0.0,
                "sum_token_surprisal_hot_px": 0.0,
                "sum_binary_sum_heatmap": 0.0,
                "num_slices": 0,
            }

        entry = patient_map[patient_id]
        entry["num_slices"] += 1
        if case_folder is not None:
            entry["case_folders"].add(str(case_folder))

        token_float = float(token_val) if token_val is not None else 0.0
        binary_float = float(binary_val) if binary_val is not None else 0.0
        entry["sum_token_surprisal_hot_px"] += token_float
        entry["sum_binary_sum_heatmap"] += binary_float
        entry["sum_all_bars_score"] += token_float + binary_float

    rows: List[dict] = []
    for entry in patient_map.values():
        case_folders = sorted(entry["case_folders"])
        is_orig = is_orig_from_identifiers(str(entry["patient_id"]), set(case_folders))
        rows.append({
            "patient_id": entry["patient_id"],
            "source_json": entry["source_json"],
            "source_id": entry["source_id"],
            "case_folders": case_folders,
            "is_orig": is_orig,
            "label": 0 if is_orig else 1,
            "sum_all_bars_score": float(entry["sum_all_bars_score"]),
            "sum_token_surprisal_hot_px": float(entry["sum_token_surprisal_hot_px"]),
            "sum_binary_sum_heatmap": float(entry["sum_binary_sum_heatmap"]),
            "num_slices": int(entry["num_slices"]),
            "global_patient_key": f"{entry['source_id']}::{entry['patient_id']}",
        })

    return sorted(rows, key=lambda r: r["sum_all_bars_score"], reverse=True)

def merge_json_payloads_for_roc(
    input_paths: List[Path],
    allowed_case_folders: Optional[Set[str]] = None,
) -> Dict[str, object]:
    """Merge multiple inference JSON files and build per-patient scores for ROC."""
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
    merged_patient_scores: List[dict] = []
    source_summaries: List[dict] = []

    for json_path in unique_paths:
        if not json_path.exists():
            logging.warning("Skipping missing JSON: %s", json_path)
            continue

        payload = load_payload(json_path)
        results = payload.get("results", [])
        patient_summary = payload.get("patient_summary", [])

        if not isinstance(results, list):
            logging.warning("Skipping malformed 'results' list in %s", json_path)
            results = []
        if not isinstance(patient_summary, list):
            logging.warning("Skipping malformed 'patient_summary' list in %s", json_path)
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

        file_patient_scores = aggregate_patient_sum_of_all_bars(
            results,
            source_json=json_path,
            allowed_case_folders=allowed_case_folders,
        )
        merged_patient_scores.extend(file_patient_scores)

        source_summaries.append({
            "source_json": str(json_path),
            "source_id": json_path.parent.name,
            "num_results": len(results),
            "num_patient_summary": len(patient_summary),
            "num_patient_scores": len(file_patient_scores),
        })

    output = {
        "timestamp": datetime.now().isoformat(),
        "input_files": [str(p) for p in unique_paths],
        "source_summaries": source_summaries,
        "summary": {
            "num_input_files": len(unique_paths),
            "num_merged_results": len(merged_results),
            "num_merged_patient_summary": len(merged_patient_summary),
            "num_merged_patient_scores": len(merged_patient_scores),
        },
        "merged_patient_scores_sum_all_bars": sorted(
            merged_patient_scores, key=lambda r: r["sum_all_bars_score"], reverse=True
        ),
        "merged_patient_summary": merged_patient_summary,
        "merged_results": merged_results,
    }
    return output

def compute_auc_trapezoid(fpr: List[float], tpr: List[float]) -> float:
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have same length")
    if len(fpr) < 2:
        return 0.0
    pairs = sorted(zip(fpr, tpr), key=lambda p: (p[0], p[1]))
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

def compute_pr_pairs(y_true: List[int], scores: List[float]) -> List[Tuple[float, float]]:
    """Compute precision-recall step pairs as (recall, precision)."""
    positives = sum(y_true)
    if positives == 0:
        raise ValueError("PR pair computation requires at least one positive sample")

    unique_scores = sorted(set(scores), reverse=True)
    thresholds = [inf] + unique_scores + [-inf]

    best_precision_by_recall: Dict[float, float] = {}
    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        recall = min(max(float(recall), 0.0), 1.0)
        precision = min(max(float(precision), 0.0), 1.0)
        prev = best_precision_by_recall.get(recall)
        if prev is None or precision > prev:
            best_precision_by_recall[recall] = precision

    recall_values = sorted(best_precision_by_recall.keys())
    precision_values = [best_precision_by_recall[r] for r in recall_values]

    if recall_values[0] > 0.0:
        recall_values.insert(0, 0.0)
        precision_values.insert(0, 1.0)
    else:
        precision_values[0] = max(precision_values[0], 1.0)

    if recall_values[-1] < 1.0:
        recall_values.append(1.0)
        precision_values.append(precision_values[-1])

    # Precision envelope keeps the PR curve non-increasing with recall.
    for idx in range(len(precision_values) - 2, -1, -1):
        precision_values[idx] = max(precision_values[idx], precision_values[idx + 1])

    return list(zip(recall_values, precision_values))

def compute_auprc_step(pr_pairs: List[Tuple[float, float]]) -> float:
    """Compute AUPRC using right-step integration over recall."""
    if len(pr_pairs) < 2:
        return 0.0

    pairs = sorted(pr_pairs, key=lambda item: (item[0], item[1]))
    area = 0.0
    prev_recall = float(pairs[0][0])
    for recall, precision in pairs[1:]:
        recall_f = float(recall)
        precision_f = float(precision)
        if recall_f < prev_recall:
            continue
        area += (recall_f - prev_recall) * precision_f
        prev_recall = recall_f
    return min(max(area, 0.0), 1.0)

def sample_step_precision_on_recall_grid(
    recall_values: List[float],
    precision_values: List[float],
    recall_grid: List[float],
) -> List[float]:
    sampled: List[float] = []
    for recall in recall_grid:
        idx = bisect_right(recall_values, recall) - 1
        if idx < 0:
            sampled.append(1.0)
        else:
            sampled.append(float(precision_values[idx]))
    return sampled

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

def compute_bootstrap_roc_pr_ci(
    y_true: List[int],
    scores: List[float],
    n_bootstrap_samples: int,
    confidence_level: float,
    random_seed: Optional[int],
    fpr_grid_size: int,
    recall_grid_size: Optional[int] = None,
) -> Dict[str, object]:
    """Compute AUROC/AUPRC CIs and curve bands with the same stratified bootstrap draws."""
    if n_bootstrap_samples <= 1:
        raise ValueError("n_bootstrap_samples must be > 1")
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if fpr_grid_size < 2:
        raise ValueError("fpr_grid_size must be >= 2")

    if recall_grid_size is None:
        recall_grid_size = fpr_grid_size
    if recall_grid_size < 2:
        raise ValueError("recall_grid_size must be >= 2")

    pos_indices = [idx for idx, label in enumerate(y_true) if int(label) == 1]
    neg_indices = [idx for idx, label in enumerate(y_true) if int(label) == 0]
    if not pos_indices or not neg_indices:
        raise ValueError("Bootstrap CI requires both classes")

    rng = random.Random(random_seed)
    fpr_grid = [idx / (fpr_grid_size - 1) for idx in range(fpr_grid_size)]
    recall_grid = [idx / (recall_grid_size - 1) for idx in range(recall_grid_size)]

    auc_samples: List[float] = []
    auprc_samples: List[float] = []
    tpr_samples_per_grid: List[List[float]] = [[] for _ in fpr_grid]
    precision_samples_per_grid: List[List[float]] = [[] for _ in recall_grid]

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

        pr_pairs = compute_pr_pairs(sampled_y_true, sampled_scores)
        auprc_samples.append(compute_auprc_step(pr_pairs))
        step_recall = [float(pair[0]) for pair in pr_pairs]
        step_precision = [float(pair[1]) for pair in pr_pairs]
        precision_on_grid = sample_step_precision_on_recall_grid(
            step_recall,
            step_precision,
            recall_grid,
        )
        for grid_idx, precision_value in enumerate(precision_on_grid):
            precision_samples_per_grid[grid_idx].append(float(precision_value))

    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q

    def summarize_samples(samples: List[float]) -> Tuple[float, float, float, float]:
        sorted_samples = sorted(samples)
        mean = sum(samples) / len(samples)
        std = (
            (sum((sample - mean) ** 2 for sample in samples) / (len(samples) - 1)) ** 0.5
            if len(samples) > 1
            else 0.0
        )
        ci_lower = percentile_from_sorted_values(sorted_samples, lower_q)
        ci_upper = percentile_from_sorted_values(sorted_samples, upper_q)
        return float(mean), float(std), float(ci_lower), float(ci_upper)

    auc_mean, auc_std, auc_ci_lower, auc_ci_upper = summarize_samples(auc_samples)
    auprc_mean, auprc_std, auprc_ci_lower, auprc_ci_upper = summarize_samples(auprc_samples)

    tpr_ci_lower: List[float] = []
    tpr_ci_upper: List[float] = []
    tpr_median: List[float] = []
    for values in tpr_samples_per_grid:
        sorted_values = sorted(values)
        tpr_ci_lower.append(percentile_from_sorted_values(sorted_values, lower_q))
        tpr_ci_upper.append(percentile_from_sorted_values(sorted_values, upper_q))
        tpr_median.append(percentile_from_sorted_values(sorted_values, 0.5))

    precision_ci_lower: List[float] = []
    precision_ci_upper: List[float] = []
    precision_median: List[float] = []
    for values in precision_samples_per_grid:
        sorted_values = sorted(values)
        precision_ci_lower.append(percentile_from_sorted_values(sorted_values, lower_q))
        precision_ci_upper.append(percentile_from_sorted_values(sorted_values, upper_q))
        precision_median.append(percentile_from_sorted_values(sorted_values, 0.5))

    return {
        "method": "stratified_bootstrap",
        "n_bootstrap_samples": int(n_bootstrap_samples),
        "confidence_level": float(confidence_level),
        "random_seed": random_seed,
        "fpr_grid": fpr_grid,
        "tpr_ci_lower": tpr_ci_lower,
        "tpr_ci_upper": tpr_ci_upper,
        "tpr_median": tpr_median,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "recall_grid": recall_grid,
        "precision_ci_lower": precision_ci_lower,
        "precision_ci_upper": precision_ci_upper,
        "precision_median": precision_median,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
        "auprc_ci_lower": auprc_ci_lower,
        "auprc_ci_upper": auprc_ci_upper,
    }

def compute_patient_roc_and_auc(
    patient_scores: List[dict],
    expected_orig_cases: Optional[int] = None,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_random_seed: Optional[int] = 42,
    ci_fpr_grid_size: int = 201,
) -> Dict[str, object]:
    """Compute patient-level ROC-AUC and AUPRC for sum_all_bars_score."""
    if not patient_scores:
        raise ValueError("patient_scores is empty; cannot compute ROC")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be >= 0")
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if ci_fpr_grid_size < 2:
        raise ValueError("ci_fpr_grid_size must be >= 2")

    y_true = [int(row.get("label", 1)) for row in patient_scores]
    scores = [float(row.get("sum_all_bars_score", 0.0)) for row in patient_scores]

    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "ROC requires both classes. Found anomalies=%d, orig=%d" % (positives, negatives)
        )

    unique_scores = sorted(set(scores), reverse=True)
    thresholds = [inf] + unique_scores + [-inf]
    roc_points: List[dict] = []

    best_idx = 0
    best_youden = float("-inf")
    for idx, threshold in enumerate(thresholds):
        y_pred = [1 if s > threshold else 0 for s in scores]

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        specificity = tn / negatives if negatives else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        youden_j = tpr - fpr

        roc_points.append({
            "threshold": threshold_json_value(threshold),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": tpr,
            "tpr": tpr,
            "fpr": fpr,
            "specificity": specificity,
            "precision": precision,
            "youden_j": youden_j,
        })

        if youden_j > best_youden:
            best_youden = youden_j
            best_idx = idx

    fpr_values = [float(p["fpr"]) for p in roc_points]
    tpr_values = [float(p["tpr"]) for p in roc_points]
    auc_value = compute_auc_trapezoid(fpr_values, tpr_values)

    pr_pairs = compute_pr_pairs(y_true, scores)
    auprc_value = compute_auprc_step(pr_pairs)
    pr_points = [{"recall": float(r), "precision": float(p)} for r, p in pr_pairs]
    pr_baseline_precision = positives / len(y_true) if y_true else 0.0

    bootstrap_ci: Optional[Dict[str, object]] = None
    auc_ci_lower: Optional[float] = None
    auc_ci_upper: Optional[float] = None
    auc_std: Optional[float] = None
    auprc_ci_lower: Optional[float] = None
    auprc_ci_upper: Optional[float] = None
    auprc_std: Optional[float] = None
    if bootstrap_samples > 1:
        bootstrap_ci = compute_bootstrap_roc_pr_ci(
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
        auprc_ci_lower = float(bootstrap_ci.get("auprc_ci_lower", 0.0))
        auprc_ci_upper = float(bootstrap_ci.get("auprc_ci_upper", 0.0))
        auprc_std = float(bootstrap_ci.get("auprc_std", 0.0))

    orig_count = negatives
    anomaly_count = positives
    expected_orig_match = None
    if expected_orig_cases is not None:
        expected_orig_match = (orig_count == expected_orig_cases)

    best_point = dict(roc_points[best_idx])
    metrics = {
        "summary": {
            "num_patients": len(patient_scores),
            "num_anomalies": anomaly_count,
            "num_orig": orig_count,
            "expected_orig_cases": expected_orig_cases,
            "expected_orig_match": expected_orig_match,
            "auc": auc_value,
            "auprc": auprc_value,
            "pr_baseline_precision": pr_baseline_precision,
            "auc_ci_lower": auc_ci_lower,
            "auc_ci_upper": auc_ci_upper,
            "auc_std": auc_std,
            "auprc_ci_lower": auprc_ci_lower,
            "auprc_ci_upper": auprc_ci_upper,
            "auprc_std": auprc_std,
            "roc_confidence_level": confidence_level if bootstrap_ci else None,
            "pr_confidence_level": confidence_level if bootstrap_ci else None,
            "roc_bootstrap_samples": int(bootstrap_samples) if bootstrap_ci else 0,
            "pr_bootstrap_samples": int(bootstrap_samples) if bootstrap_ci else 0,
        },
        "best_threshold_by_youden_j": best_point,
        "roc_points": roc_points,
        "pr_points": pr_points,
        "roc_bootstrap_ci": bootstrap_ci,
        "pr_bootstrap_ci": bootstrap_ci,
    }
    return metrics

def select_threshold_for_target_fpr(roc_metrics: Dict[str, object], target_fpr: float) -> Dict[str, object]:
    """Select threshold with best sensitivity while keeping FPR <= target_fpr."""
    if target_fpr < 0.0 or target_fpr > 1.0:
        raise ValueError("target_fpr must be in [0, 1]")

    roc_points = roc_metrics.get("roc_points", [])
    if not isinstance(roc_points, list) or not roc_points:
        raise ValueError("ROC metrics are missing roc_points")

    finite_points: List[Dict[str, object]] = []
    for point in roc_points:
        try:
            threshold = parse_threshold_json(point.get("threshold", "inf"))
        except (TypeError, ValueError):
            continue
        if isinf(threshold):
            continue
        finite_points.append(point)

    if not finite_points:
        raise ValueError("No finite ROC thresholds available")

    constrained = [p for p in finite_points if float(p.get("fpr", 1.0)) <= target_fpr]
    candidate_points = constrained if constrained else finite_points

    # Prioritize higher sensitivity under target FPR, then lower FPR and higher precision.
    best = max(
        candidate_points,
        key=lambda p: (
            float(p.get("sensitivity", 0.0)),
            -float(p.get("fpr", 1.0)),
            float(p.get("precision", 0.0)),
            float(p.get("specificity", 0.0)),
        ),
    )
    return dict(best)

def plot_roc_curve_from_metrics(
    metrics: Dict[str, object],
    output_path: Path,
    show_best_threshold_marker: bool = False,
    title: Optional[str] = None,
    curve_color: str = "#e7a46e",
    ci_band_color: Optional[str] = None,
    random_line_label: str = "Random",
) -> Path:
    """Create ROC curve plot from metrics payload generated by compute_patient_roc_and_auc."""
    roc_points = metrics.get("roc_points", [])
    if not isinstance(roc_points, list) or not roc_points:
        raise ValueError("metrics.roc_points is empty; cannot plot ROC")

    pairs = sorted(
        ((float(p["fpr"]), float(p["tpr"])) for p in roc_points),
        key=lambda item: (item[0], item[1]),
    )
    step_fpr, step_tpr = build_step_curve_from_pairs(pairs)

    summary = metrics.get("summary", {})
    auc_value = float(summary.get("auc", 0.0))
    auprc_value = float(summary.get("auprc", 0.0))
    auc_ci_lower = summary.get("auc_ci_lower")
    auc_ci_upper = summary.get("auc_ci_upper")
    confidence_level = float(summary.get("roc_confidence_level", 0.95) or 0.95)
    confidence_pct = int(round(100.0 * confidence_level))
    ci_color = ci_band_color if ci_band_color else curve_color

    if auc_ci_lower is not None and auc_ci_upper is not None:
        roc_label = (
            f"ROC (AUC={auc_value:.2f}, AUPRC={auprc_value:.2f}, "
            f"{confidence_pct}% CI={float(auc_ci_lower):.2f}-{float(auc_ci_upper):.2f})"
        )
    else:
        roc_label = f"ROC (AUC={auc_value:.2f}, AUPRC={auprc_value:.2f})"

    fig, ax = plt.subplots(figsize=(7, 6))

    bootstrap_ci = metrics.get("roc_bootstrap_ci", {})
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
                color=ci_color,
                alpha=0.25,
                linewidth=0.0,
                label=f"{confidence_pct}% ROC CI band",
            )

    ax.step(
        step_fpr,
        step_tpr,
        where="post",
        linewidth=2.2,
        color=curve_color,
        label=roc_label,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label=random_line_label)

    best_point = metrics.get("best_threshold_by_youden_j", {})
    if show_best_threshold_marker and best_point:
        best_fpr = float(best_point.get("fpr", 0.0))
        best_tpr = float(best_point.get("tpr", 0.0))
        thr = best_point.get("threshold")
        if isinstance(thr, str):
            thr_str = thr
        else:
            thr_str = format_threshold_value(float(thr))
        ax.scatter([best_fpr], [best_tpr], color="red", s=40, zorder=3, label=f"Best J threshold={thr_str}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(title or "Patient-level ROC: sum_of_all_bars score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def plot_precision_recall_curve_from_metrics(
    metrics: Dict[str, object],
    output_path: Path,
    title: Optional[str] = None,
    curve_color: str = "#2a9d8f",
    ci_band_color: Optional[str] = None,
    baseline_color: str = "gray",
    baseline_label: str = "No-skill",
) -> Path:
    """Create precision-recall curve plot from metrics payload generated by compute_patient_roc_and_auc."""
    pr_points = metrics.get("pr_points", [])
    if not isinstance(pr_points, list) or not pr_points:
        raise ValueError("metrics.pr_points is empty; cannot plot PR curve")

    pairs = sorted(
        ((float(p["recall"]), float(p["precision"])) for p in pr_points),
        key=lambda item: (item[0], item[1]),
    )
    recalls = [p[0] for p in pairs]
    precisions = [p[1] for p in pairs]

    summary = metrics.get("summary", {})
    auprc_value = float(summary.get("auprc", 0.0))
    auprc_ci_lower = summary.get("auprc_ci_lower")
    auprc_ci_upper = summary.get("auprc_ci_upper")
    confidence_level = float(summary.get("pr_confidence_level", 0.95) or 0.95)
    confidence_pct = int(round(100.0 * confidence_level))
    baseline_precision = float(summary.get("pr_baseline_precision", 0.0) or 0.0)
    ci_color = ci_band_color if ci_band_color else curve_color

    if auprc_ci_lower is not None and auprc_ci_upper is not None:
        pr_label = (
            f"PR (AUPRC={auprc_value:.2f}, "
            f"{confidence_pct}% CI={float(auprc_ci_lower):.2f}-{float(auprc_ci_upper):.2f})"
        )
    else:
        pr_label = f"PR (AUPRC={auprc_value:.2f})"

    fig, ax = plt.subplots(figsize=(7, 6))

    bootstrap_ci = metrics.get("pr_bootstrap_ci", {})
    if isinstance(bootstrap_ci, dict):
        ci_recall = bootstrap_ci.get("recall_grid")
        ci_low = bootstrap_ci.get("precision_ci_lower")
        ci_high = bootstrap_ci.get("precision_ci_upper")
        if (
            isinstance(ci_recall, list)
            and isinstance(ci_low, list)
            and isinstance(ci_high, list)
            and len(ci_recall) == len(ci_low) == len(ci_high)
            and len(ci_recall) > 1
        ):
            ax.fill_between(
                ci_recall,
                ci_low,
                ci_high,
                step="post",
                color=ci_color,
                alpha=0.25,
                linewidth=0.0,
                label=f"{confidence_pct}% PR CI band",
            )

    ax.step(
        recalls,
        precisions,
        where="post",
        linewidth=2.2,
        color=curve_color,
        label=pr_label,
    )
    ax.plot(
        [0, 1],
        [baseline_precision, baseline_precision],
        linestyle="--",
        color=baseline_color,
        linewidth=1.0,
        label=f"{baseline_label} (P={baseline_precision:.2f})",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Patient-level Precision-Recall: sum_of_all_bars score")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def plot_multi_roc_curves_from_metrics(
    curve_specs: List[Dict[str, object]],
    output_path: Path,
    title: Optional[str] = None,
    random_line_label: str = "Random",
) -> Path:
    """Create one ROC plot containing multiple stratified curves and CI bands."""
    if not curve_specs:
        raise ValueError("curve_specs is empty; cannot plot combined ROC")

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted_count = 0

    for spec in curve_specs:
        metrics = spec.get("metrics")
        if not isinstance(metrics, dict):
            continue

        name = str(spec.get("name") or "ROC")
        curve_color = str(spec.get("curve_color") or "#e7a46e")
        ci_band_color = str(spec.get("ci_band_color") or curve_color)
        ci_alpha = float(spec.get("ci_alpha", 0.22))

        roc_points = metrics.get("roc_points", [])
        if not isinstance(roc_points, list) or not roc_points:
            continue

        pairs = sorted(
            ((float(p["fpr"]), float(p["tpr"])) for p in roc_points),
            key=lambda item: (item[0], item[1]),
        )
        step_fpr, step_tpr = build_step_curve_from_pairs(pairs)

        summary = metrics.get("summary", {})
        auc_value = float(summary.get("auc", 0.0))
        auprc_value = float(summary.get("auprc", 0.0))
        auc_ci_lower = summary.get("auc_ci_lower")
        auc_ci_upper = summary.get("auc_ci_upper")
        confidence_level = float(summary.get("roc_confidence_level", 0.95) or 0.95)
        confidence_pct = int(round(100.0 * confidence_level))

        if auc_ci_lower is not None and auc_ci_upper is not None:
            roc_label = (
                f"{name} ROC (AUC={auc_value:.2f}, AUPRC={auprc_value:.2f}, "
                f"{confidence_pct}% CI={float(auc_ci_lower):.2f}-{float(auc_ci_upper):.2f})"
            )
        else:
            roc_label = f"{name} ROC (AUC={auc_value:.2f}, AUPRC={auprc_value:.2f})"

        bootstrap_ci = metrics.get("roc_bootstrap_ci", {})
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
                    color=ci_band_color,
                    alpha=ci_alpha,
                    linewidth=0.0,
                    label=f"{name} {confidence_pct}% ROC CI band",
                )

        ax.step(
            step_fpr,
            step_tpr,
            where="post",
            linewidth=2.2,
            color=curve_color,
            label=roc_label,
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        raise ValueError("No valid ROC curves available to plot in combined figure")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label=random_line_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(title or "Patient-level ROC: Synthetic global vs Clinical sum_of_all_bars score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def plot_multi_pr_curves_from_metrics(
    curve_specs: List[Dict[str, object]],
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Create one precision-recall plot containing multiple stratified curves."""
    if not curve_specs:
        raise ValueError("curve_specs is empty; cannot plot combined PR")

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted_count = 0

    for spec in curve_specs:
        metrics = spec.get("metrics")
        if not isinstance(metrics, dict):
            continue

        name = str(spec.get("name") or "PR")
        curve_color = str(spec.get("curve_color") or "#2a9d8f")
        ci_band_color = str(spec.get("ci_band_color") or curve_color)
        ci_alpha = float(spec.get("ci_alpha", 0.22))

        pr_points = metrics.get("pr_points", [])
        if not isinstance(pr_points, list) or not pr_points:
            continue

        pairs = sorted(
            ((float(p["recall"]), float(p["precision"])) for p in pr_points),
            key=lambda item: (item[0], item[1]),
        )
        recalls = [p[0] for p in pairs]
        precisions = [p[1] for p in pairs]

        summary = metrics.get("summary", {})
        auprc_value = float(summary.get("auprc", 0.0))
        auprc_ci_lower = summary.get("auprc_ci_lower")
        auprc_ci_upper = summary.get("auprc_ci_upper")
        confidence_level = float(summary.get("pr_confidence_level", 0.95) or 0.95)
        confidence_pct = int(round(100.0 * confidence_level))
        if auprc_ci_lower is not None and auprc_ci_upper is not None:
            pr_label = (
                f"{name} PR (AUPRC={auprc_value:.2f}, "
                f"{confidence_pct}% CI={float(auprc_ci_lower):.2f}-{float(auprc_ci_upper):.2f})"
            )
        else:
            pr_label = f"{name} PR (AUPRC={auprc_value:.2f})"

        bootstrap_ci = metrics.get("pr_bootstrap_ci", {})
        if isinstance(bootstrap_ci, dict):
            ci_recall = bootstrap_ci.get("recall_grid")
            ci_low = bootstrap_ci.get("precision_ci_lower")
            ci_high = bootstrap_ci.get("precision_ci_upper")
            if (
                isinstance(ci_recall, list)
                and isinstance(ci_low, list)
                and isinstance(ci_high, list)
                and len(ci_recall) == len(ci_low) == len(ci_high)
                and len(ci_recall) > 1
            ):
                ax.fill_between(
                    ci_recall,
                    ci_low,
                    ci_high,
                    step="post",
                    color=ci_band_color,
                    alpha=ci_alpha,
                    linewidth=0.0,
                    label=f"{name} {confidence_pct}% PR CI band",
                )

        ax.step(
            recalls,
            precisions,
            where="post",
            linewidth=2.2,
            color=curve_color,
            label=pr_label,
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        raise ValueError("No valid PR curves available to plot in combined figure")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Patient-level PR: Synthetic global vs Clinical sum_of_all_bars score")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def build_patient_category_map(merged_results: List[dict]) -> Dict[str, str]:
    """Map each patient key to dominant category from slice-level merged results."""
    category_votes: Dict[str, Counter] = defaultdict(Counter)
    for item in merged_results:
        key = patient_key_from_merged_result(item)
        category = item.get("category")
        if category is None or str(category).strip() == "":
            case_folder = str(item.get("case_folder") or "").strip()
            category = case_folder if case_folder else "Unknown"
        category_votes[key][str(category)] += 1

    dominant: Dict[str, str] = {}
    for key, votes in category_votes.items():
        dominant[key] = votes.most_common(1)[0][0]
    return dominant

def build_focus_category_tokens(focus_categories: Optional[Set[str]]) -> Optional[List[Tuple[str, str]]]:
    if not focus_categories:
        return None
    canonical_priority = {
        name.lower(): idx for idx, (name, _token) in enumerate(CANONICAL_CATEGORY_RULES)
    }
    # Follow the canonical order so specific synthetic subtypes such as
    # RandomGhosting win before broad fallback buckets such as SyntheticVariations.
    return sorted(
        ((name, name.lower()) for name in focus_categories),
        key=lambda pair: (canonical_priority.get(pair[1], len(canonical_priority)), -len(pair[1]), pair[1]),
    )

def match_focus_category(category: str, focus_tokens: Optional[List[Tuple[str, str]]]) -> Optional[str]:
    if focus_tokens is None:
        return category
    category_l = category.lower()
    return next((name for name, token in focus_tokens if token in category_l), None)

def canonicalize_anomaly_category(
    patient_id: str,
    category: str,
    case_folders: Optional[List[str]] = None,
) -> str:
    """Group semantically identical anomaly categories into a single canonical label."""
    text_parts: List[str] = [patient_id, category]
    if case_folders:
        text_parts.extend(str(cf) for cf in case_folders)
    haystack = " ".join(part for part in text_parts if part).lower()
    for canonical_name, token in CANONICAL_CATEGORY_RULES:
        if token in haystack:
            return canonical_name
    return category if category else "Unknown"

def filter_patient_scores_for_focus_categories(
    patient_scores: List[dict],
    merged_results: List[dict],
    focus_categories: Optional[Set[str]],
) -> Tuple[List[dict], Dict[str, int]]:
    """Return orig + focus-anomaly patient scores for a focus-only ROC.

    Includes all normal/orig patients and only anomaly patients whose dominant
    category matches one of the focus category tokens (substring, case-insensitive).
    """
    patient_category_map = build_patient_category_map(merged_results)
    focus_tokens = build_focus_category_tokens(focus_categories)

    filtered: List[dict] = []
    total_orig = 0
    total_anomaly = 0
    included_anomaly = 0

    for row in patient_scores:
        label = int(row.get("label", 1))
        if label == 0:
            filtered.append(dict(row))
            total_orig += 1
            continue

        total_anomaly += 1
        # Prefer matching directly from patient_id because it is the most stable
        # identifier even when category/case_folder naming varies.
        patient_id = str(row.get("patient_id") or "")
        matched_focus = match_focus_category(patient_id, focus_tokens) if patient_id else None
        if matched_focus is None:
            key = str(row.get("global_patient_key") or "")
            category = patient_category_map.get(key)
            if category is None:
                case_folders = row.get("case_folders") or []
                category = str(case_folders[0]) if case_folders else "Unknown"
            category = str(category)
            matched_focus = match_focus_category(category, focus_tokens)
        if matched_focus is None:
            continue

        updated = dict(row)
        updated["matched_focus_category"] = matched_focus
        filtered.append(updated)
        included_anomaly += 1

    summary = {
        "total_orig_included": total_orig,
        "total_anomaly_input": total_anomaly,
        "anomaly_included": included_anomaly,
        "anomaly_excluded": total_anomaly - included_anomaly,
    }
    return filtered, summary

def pairwise_auc_vs_orig(anomaly_scores: List[float], orig_scores: List[float]) -> float:
    """Compute one-vs-orig AUC using pairwise ranking probability."""
    total = len(anomaly_scores) * len(orig_scores)
    if total == 0:
        return 0.0
    wins = 0.0
    for a_score in anomaly_scores:
        for o_score in orig_scores:
            if a_score > o_score:
                wins += 1.0
            elif a_score == o_score:
                wins += 0.5
    return wins / total

def pairwise_auprc_vs_orig(anomaly_scores: List[float], orig_scores: List[float]) -> float:
    """Compute one-vs-orig AUPRC with anomaly as positive class."""
    if not anomaly_scores or not orig_scores:
        return 0.0
    y_true = [1] * len(anomaly_scores) + [0] * len(orig_scores)
    scores = [float(s) for s in anomaly_scores] + [float(s) for s in orig_scores]
    pr_pairs = compute_pr_pairs(y_true, scores)
    return compute_auprc_step(pr_pairs)

def compute_category_stratified_performance(
    patient_scores: List[dict],
    merged_results: List[dict],
    threshold: float,
    focus_categories: Optional[Set[str]] = None,
) -> Dict[str, object]:
    """Compute per-category detection quality at a fixed threshold."""
    if not patient_scores:
        raise ValueError("patient_scores is empty; cannot compute category stratified performance")

    patient_category_map = build_patient_category_map(merged_results)

    orig_scores = [float(row.get("sum_all_bars_score", 0.0)) for row in patient_scores if int(row.get("label", 1)) == 0]
    if not orig_scores:
        raise ValueError("No orig patients found in patient_scores")

    orig_fp = sum(1 for score in orig_scores if score > threshold)
    orig_tn = len(orig_scores) - orig_fp
    fpr = orig_fp / len(orig_scores)
    specificity = orig_tn / len(orig_scores)

    focus_tokens = build_focus_category_tokens(focus_categories)

    category_scores: Dict[str, List[float]] = defaultdict(list)
    for row in patient_scores:
        if int(row.get("label", 1)) != 1:
            continue

        # Prefer category resolution from patient_id when available; this avoids
        # under-counting when category/case_folder strings vary across sources.
        patient_id = str(row.get("patient_id") or "")
        category_from_pid = patient_id if patient_id else ""
        case_folders = [str(cf) for cf in (row.get("case_folders") or [])]

        key = str(row.get("global_patient_key") or "")
        category = patient_category_map.get(key)
        if category is None:
            category = case_folders[0] if case_folders else "Unknown"
        category = str(category)

        if focus_tokens is not None:
            matched_focus = None
            if category_from_pid:
                matched_focus = match_focus_category(category_from_pid, focus_tokens)
            if matched_focus is None:
                matched_focus = match_focus_category(category, focus_tokens)
            if matched_focus is None:
                continue
            # Canonicalize into the configured focus label for grouped reporting.
            category = matched_focus
        else:
            category = canonicalize_anomaly_category(
                patient_id=category_from_pid,
                category=category,
                case_folders=case_folders,
            )

        category_scores[category].append(float(row.get("sum_all_bars_score", 0.0)))

    if not category_scores:
        raise ValueError("No anomaly patients matched selected focus categories")

    rows: List[Dict[str, object]] = []
    for category, scores in category_scores.items():
        num_patients = len(scores)
        tp = sum(1 for score in scores if score > threshold)
        fn = num_patients - tp
        sensitivity = tp / num_patients if num_patients else 0.0
        auc_vs_orig = pairwise_auc_vs_orig(scores, orig_scores)
        auprc_vs_orig = pairwise_auprc_vs_orig(scores, orig_scores)
        precision = tp / (tp + orig_fp) if (tp + orig_fp) > 0 else 0.0

        rows.append({
            "category": category,
            "num_patients": num_patients,
            "threshold": threshold,
            "tp_detected": tp,
            "fn_missed": fn,
            "sensitivity": sensitivity,
            "auc_vs_orig": auc_vs_orig,
            "auprc_vs_orig": auprc_vs_orig,
            "mean_score": sum(scores) / num_patients,
            "min_score": min(scores),
            "max_score": max(scores),
            "precision_vs_orig_at_threshold": precision,
            "orig_count": len(orig_scores),
            "orig_fp_at_threshold": orig_fp,
            "fpr_at_threshold": fpr,
            "specificity_at_threshold": specificity,
        })

    rows.sort(
        key=lambda row: (
            -float(row["sensitivity"]),
            -float(row["auprc_vs_orig"]),
            -float(row["auc_vs_orig"]),
            -int(row["num_patients"]),
            str(row["category"]),
        )
    )
    return {
        "threshold": threshold,
        "focus_categories": sorted(focus_categories) if focus_categories else None,
        "orig_count": len(orig_scores),
        "orig_fp_at_threshold": orig_fp,
        "fpr_at_threshold": fpr,
        "specificity_at_threshold": specificity,
        "num_categories": len(rows),
        "rows": rows,
    }

def write_category_stratified_csv(rows: List[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "num_patients",
        "threshold",
        "tp_detected",
        "fn_missed",
        "sensitivity",
        "auc_vs_orig",
        "auprc_vs_orig",
        "mean_score",
        "min_score",
        "max_score",
        "precision_vs_orig_at_threshold",
        "orig_count",
        "orig_fp_at_threshold",
        "fpr_at_threshold",
        "specificity_at_threshold",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path

def sensitivity_color(value: float) -> str:
    if value >= 0.9:
        return "#d8f3dc"
    if value >= 0.75:
        return "#e9f5db"
    if value >= 0.6:
        return "#fff3bf"
    if value >= 0.4:
        return "#ffe8cc"
    return "#ffd6d6"

def plot_category_stratified_table_figure(payload: Dict[str, object], output_path: Path) -> Path:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No category rows available for table figure")

    threshold = float(payload.get("threshold", 0.0))
    fpr = float(payload.get("fpr_at_threshold", 0.0))
    target_fpr = payload.get("target_fpr")

    headers = [
        "Category",
        "N",
        "TP/FN",
        "Sensitivity",
        "AUC vs orig",
        "AUPRC vs orig",
        "Mean score",
        "FPR",
        "Threshold",
    ]
    table_data: List[List[str]] = []
    for row in rows:
        tp = int(row["tp_detected"])
        fn = int(row["fn_missed"])
        n = int(row["num_patients"])
        sens = float(row["sensitivity"])
        auc_v = float(row["auc_vs_orig"])
        auprc_v = float(row.get("auprc_vs_orig", 0.0))
        mean_score = float(row["mean_score"])
        row_fpr = float(row["fpr_at_threshold"])
        row_thr = float(row["threshold"])
        table_data.append([
            str(row["category"]),
            str(n),
            f"{tp}/{fn}",
            f"{sens * 100:.1f}%",
            f"{auc_v:.3f}",
            f"{auprc_v:.3f}",
            f"{mean_score:.1f}",
            f"{row_fpr * 100:.1f}%",
            f"{row_thr:.1f}",
        ])

    fig_height = max(4.0, 1.8 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    title = "Stratified Category Performance (Focused Anomalies)"
    if target_fpr is not None:
        title += f"\nSelected for target FPR <= {float(target_fpr) * 100:.1f}%"
    subtitle = f"Applied threshold={threshold:.1f}, achieved FPR={fpr * 100:.1f}%"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=10)

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.28)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#264653")
            continue
        data_row = rows[row_idx - 1]
        if col_idx == 3:
            cell.set_facecolor(sensitivity_color(float(data_row["sensitivity"])))
        elif col_idx == 4:
            cell.set_facecolor(sensitivity_color(float(data_row["auc_vs_orig"])))
        elif col_idx == 5:
            cell.set_facecolor(sensitivity_color(float(data_row.get("auprc_vs_orig", 0.0))))
        else:
            cell.set_facecolor("#f8f9fa")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path

def build_roc_threshold_table_rows(
    roc_metrics: Dict[str, object],
    include_infinite: bool = False,
) -> List[Dict[str, object]]:
    """Build rows for threshold-by-threshold ROC performance inspection."""
    roc_points = roc_metrics.get("roc_points", [])
    if not isinstance(roc_points, list) or not roc_points:
        raise ValueError("ROC metrics are missing roc_points")

    rows: List[Dict[str, object]] = []
    for point in roc_points:
        threshold_val = parse_threshold_json(point.get("threshold", "inf"))
        if not include_infinite and isinf(threshold_val):
            continue
        rows.append({
            "threshold": format_threshold_value(threshold_val),
            "threshold_value": threshold_val,
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
        raise ValueError("No threshold rows available after filtering")
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
            writer.writerow({
                "threshold": row["threshold"],
                "sensitivity": row["sensitivity"],
                "fpr": row["fpr"],
                "specificity": row["specificity"],
                "precision": row["precision"],
                "tp": row["tp"],
                "fp": row["fp"],
                "tn": row["tn"],
                "fn": row["fn"],
                "youden_j": row["youden_j"],
            })
    return output_path

def fpr_color(value: float) -> str:
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
    title_prefix: str = "ROC Threshold Performance Table",
) -> List[Path]:
    """Render threshold-level ROC stats as one or multiple table figures."""
    if rows_per_page < 1:
        raise ValueError("rows_per_page must be >= 1")
    if not rows:
        raise ValueError("rows is empty for ROC threshold table figure")

    prefix = output_prefix.with_suffix("") if output_prefix.suffix else output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    pages: List[List[Dict[str, object]]] = [
        rows[idx: idx + rows_per_page] for idx in range(0, len(rows), rows_per_page)
    ]
    total_pages = len(pages)
    output_files: List[Path] = []

    headers = [
        "Threshold",
        "Sensitivity",
        "FPR",
        "Specificity",
        "Precision",
        "TP",
        "FP",
        "TN",
        "FN",
        "Youden J",
    ]

    for page_idx, page_rows in enumerate(pages, start=1):
        fig_height = max(4.0, 1.8 + 0.36 * len(page_rows))
        fig, ax = plt.subplots(figsize=(14, fig_height))
        ax.axis("off")

        table_data: List[List[str]] = []
        for row in page_rows:
            table_data.append([
                str(row["threshold"]),
                f"{float(row['sensitivity']) * 100:.2f}%",
                f"{float(row['fpr']) * 100:.2f}%",
                f"{float(row['specificity']) * 100:.2f}%",
                f"{float(row['precision']) * 100:.2f}%",
                str(int(row["tp"])),
                str(int(row["fp"])),
                str(int(row["tn"])),
                str(int(row["fn"])),
                f"{float(row['youden_j']):.4f}",
            ])

        page_title = f"{title_prefix} (Page {page_idx}/{total_pages})"
        ax.set_title(page_title, fontsize=12, pad=10)

        table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
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
                cell.set_facecolor(sensitivity_color(float(row["sensitivity"])))
            elif c_idx == 2:
                cell.set_facecolor(fpr_color(float(row["fpr"])))
            else:
                cell.set_facecolor("#f8f9fa")

        fig.tight_layout()
        if total_pages == 1:
            out_path = prefix.with_suffix(".png")
        else:
            out_path = prefix.parent / f"{prefix.name}_page_{page_idx:02d}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        output_files.append(out_path)

    return output_files

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot perceptual bar charts from a results JSON.")
    default_json = "/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_test_LUND_PROBE_extended_npy/results_v4_zscore.json"   
    
    
    parser.add_argument("--input", type=Path, default=default_json, help="Path to results_v4.json")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory to save plots")
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
    
    parser.add_argument(
        "--run-merged-roc",
        action="store_true",
        help="Merge multiple JSON files and compute patient-level ROC-AUC + AUPRC for sum_of_all_bars score.",
    )
    parser.add_argument(
        "--roc-input",
        dest="roc_inputs",
        action="append",
        type=Path,
        help="Path to results_v4_zscore.json for merged ROC mode. Repeat for multiple files.",
    )
    parser.add_argument(
        "--merged-json-output",
        type=Path,
        default=None,
        help="Output path for merged JSON used for ROC computation.",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=None,
        help="Output path for ROC curve image.",
    )
    parser.add_argument(
        "--pr-output",
        type=Path,
        default=None,
        help="Output path for Precision-Recall curve image.",
    )
    parser.add_argument(
        "--show-best-j-marker",
        action="store_true",
        help="Show the Best-Youden threshold marker on the ROC plot (default: hidden for cleaner figure).",
    )
    parser.add_argument(
        "--roc-metrics-output",
        type=Path,
        default=None,
        help="Output path for ROC/PR metrics JSON (includes AUC, AUPRC, ROC points, and PR points).",
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
        help="Output prefix for ROC threshold table figure pages.",
    )
    parser.add_argument(
        "--roc-table-rows-per-figure",
        type=int,
        default=40,
        help="Number of threshold rows per ROC table figure page.",
    )
    parser.add_argument(
        "--include-infinite-threshold-rows",
        action="store_true",
        help="Include inf and -inf sentinel thresholds in ROC threshold table outputs.",
    )
    parser.add_argument(
        "--expected-orig-cases",
        type=int,
        default=49,
        help="Expected number of normal/orig patients for sanity check in merged ROC mode.",
    )
    parser.add_argument(
        "--fpr-target",
        type=float,
        default=0.20,
        help="Target max false positive rate when selecting threshold for stratified category table (0.20 = 20%%).",
    )
    parser.add_argument(
        "--focus-category",
        dest="focus_categories",
        action="append",
        help=(
            "Category to include in stratified table figure. Repeat for multiple categories. "
            "Default is all canonical categories from CANONICAL_CATEGORY_RULES."
        ),
    )
    parser.add_argument(
        "--category-table-csv-output",
        type=Path,
        default=None,
        help="Output CSV for stratified category performance table.",
    )
    parser.add_argument(
        "--category-table-json-output",
        type=Path,
        default=None,
        help="Output JSON for stratified category performance table.",
    )
    parser.add_argument(
        "--category-table-figure-output",
        type=Path,
        default=None,
        help="Output PNG figure for stratified category performance table.",
    )

    # AVAILABLE YET NOT USED parser arguments

    parser.add_argument("--first-heatmap-threshold", type=float, default=10000.0, help="AYNU: Threshold for total first-heatmap pixel sum to color patient bars (default: 500)")

    parser.add_argument("--threshold-lpips-in-rec", type=float, default=5000.0, help="AYNU: Threshold for per-patient lpips_input_recon_sum_mask plot")

    parser.add_argument("--clamp-sum-threshold", type=float, default=140.0, help="AYNU: Threshold for total clamped pixel sum to mark patient as anomaly (default: 450)")

    parser.add_argument("--binary-heatmap-threshold", type=float, default=700.0, help="AYNU: Threshold for per-patient Binary_Sum_Heatmap plot")

    parser.add_argument("--sharpness-threshold", type=float, default=5.0, help="AYNU: Threshold for per-patient total sharpness plot (anomaly if below)")

    parser.add_argument("--sharpness-low-threshold", type=float, default=7.0, help="AYNU: Stage-1 sharpness lower bound (anomaly if below)")

    parser.add_argument("--sharpness-high-threshold", type=float, default=20.0, help="AYNU: Stage-1 sharpness upper bound (anomaly if above)")

    parser.add_argument(
        "--combined-threshold",
        type=float,
        default=150.0,
        help="AYNU: Threshold for combined token_surprisal_hot_px + Binary_Sum_Heatmap (used for red bars)",
    )

    parser.add_argument(
        "--min-red-bars-per-patient",
        type=int,
        default=0,
        help="AYNU: Patient is counted as anomaly when red-bar count is strictly greater than this value (default: 2)",
    )

    parser.add_argument(
        "--sum-all-bars-threshold",
        "--sum-of-all-bars-threshold",
        dest="sum_all_bars_threshold",
        type=float,
        default=80.0,
        help="AYNU: Threshold for per-patient sum of all combined bars (token_surprisal_hot_px + Binary_Sum_Heatmap)",
    )

    parser.add_argument(
        "--run-aynu-diagnostics",
        action="store_true",
        help="AYNU: produce auxiliary per-patient bar charts (combined token+binary, "
             "unique anomaly counter, sum-of-all-bars, mask-score sum). "
             "Not used for AUROC reproduction.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading results from %s", args.input)
    payload = load_payload(args.input)
    results = payload.get("results", [])
    patient_summary = payload.get("patient_summary", [])
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
            results = filter_items_by_category(results, allowed_categories)
            patient_summary = filter_items_by_category(patient_summary, allowed_categories)
            logging.info(
                "After category filtering: %d slices, %d patient-summary entries",
                len(results),
                len(patient_summary),
            )

    if args.run_merged_roc:
        try:
            roc_inputs = args.roc_inputs if args.roc_inputs else list(DEFAULT_ROC_INPUT_PATHS)
            logging.info("Running merged ROC mode with %d input files", len(roc_inputs))

            merged_payload = merge_json_payloads_for_roc(
                roc_inputs,
                allowed_case_folders=allowed_case_folders,
            )

            merged_json_output = args.merged_json_output or (args.output_dir / "Merged_AllInvestigatedAnomalies_results_v4_zscore.json")
            write_json_file(merged_payload, merged_json_output)
            logging.info("Wrote merged JSON for ROC -> %s", merged_json_output)

            patient_scores = merged_payload.get("merged_patient_scores_sum_all_bars", [])
            if not isinstance(patient_scores, list) or not patient_scores:
                raise ValueError("Merged patient score list is empty; ROC cannot be computed")

            roc_metrics = compute_patient_roc_and_auc(
                patient_scores,
                expected_orig_cases=int(args.expected_orig_cases),
                bootstrap_samples=max(0, int(args.roc_ci_bootstrap_samples)),
                confidence_level=float(args.roc_ci_confidence_level),
                bootstrap_random_seed=(None if int(args.roc_ci_random_seed) < 0 else int(args.roc_ci_random_seed)),
                ci_fpr_grid_size=max(2, int(args.roc_ci_fpr_grid_size)),
            )

            target_fpr = float(args.fpr_target)
            selected_for_target_fpr = select_threshold_for_target_fpr(roc_metrics, target_fpr)
            selected_threshold_value = parse_threshold_json(selected_for_target_fpr.get("threshold", "inf"))
            roc_metrics["recommended_threshold_for_target_fpr"] = {
                "target_fpr": target_fpr,
                **selected_for_target_fpr,
            }

            # Also keep a strict 2% FPR reference to avoid confusion between 0.20 and 0.02.
            two_pct_reference = select_threshold_for_target_fpr(roc_metrics, 0.02)
            roc_metrics["reference_threshold_for_2pct_fpr"] = {
                "target_fpr": 0.02,
                **two_pct_reference,
            }

            roc_metrics_output = args.roc_metrics_output or (args.output_dir / "ROC_sum_of_all_bars_metrics.json")
            write_json_file(roc_metrics, roc_metrics_output)

            roc_output = args.roc_output or (args.output_dir / "ROC_sum_of_all_bars.png")
            plot_roc_curve_from_metrics(
                roc_metrics,
                roc_output,
                show_best_threshold_marker=bool(args.show_best_j_marker),
            )

            pr_output = args.pr_output or (args.output_dir / "PR_sum_of_all_bars.png")
            plot_precision_recall_curve_from_metrics(
                roc_metrics,
                pr_output,
                title="Patient-level Precision-Recall: sum_of_all_bars score",
            )

            roc_threshold_rows = build_roc_threshold_table_rows(
                roc_metrics,
                include_infinite=bool(args.include_infinite_threshold_rows),
            )
            roc_threshold_table_csv_output = args.roc_threshold_table_csv_output or (args.output_dir / "ROC_Threshold_Table.csv")
            write_roc_threshold_table_csv(roc_threshold_rows, roc_threshold_table_csv_output)

            roc_threshold_table_figure_prefix = args.roc_threshold_table_figure_prefix or (args.output_dir / "ROC_Threshold_Table")
            roc_threshold_table_figures = plot_roc_threshold_table_figures(
                roc_threshold_rows,
                roc_threshold_table_figure_prefix,
                rows_per_page=max(1, int(args.roc_table_rows_per_figure)),
            )

            summary = roc_metrics.get("summary", {})
            if summary.get("auc_ci_lower") is not None and summary.get("auc_ci_upper") is not None:
                logging.info(
                    "ROC/PR complete: AUROC=%.4f (%d%% CI %.4f-%.4f), AUPRC=%.4f (%d%% CI %.4f-%.4f), anomalies=%s, orig=%s",
                    float(summary.get("auc", 0.0)),
                    int(round(float(summary.get("roc_confidence_level", 0.95)) * 100.0)),
                    float(summary.get("auc_ci_lower", 0.0)),
                    float(summary.get("auc_ci_upper", 0.0)),
                    float(summary.get("auprc", 0.0)),
                    int(round(float(summary.get("pr_confidence_level", 0.95)) * 100.0)),
                    float(summary.get("auprc_ci_lower", 0.0)),
                    float(summary.get("auprc_ci_upper", 0.0)),
                    summary.get("num_anomalies"),
                    summary.get("num_orig"),
                )
            else:
                logging.info(
                    "ROC complete: AUC=%.4f, AUPRC=%.4f, anomalies=%s, orig=%s",
                    float(summary.get("auc", 0.0)),
                    float(summary.get("auprc", 0.0)),
                    summary.get("num_anomalies"),
                    summary.get("num_orig"),
                )
            logging.info(
                "Threshold for target FPR<=%.1f%%: %s (actual FPR=%.2f%%, sensitivity=%.2f%%)",
                target_fpr * 100.0,
                format_threshold_value(selected_threshold_value),
                float(selected_for_target_fpr.get("fpr", 0.0)) * 100.0,
                float(selected_for_target_fpr.get("sensitivity", 0.0)) * 100.0,
            )
            logging.info(
                "Reference threshold for FPR<=2%%: %s (actual FPR=%.2f%%, sensitivity=%.2f%%)",
                format_threshold_value(parse_threshold_json(two_pct_reference.get("threshold", "inf"))),
                float(two_pct_reference.get("fpr", 0.0)) * 100.0,
                float(two_pct_reference.get("sensitivity", 0.0)) * 100.0,
            )
            if float(selected_for_target_fpr.get("fpr", 1.0)) > target_fpr:
                logging.warning(
                    "No threshold met target FPR<=%.2f%% exactly; selected best available point with FPR=%.2f%%",
                    target_fpr * 100.0,
                    float(selected_for_target_fpr.get("fpr", 0.0)) * 100.0,
                )

            default_focus_categories = [name for name, _ in CANONICAL_CATEGORY_RULES]
            focus_categories = {
                c.strip() for c in (args.focus_categories if args.focus_categories else default_focus_categories)
                if c and c.strip()
            }
            logging.info(
                "Category-table focus categories: %s",
                ", ".join(sorted(focus_categories)),
            )
            category_payload = compute_category_stratified_performance(
                patient_scores,
                merged_payload.get("merged_results", []),
                threshold=selected_threshold_value,
                focus_categories=focus_categories,
            )
            category_payload["target_fpr"] = target_fpr
            category_payload["selected_threshold_point"] = selected_for_target_fpr
            category_payload["reference_threshold_for_2pct_fpr"] = two_pct_reference

            category_table_json_output = args.category_table_json_output or (args.output_dir / "Category_Stratified_Performance_Table.json")
            write_json_file(category_payload, category_table_json_output)

            category_table_csv_output = args.category_table_csv_output or (args.output_dir / "Category_Stratified_Performance_Table.csv")
            write_category_stratified_csv(category_payload.get("rows", []), category_table_csv_output)

            category_table_figure_output = args.category_table_figure_output or (args.output_dir / "Category_Stratified_Performance_Table.png")
            plot_category_stratified_table_figure(category_payload, category_table_figure_output)

            # Build split ROC curves for focused anomaly groups while retaining all orig/normal patients.
            split_focus_groups: List[Dict[str, object]] = [
                {
                    "slug": "Synthetic_global",
                    "display_name": "Synthetic global",
                    "focus_categories": set(SYNTHETIC_GLOBAL_FOCUS_CATEGORIES),
                    "curve_color": "black",
                    "ci_band_color": "black",
                },
                {
                    "slug": "Clinical",
                    "display_name": "Clinical",
                    "focus_categories": set(CLINICAL_FOCUS_CATEGORIES),
                    "curve_color": "#e7a46e",
                    "ci_band_color": "#e7a46e",
                },
            ]
            combined_focus_curve_specs: List[Dict[str, object]] = []

            merged_results = merged_payload.get("merged_results", [])
            for group in split_focus_groups:
                group_name = str(group["display_name"])
                group_slug = str(group["slug"])
                group_focus_categories = set(group.get("focus_categories", set()))
                group_curve_color = str(group["curve_color"])
                group_ci_color = str(group["ci_band_color"])

                focus_patient_scores, focus_filter_summary = filter_patient_scores_for_focus_categories(
                    patient_scores,
                    merged_results,
                    focus_categories=group_focus_categories,
                )
                if not focus_patient_scores:
                    logging.warning("Skipping %s ROC: no patients matched selected focus categories", group_name)
                    continue

                focus_roc_metrics = compute_patient_roc_and_auc(
                    focus_patient_scores,
                    expected_orig_cases=int(args.expected_orig_cases),
                    bootstrap_samples=max(0, int(args.roc_ci_bootstrap_samples)),
                    confidence_level=float(args.roc_ci_confidence_level),
                    bootstrap_random_seed=(
                        None if int(args.roc_ci_random_seed) < 0 else int(args.roc_ci_random_seed)
                    ),
                    ci_fpr_grid_size=max(2, int(args.roc_ci_fpr_grid_size)),
                )
                focus_selected_for_target_fpr = select_threshold_for_target_fpr(focus_roc_metrics, target_fpr)
                focus_selected_threshold_value = parse_threshold_json(focus_selected_for_target_fpr.get("threshold", "inf"))
                focus_two_pct_reference = select_threshold_for_target_fpr(focus_roc_metrics, 0.02)

                focus_roc_metrics["focus_filter"] = {
                    **focus_filter_summary,
                    "focus_categories": sorted(group_focus_categories),
                    "description": (
                        "Includes all orig patients and anomaly patients whose dominant category "
                        "matches focus tokens (case-insensitive substring)."
                    ),
                    "split_group": group_name,
                }
                focus_roc_metrics["recommended_threshold_for_target_fpr"] = {
                    "target_fpr": target_fpr,
                    **focus_selected_for_target_fpr,
                }
                focus_roc_metrics["reference_threshold_for_2pct_fpr"] = {
                    "target_fpr": 0.02,
                    **focus_two_pct_reference,
                }

                focus_roc_metrics_output = args.output_dir / f"ROC_sum_of_all_bars_{group_slug}_metrics.json"
                write_json_file(focus_roc_metrics, focus_roc_metrics_output)

                focus_pr_output = args.output_dir / f"PR_sum_of_all_bars_{group_slug}.png"
                plot_precision_recall_curve_from_metrics(
                    focus_roc_metrics,
                    focus_pr_output,
                    title=f"Patient-level Precision-Recall: {group_name} sum_of_all_bars score",
                    curve_color=group_curve_color,
                )

                combined_focus_curve_specs.append({
                    "name": group_name,
                    "metrics": focus_roc_metrics,
                    "curve_color": group_curve_color,
                    "ci_band_color": group_ci_color,
                    "ci_alpha": 0.18 if group_curve_color.lower() == "black" else 0.22,
                })

                focus_roc_threshold_rows = build_roc_threshold_table_rows(
                    focus_roc_metrics,
                    include_infinite=bool(args.include_infinite_threshold_rows),
                )
                focus_roc_threshold_csv_output = args.output_dir / f"ROC_Threshold_Table_{group_slug}.csv"
                write_roc_threshold_table_csv(focus_roc_threshold_rows, focus_roc_threshold_csv_output)

                focus_roc_threshold_figure_prefix = args.output_dir / f"ROC_Threshold_Table_{group_slug}"
                focus_roc_threshold_figures = plot_roc_threshold_table_figures(
                    focus_roc_threshold_rows,
                    focus_roc_threshold_figure_prefix,
                    rows_per_page=max(1, int(args.roc_table_rows_per_figure)),
                    title_prefix=f"{group_name} ROC Threshold Performance Table",
                )

                focus_summary = focus_roc_metrics.get("summary", {})
                if focus_summary.get("auc_ci_lower") is not None and focus_summary.get("auc_ci_upper") is not None:
                    logging.info(
                        "%s ROC/PR complete: AUROC=%.4f (%d%% CI %.4f-%.4f), AUPRC=%.4f (%d%% CI %.4f-%.4f), anomalies=%s, orig=%s",
                        group_name,
                        float(focus_summary.get("auc", 0.0)),
                        int(round(float(focus_summary.get("roc_confidence_level", 0.95)) * 100.0)),
                        float(focus_summary.get("auc_ci_lower", 0.0)),
                        float(focus_summary.get("auc_ci_upper", 0.0)),
                        float(focus_summary.get("auprc", 0.0)),
                        int(round(float(focus_summary.get("pr_confidence_level", 0.95)) * 100.0)),
                        float(focus_summary.get("auprc_ci_lower", 0.0)),
                        float(focus_summary.get("auprc_ci_upper", 0.0)),
                        focus_summary.get("num_anomalies"),
                        focus_summary.get("num_orig"),
                    )
                else:
                    logging.info(
                        "%s ROC complete: AUC=%.4f, AUPRC=%.4f, anomalies=%s, orig=%s",
                        group_name,
                        float(focus_summary.get("auc", 0.0)),
                        float(focus_summary.get("auprc", 0.0)),
                        focus_summary.get("num_anomalies"),
                        focus_summary.get("num_orig"),
                    )
                logging.info(
                    "%s threshold for target FPR<=%.1f%%: %s (actual FPR=%.2f%%, sensitivity=%.2f%%)",
                    group_name,
                    target_fpr * 100.0,
                    format_threshold_value(focus_selected_threshold_value),
                    float(focus_selected_for_target_fpr.get("fpr", 0.0)) * 100.0,
                    float(focus_selected_for_target_fpr.get("sensitivity", 0.0)) * 100.0,
                )
                logging.info("%s ROC metrics JSON -> %s", group_name, focus_roc_metrics_output)
                logging.info("%s PR curve PNG -> %s", group_name, focus_pr_output)
                logging.info("%s ROC threshold table CSV -> %s", group_name, focus_roc_threshold_csv_output)
                logging.info(
                    "%s ROC threshold table figure pages generated: %d (prefix: %s)",
                    group_name,
                    len(focus_roc_threshold_figures),
                    focus_roc_threshold_figure_prefix,
                )

            if combined_focus_curve_specs:
                combined_focus_roc_output = args.output_dir / "ROC_sum_of_all_bars_Synthetic_vs_Clinical.png"
                plot_multi_roc_curves_from_metrics(
                    combined_focus_curve_specs,
                    combined_focus_roc_output,
                    title="Patient-level ROC: Synthetic global vs Clinical sum_of_all_bars score",
                    random_line_label="Random",
                )
                combined_focus_pr_output = args.output_dir / "PR_sum_of_all_bars_Synthetic_vs_Clinical.png"
                plot_multi_pr_curves_from_metrics(
                    combined_focus_curve_specs,
                    combined_focus_pr_output,
                    title="Patient-level PR: Synthetic global vs Clinical sum_of_all_bars score",
                )
                logging.info("Synthetic vs Clinical combined ROC curve PNG -> %s", combined_focus_roc_output)
                logging.info("Synthetic vs Clinical combined PR curve PNG -> %s", combined_focus_pr_output)
            else:
                logging.warning("Skipping Synthetic-vs-Clinical combined ROC/PR plots: no split curves were generated")

            if summary.get("expected_orig_match") is False:
                logging.warning(
                    "Expected orig count %s but found %s in merged ROC set",
                    summary.get("expected_orig_cases"),
                    summary.get("num_orig"),
                )
            logging.info("ROC metrics JSON -> %s", roc_metrics_output)
            logging.info("ROC curve PNG -> %s", roc_output)
            logging.info("PR curve PNG -> %s", pr_output)
            logging.info("ROC threshold table CSV -> %s", roc_threshold_table_csv_output)
            logging.info(
                "ROC threshold table figure pages generated: %d (prefix: %s)",
                len(roc_threshold_table_figures),
                roc_threshold_table_figure_prefix,
            )
            logging.info("Category stratified table JSON -> %s", category_table_json_output)
            logging.info("Category stratified table CSV -> %s", category_table_csv_output)
            logging.info("Category stratified table figure -> %s", category_table_figure_output)
        except ValueError as exc:
            logging.warning("Skipping merged ROC mode: %s", exc)

    # AYNU: optional auxiliary diagnostics, not used for AUROC reproduction
    if getattr(args, "run_aynu_diagnostics", False):
        try:
            if not results:
                raise ValueError("No slices remain after applying filters (category/case-folder).")
            combined_threshold = float(args.combined_threshold)
            logging.info("Using combined threshold: %.3f", combined_threshold)

            output_path = args.output_dir / "TokenSurprisalPlusBinarySum_per_sample.png"
            plot_combined_token_binary_per_sample(
                results,
                output_path,
                top_n=args.top_n,
                #threshold=350.0, #350 FastMRI-specific threshold for combined token surprisal and binary sum to flag anomalies; adjust as needed
                threshold=combined_threshold,
                allowed_case_folders=allowed_case_folders,
            )
            logging.info(
                "Plotted combined token_surprisal_hot_px + Binary_Sum_Heatmap per sample -> %s",
                output_path,
            )

            unique_counter_path = args.output_dir / "Uniqe_anomaly_patients_counter.png"
            plot_unique_anomaly_patients_counter(
                results,
                unique_counter_path,
                threshold=combined_threshold,
                min_red_bars=args.min_red_bars_per_patient,
                top_n=args.top_n,
                allowed_case_folders=allowed_case_folders,
            )
            logging.info(
                "Plotted unique anomaly patient counter (> %d red bars) -> %s",
                args.min_red_bars_per_patient,
                unique_counter_path,
            )

            patient_sum_output_path = args.output_dir / "Uniqe_patients_sum_of_all_bars.png"
            plot_unique_patients_sum_of_all_bars(
                results,
                patient_sum_output_path,
                threshold=float(args.sum_all_bars_threshold),
                top_n=args.top_n,
                allowed_case_folders=allowed_case_folders,
            )
            logging.info(
                "Plotted unique patients sum-of-all-bars figure (threshold=%.3f) -> %s",
                float(args.sum_all_bars_threshold),
                patient_sum_output_path,
            )

            unique_mask_scores_path = args.output_dir / "Unique_patients_sum_of_mask_scores.png"
            plot_unique_patients_sum_of_mask_scores(
                results,
                unique_mask_scores_path,
                threshold=5000.0,
                top_n=args.top_n,
                allowed_case_folders=allowed_case_folders,
            )
            logging.info(
                "Plotted unique patients sum-of-mask-scores figure (threshold=5000.0) -> %s",
                unique_mask_scores_path,
            )
        except ValueError as exc:
            logging.warning("Skipping AYNU diagnostics: %s", exc)
    return

# =============================================================================
# AVAILABLE YET NOT USED
# -----------------------------------------------------------------------------
# The items below are not on the AUROC reproduction path. They are kept for
# transparency and for readers interested in alternative analyses, training,
# calibration generation, or visualisation. None of them affects the
# patient-level AUROC reported in the manuscript.
# =============================================================================

DEFAULT_FOCUS_CATEGORIES: Tuple[str, ...] = (
    "RandomGhosting",
    "RandomNoise",
    "RandomSpike",
    "RandomMotion",
    "WholeImageGaussian",
    "Stor_T2_till_sCT",
    "ClinicalVariations",
    "Unknown",
    "Spacer",

)

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

def patient_display_label(p: dict) -> str:
    """Choose a display label for a patient summary entry.

    Prefers case_folder when informative; otherwise derives from first slice filename.
    For filenames like Unknown_Unknown_5_Stor_T2_till_sCT-motion_slice_039.npy, returns
    the portion after leading "Unknown_" prefixes and before "_slice_", e.g., "5_Stor_T2_till_sCT-motion".
    """
    case_folder = str(p.get("case_folder", ""))
    if case_folder and case_folder.lower() != "unknown":
        return case_folder

    slice_details = p.get("slice_details", []) or []
    if slice_details:
        fname = slice_details[0].get("filename") or ""
        base = fname.split("_slice_", 1)[0] if "_slice_" in fname else fname.rsplit(".", 1)[0]
        # Strip leading "unknown_" segments iteratively
        lowered = base.lower()
        while lowered.startswith("unknown_"):
            if "_" in base:
                base = base.split("_", 1)[1]
            else:
                break
            lowered = base.lower()
        return base or case_folder or p.get("patient_id", "unknown")

    return case_folder or p.get("patient_id", "unknown")

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
                stem = Path(fname).stem
                if "_slice_" in stem:
                    stem = stem.split("_slice_", 1)[0]
                return strip_unknown_prefix(stem)
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
        if "_slice_" not in name:
            return None
        try:
            return int(name.split("_slice_", 1)[1].split(".", 1)[0])
        except (ValueError, IndexError):
            return None

    def extract_patient_base(filename: str) -> str:
        """Extract base patient identifier before _slice_."""
        if "_slice_" in filename:
            base = filename.split("_slice_", 1)[0]
        else:
            base = filename.rsplit(".", 1)[0] if "." in filename else filename
        return strip_unknown_prefix(base)

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
        if "_slice_" not in name:
            return None
        try:
            return int(name.split("_slice_", 1)[1].split(".", 1)[0])
        except (ValueError, IndexError):
            return None

    entries: List[Tuple[str, float, bool]] = []  # (label, sum, is_orig)
    for p in patient_summary:
        case_folder = p.get("case_folder")
        if not matches_case_folder(case_folder, allowed_case_folders):
            continue
        slice_details = p.get("slice_details", []) or []
        if not slice_details:
            continue
        base_label = slice_details[0].get("filename", "unknown")
        if "_slice_" in base_label:
            base_label = base_label.split("_slice_", 1)[0]
        base_label = strip_unknown_prefix(base_label)

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
        if "_slice_" not in name:
            return None
        try:
            return int(name.split("_slice_", 1)[1].split(".", 1)[0])
        except (ValueError, IndexError):
            return None

    def extract_patient_base(filename: str) -> str:
        """Extract base patient identifier before _slice_."""
        if "_slice_" in filename:
            base = filename.split("_slice_", 1)[0]
        else:
            base = filename.rsplit(".", 1)[0] if "." in filename else filename
        return strip_unknown_prefix(base)

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

if __name__ == "__main__":
    main()
