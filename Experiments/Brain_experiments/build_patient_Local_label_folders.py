#!/usr/bin/env python3
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_LABELS = [
    "Edema",
    "Enlarged ventricles",
    "Craniotomy",
    "Mass",
    "Nonspecific lesion",
    "Resection cavity",
    "Intraventricular substance",
    "Paranasal sinus opacification",
    "Posttreatment change",
    "Nonspecific white matter lesion",
    "Encephalomalacia",
    "Dural thickening",
    "Absent septum pellucidum",
    "Lacunar infarct",
    "Likely cysts",
]

# Maps detailed label -> anomalies CSV filename stem
DEFAULT_LABEL_TO_CSV = {
    "Edema": "Edema.csv",
    "Enlarged ventricles": "Enlarged_Vents.csv",
    "Craniotomy": "Craniotomy.csv",
    "Mass": "Mass.csv",
    "Nonspecific lesion": "Lesions.csv",
    "Resection cavity": "Resections.csv",
    "Intraventricular substance": "Intra_ventrical.csv",
    "Paranasal sinus opacification": "Sinus.csv",
    "Posttreatment change": "PostTreat_changes.csv",
    "Nonspecific white matter lesion": "WhiteMatter.csv",
    "Encephalomalacia": "Encephalo.csv",
    "Dural thickening": "Dural_thickining.csv",
    "Absent septum pellucidum": "Absent_septum.csv",
    "Lacunar infarct": "Lacunar_infarct.csv",
    "Likely cysts": "Likely_cysts.csv",
}


def normalize_patient_id(raw: str) -> str:
    return raw.strip()


def read_patient_list(csv_path: Path) -> Set[str]:
    patients: Set[str] = set()
    with csv_path.open("r", newline="") as f:
        for line in f:
            value = line.strip()
            if not value:
                continue
            if value.lower() in {"file", "patient", "id"}:
                continue
            patients.add(normalize_patient_id(value))
    return patients


def read_detailed_labels(detailed_csv: Path) -> Dict[str, Set[str]]:
    label_to_patients: Dict[str, Set[str]] = {}
    with detailed_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = normalize_patient_id(row.get("file", ""))
            label = (row.get("label") or "").strip()
            if not patient or not label:
                continue
            label_to_patients.setdefault(label, set()).add(patient)
    return label_to_patients


def read_detailed_labels_with_slices(
    detailed_csv: Path,
) -> Dict[str, Dict[str, Set[int]]]:
    label_patient_slices: Dict[str, Dict[str, Set[int]]] = {}
    with detailed_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = normalize_patient_id(row.get("file", ""))
            label = (row.get("label") or "").strip()
            slice_raw = (row.get("slice") or "").strip()
            if not patient or not label or slice_raw == "":
                continue
            try:
                slice_idx = int(slice_raw)
            except ValueError:
                continue
            label_patient_slices.setdefault(label, {}).setdefault(patient, set()).add(slice_idx)
    return label_patient_slices


def sanitize_label_for_folder(label: str) -> str:
    return label.replace("/", "-").replace(" ", "_")


def ensure_patient_folders(
    output_dir: Path,
    label: str,
    patients: Iterable[str],
) -> None:
    label_dir = output_dir / sanitize_label_for_folder(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    for patient in sorted(patients):
        (label_dir / patient).mkdir(parents=True, exist_ok=True)


def write_label_patients_csv(label_dir: Path, patients: Iterable[str]) -> None:
    csv_path = label_dir / "patients.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id"])
        for patient in sorted(patients):
            writer.writerow([patient])


def write_anomaly_csv_list(csv_path: Path, patients: Iterable[str]) -> None:
    with csv_path.open("w", newline="") as f:
        for patient in sorted(patients):
            f.write(f"{patient}\n")


def write_patient_slices_csv(patient_dir: Path, slices: Iterable[int]) -> None:
    csv_path = patient_dir / "slices.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice"])
        for slice_idx in sorted(slices):
            writer.writerow([slice_idx])


def write_summary_csv(
    output_dir: Path,
    rows: List[Tuple[str, int, int, int, int, int, int]],
) -> None:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "patients_in_anomaly_csv",
                "patients_in_detailed_csv",
                "patients_in_intersection",
                "patients_in_union",
                "missing_in_detailed",
                "missing_in_anomaly_csv",
            ]
        )
        for row in rows:
            writer.writerow(list(row))


def log_missing_patients(label: str, missing: Set[str], reason: str) -> None:
    if not missing:
        return
    sample = ", ".join(sorted(missing))
    logging.info(
        "Label '%s' %s (%d): %s",
        label,
        reason,
        len(missing),
        sample,
    )


def parse_labels(labels_arg: str) -> List[str]:
    if not labels_arg:
        return list(DEFAULT_LABELS)
    return [label.strip() for label in labels_arg.split(",") if label.strip()]


def build_label_csv_map(anomalies_dir: Path) -> Dict[str, Path]:
    return {path.name: path for path in anomalies_dir.glob("*.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare anomaly CSV patient lists to the detailed brain CSV and create label/patient folders for the intersection.")
    parser.add_argument("--anomalies-dir", default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection", help="Directory containing anomaly CSV files.")
    parser.add_argument("--detailed-csv", default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Annotated_fastMRI_Brains_Detailed.csv", help="Detailed brain CSV with per-slice labels.")
    parser.add_argument("--output-dir", default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_Anomalies_ByLabel", help="Output directory for label/patient folders.")
    parser.add_argument("--labels", default=", ".join(DEFAULT_LABELS), help="Comma-separated list of label names to process.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    anomalies_dir = Path(args.anomalies_dir)
    detailed_csv = Path(args.detailed_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not anomalies_dir.exists():
        raise FileNotFoundError(f"Anomalies dir not found: {anomalies_dir}")
    if not detailed_csv.exists():
        raise FileNotFoundError(f"Detailed CSV not found: {detailed_csv}")

    labels = parse_labels(args.labels)
    logging.info("Processing labels: %s", ", ".join(labels))

    csv_files = build_label_csv_map(anomalies_dir)
    detailed_label_patients = read_detailed_labels(detailed_csv)
    detailed_label_patient_slices = read_detailed_labels_with_slices(detailed_csv)

    summary_rows: List[Tuple[str, int, int, int, int, int, int]] = []

    for label in labels:
        csv_name = DEFAULT_LABEL_TO_CSV.get(label)
        if not csv_name:
            logging.warning("No default CSV mapping for label: %s", label)
            continue
        csv_path = csv_files.get(csv_name)
        if not csv_path:
            logging.warning("CSV file not found for label %s (expected %s)", label, csv_name)
            continue

        patients_from_csv = read_patient_list(csv_path)
        patients_from_detailed = detailed_label_patients.get(label, set())

        intersection = patients_from_csv & patients_from_detailed
        union_patients = patients_from_csv | patients_from_detailed
        missing_in_detailed = patients_from_csv - patients_from_detailed
        missing_in_anomaly = patients_from_detailed - patients_from_csv

        if missing_in_anomaly:
            write_anomaly_csv_list(csv_path, union_patients)
            logging.info(
                "Updated anomaly CSV for label '%s' with %d total patients",
                label,
                len(union_patients),
            )

        if not union_patients:
            logging.warning("No patients found for label: %s", label)
        else:
            ensure_patient_folders(output_dir, label, union_patients)
            label_dir = output_dir / sanitize_label_for_folder(label)
            write_label_patients_csv(label_dir, union_patients)

            label_slices = detailed_label_patient_slices.get(label, {})
            for patient in union_patients:
                slices = label_slices.get(patient, set())
                patient_dir = label_dir / patient
                if slices:
                    write_patient_slices_csv(patient_dir, slices)

        summary_rows.append(
            (
                label,
                len(patients_from_csv),
                len(patients_from_detailed),
                len(intersection),
                len(union_patients),
                len(missing_in_detailed),
                len(missing_in_anomaly),
            )
        )

        if missing_in_detailed:
            log_missing_patients(
                label,
                missing_in_detailed,
                "missing in detailed CSV",
            )
        if missing_in_anomaly:
            log_missing_patients(
                label,
                missing_in_anomaly,
                "missing in anomaly CSV",
            )

    write_summary_csv(output_dir, summary_rows)
    logging.info("Done. Output written to %s", output_dir)


if __name__ == "__main__":
    main()
