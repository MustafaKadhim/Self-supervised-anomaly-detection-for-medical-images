#!/usr/bin/env python3
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_LABELS = [
    "Motion artifact",
    "Possible artifact",
    "Colpocephaly",
    "Extra-axial collection",
    "Global label: Small vessel chronic white matter ischemic change",
]

# Maps detailed label -> anomalies CSV filename stem
DEFAULT_LABEL_TO_CSV = {
    "Motion artifact": "All_GlobalAnomalies_xxx.csv",
    "Possible artifact": "All_GlobalAnomalies_xxx.csv",
    "Colpocephaly": "All_GlobalAnomalies_xxx.csv",
    "Extra-axial collection": "All_GlobalAnomalies_xxx.csv",
    "Global label: Small vessel chronic white matter ischemic change": "All_GlobalAnomalies_xxx.csv",
}


def normalize_patient_id(raw: str) -> str:
    return raw.strip()


def normalize_label(raw: str) -> str:
    return raw.strip().casefold()


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


def read_anomaly_labels(csv_path: Path, label_column: str | None = None) -> tuple[Dict[str, Set[str]], bool]:
    """Read a global anomaly CSV and return (label -> patients, has_label_column).

    If no label column exists, all patients are returned under an empty label key.
    """
    label_to_patients: Dict[str, Set[str]] = {}
    with csv_path.open("r", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "," in sample
        if not has_header:
            label_to_patients[""] = read_patient_list(csv_path)
            return label_to_patients, False

        reader = csv.DictReader(f)
        fieldnames = [name for name in (reader.fieldnames or []) if name]
        fieldnames_lc = {name.lower() for name in fieldnames}
        label_key = None
        if label_column:
            if label_column.lower() in fieldnames_lc:
                for name in fieldnames:
                    if name.lower() == label_column.lower():
                        label_key = name
                        break
            else:
                raise ValueError(f"Label column '{label_column}' not found in {csv_path}")
        else:
            for candidate in ("label", "labels", "finding", "anomaly", "anomaly_label", "class"):
                if candidate in fieldnames_lc:
                    for name in fieldnames:
                        if name.lower() == candidate:
                            label_key = name
                            break
                    if label_key:
                        break

        if not label_key:
            label_to_patients[""] = read_patient_list(csv_path)
            return label_to_patients, False

        for row in reader:
            patient = normalize_patient_id(
                (row.get("file") or row.get("File") or row.get("patient") or row.get("Patient") or "").strip()
            )
            label = normalize_label(row.get(label_key) or "")
            if not patient:
                continue
            if not label:
                continue
            label_to_patients.setdefault(label, set()).add(patient)

    return label_to_patients, True


def read_detailed_labels(detailed_csv: Path) -> Dict[str, Set[str]]:
    label_to_patients: Dict[str, Set[str]] = {}
    with detailed_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = normalize_patient_id(row.get("file", ""))
            label = normalize_label(row.get("label") or "")
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
            label = normalize_label(row.get("label") or "")
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
    parser.add_argument("--output-dir", default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_Global_Anomalies_ByLabel", help="Output directory for label/patient folders.")
    parser.add_argument("--labels", default=", ".join(DEFAULT_LABELS), help="Comma-separated list of label names to process.")
    parser.add_argument("--use-detailed", action="store_true", help="Use detailed CSV to compare labels and write per-slice CSVs.")
    parser.add_argument("--create-patient-folders", action="store_true", help="Create per-patient subfolders even without slices.csv.")
    parser.add_argument("--label-column", default=None, help="Column name in the global CSV that contains labels.")
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
    detailed_label_patients = read_detailed_labels(detailed_csv) if args.use_detailed else {}
    detailed_label_patient_slices = read_detailed_labels_with_slices(detailed_csv) if args.use_detailed else {}

    summary_rows: List[Tuple[str, int, int, int, int, int, int]] = []

    csv_label_maps: Dict[Path, Dict[str, Set[str]]] = {}
    csv_has_label_column: Dict[Path, bool] = {}
    for label in labels:
        csv_name = DEFAULT_LABEL_TO_CSV.get(label)
        if not csv_name:
            logging.warning("No default CSV mapping for label: %s", label)
            continue
        csv_path = csv_files.get(csv_name)
        if not csv_path:
            logging.warning("CSV file not found for label %s (expected %s)", label, csv_name)
            continue

        if csv_path not in csv_label_maps:
            label_map, has_label_column = read_anomaly_labels(csv_path, args.label_column)
            csv_label_maps[csv_path] = label_map
            csv_has_label_column[csv_path] = has_label_column

        label_key = normalize_label(label)
        label_map = csv_label_maps[csv_path]
        has_label_column = csv_has_label_column.get(csv_path, False)
        patients_from_csv = label_map.get(label_key)
        if patients_from_csv is None:
            if has_label_column:
                logging.warning(
                    "Label '%s' not found in %s; check --label-column or label names.",
                    label,
                    csv_path.name,
                )
                patients_from_csv = set()
            else:
                patients_from_csv = label_map.get("", set())
        patients_from_detailed = detailed_label_patients.get(label_key, set())
        has_detailed_label = args.use_detailed and (
            label_key in detailed_label_patients or label_key in detailed_label_patient_slices
        )

        if not has_label_column:
            if args.use_detailed:
                patients_from_csv = patients_from_csv & patients_from_detailed
            else:
                logging.warning(
                    "CSV %s has no label column and --use-detailed is off; skipping label '%s'.",
                    csv_path.name,
                    label,
                )
                patients_from_csv = set()

        intersection = patients_from_csv & patients_from_detailed
        union_patients = patients_from_csv | patients_from_detailed
        missing_in_detailed = patients_from_csv - patients_from_detailed
        missing_in_anomaly = patients_from_detailed - patients_from_csv

        if not has_detailed_label:
            missing_in_detailed = set()

        if missing_in_anomaly and not has_label_column:
            write_anomaly_csv_list(csv_path, union_patients)
            logging.info(
                "Updated anomaly CSV for label '%s' with %d total patients",
                label,
                len(union_patients),
            )

        if not union_patients:
            logging.warning("No patients found for label: %s", label)
        else:
            label_dir = output_dir / sanitize_label_for_folder(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            if args.use_detailed or args.create_patient_folders:
                ensure_patient_folders(output_dir, label, union_patients)
            write_label_patients_csv(label_dir, union_patients)

            if args.use_detailed:
                label_slices = detailed_label_patient_slices.get(label_key, {})
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
        elif not has_detailed_label and args.use_detailed:
            logging.info("Label '%s' not found in detailed CSV labels; skipping detailed comparison.", label)
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
