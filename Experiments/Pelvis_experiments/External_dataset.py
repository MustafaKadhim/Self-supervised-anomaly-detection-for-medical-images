"""
External Dataset Preprocessing for Anomaly Detection Inference
===============================================================

This script processes external .nii.gz files for inference with the 
RVQ-MaskGIT anomaly detection framework.

It applies identical preprocessing to the training pipeline:
1. Load 3D NIfTI volumes
2. Extract 2D axial slices
3. Apply z-score normalization
4. Rotate 90° CCW (matching training orientation)
5. Resize to 320x320, then center crop to 256x256
6. Save as .npy files with informative naming

Folder structure expected:
    Batch2_15pat_finalForESTRO/
    ├── ClinicalVariations/
    │   ├── MAVRIC_protes/
    │   │   └── *.nii.gz
    │   ├── Spacer/
    │   ├── T2_CUBE_FemaleBrachy/
    │   └── ...
    └── SyntheticVariations/
        ├── Noise/
        ├── Motion/
        └── ...

Output naming convention:
    {category}_{case_folder}_{volume_name}_slice_{slice_idx:03d}.npy

Example:
    ClinicalVariations_T2_CUBE_FemaleBrachy_Cube1_slice_045.npy

Author: External Dataset Processor for ESTRO Abstract 2025
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


# =============================================================================
# Preprocessing Functions (matching training pipeline)
# =============================================================================

def load_nifti(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load a NIfTI file and return the volume data and metadata.
    
    Args:
        filepath: Path to .nii or .nii.gz file
        
    Returns:
        Tuple of (volume_data, metadata_dict)
    """
    nii = nib.load(filepath)
    data = nii.get_fdata().astype(np.float32)
    
    metadata = {
        "affine": nii.affine,
        "header": dict(nii.header),
        "shape": data.shape,
        "filepath": filepath,
    }
    
    return data, metadata


def normalize_slice(slice_2d: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize a 2D slice.
    
    Args:
        slice_2d: 2D numpy array
        method: "zscore" (mean=0, std=1) or "minmax" (0 to 1)
        
    Returns:
        Normalized slice
    """
    if method == "zscore":
        mean = slice_2d.mean()
        std = slice_2d.std()
        if std < 1e-8:
            return slice_2d - mean
        return (slice_2d - mean) / std
    elif method == "minmax":
        min_val = slice_2d.min()
        max_val = slice_2d.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(slice_2d)
        return (slice_2d - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_and_crop(
    slice_2d: np.ndarray,
    resize_size: Tuple[int, int] = (320, 320),
    crop_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Resize slice to intermediate size, then center crop to final size.
    Matches the training pipeline: Resize(320) -> CenterCrop(256).
    
    Args:
        slice_2d: Input 2D slice
        resize_size: Intermediate resize dimensions
        crop_size: Final crop dimensions
        
    Returns:
        Processed slice
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors for resize
    h, w = slice_2d.shape
    zoom_h = resize_size[0] / h
    zoom_w = resize_size[1] / w
    
    # Resize using area-like interpolation (order=1 for bilinear, similar to "area")
    resized = zoom(slice_2d, (zoom_h, zoom_w), order=1)
    
    # Center crop
    h_resized, w_resized = resized.shape
    h_crop, w_crop = crop_size
    
    start_h = (h_resized - h_crop) // 2
    start_w = (w_resized - w_crop) // 2
    
    cropped = resized[start_h:start_h + h_crop, start_w:start_w + w_crop]
    
    return cropped


def preprocess_slice(
    slice_2d: np.ndarray,
    resize_size: Tuple[int, int] = (320, 320),
    crop_size: Tuple[int, int] = (256, 256),
    normalize: str = "zscore",
    rotate_ccw: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single slice.
    Matches training: normalize -> rotate 90° CCW -> resize -> crop
    
    Args:
        slice_2d: Input 2D slice
        resize_size: Intermediate resize dimensions
        crop_size: Final crop dimensions
        normalize: Normalization method
        rotate_ccw: Whether to rotate 90° counter-clockwise
        
    Returns:
        Preprocessed slice ready for inference
    """
    # 1. Normalize
    processed = normalize_slice(slice_2d, method=normalize)
    
    # 2. Rotate 90° CCW (k=-1 means clockwise, k=1 means CCW)
    # In training: np.rot90(arr, k=-1) which is 90° CW
    # Let me check the training code again... it says "Rotate 90° CCW" but uses k=-1
    # k=-1 is actually 90° clockwise. Let's match exactly what training does.
    if rotate_ccw:
        processed = np.rot90(processed, k=-1).copy()
    
    # 3. Resize and crop
    processed = resize_and_crop(processed, resize_size, crop_size)
    
    return processed.astype(np.float32)


# =============================================================================
# Volume Processing
# =============================================================================

def extract_slices_from_volume(
    volume: np.ndarray,
    axis: int = 2,  # Axial slices (z-axis)
    slice_range: Optional[Tuple[int, int]] = None,
    min_nonzero_ratio: float = 0.1,
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract 2D slices from a 3D volume.
    
    Args:
        volume: 3D numpy array
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        slice_range: Optional (start, end) to limit slices
        min_nonzero_ratio: Minimum ratio of non-zero pixels to keep slice
        
    Returns:
        List of (slice_index, slice_2d) tuples
    """
    num_slices = volume.shape[axis]
    
    if slice_range is None:
        start, end = 0, num_slices
    else:
        start, end = slice_range
        start = max(0, start)
        end = min(num_slices, end)
    
    slices = []
    for i in range(start, end):
        if axis == 0:
            slice_2d = volume[i, :, :]
        elif axis == 1:
            slice_2d = volume[:, i, :]
        else:  # axis == 2
            slice_2d = volume[:, :, i]
        
        # Filter out empty or nearly-empty slices
        nonzero_ratio = np.count_nonzero(slice_2d) / slice_2d.size
        if nonzero_ratio >= min_nonzero_ratio:
            slices.append((i, slice_2d))
    
    return slices


def process_nifti_volume(
    filepath: str,
    output_dir: str,
    output_prefix: str,
    resize_size: Tuple[int, int] = (320, 320),
    crop_size: Tuple[int, int] = (256, 256),
    slice_axis: int = 2,
    slice_range: Optional[Tuple[int, int]] = (38, 50),
    min_nonzero_ratio: float = 0.1,
    normalize: str = "zscore",
    verbose: bool = True,
) -> List[str]:
    """
    Process a single NIfTI volume: extract slices, preprocess, save.
    
    Args:
        filepath: Path to .nii.gz file
        output_dir: Directory to save .npy slices
        output_prefix: Prefix for output filenames
        resize_size: Intermediate resize dimensions
        crop_size: Final crop dimensions
        slice_axis: Axis to slice along
        slice_range: Optional slice range to extract
        min_nonzero_ratio: Minimum non-zero ratio to keep slice
        normalize: Normalization method
        verbose: Print progress
        
    Returns:
        List of saved file paths
    """
    # Load volume
    volume, metadata = load_nifti(filepath)
    
    if verbose:
        print(f"  Loaded volume: {volume.shape}, dtype: {volume.dtype}")
    
    # Handle 4D volumes (take first 3D volume)
    if volume.ndim == 4:
        volume = volume[:, :, :, 0]
        if verbose:
            print(f"  Reduced 4D to 3D: {volume.shape}")
    
    # Extract slices
    slices = extract_slices_from_volume(
        volume,
        axis=slice_axis,
        slice_range=slice_range,
        min_nonzero_ratio=min_nonzero_ratio,
    )
    
    if verbose:
        print(f"  Extracted {len(slices)} valid slices")
    
    # Process and save each slice
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for slice_idx, slice_2d in slices:
        # Preprocess
        processed = preprocess_slice(
            slice_2d,
            resize_size=resize_size,
            crop_size=crop_size,
            normalize=normalize,
        )
        
        # Generate output filename
        # Format: {prefix}_slice_{slice_idx:03d}.npy
        output_name = f"{output_prefix}_slice_{slice_idx:03d}.npy"
        output_path = os.path.join(output_dir, output_name)
        
        # Save
        np.save(output_path, processed)
        saved_paths.append(output_path)
    
    return saved_paths


# =============================================================================
# Directory Traversal and Batch Processing
# =============================================================================

def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for filenames.
    Removes/replaces problematic characters.
    """
    # Remove or replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def find_nifti_files(root_dir: str) -> List[Dict]:
    """
    Recursively find all NIfTI files and extract folder hierarchy info.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of dicts with 'filepath', 'category', 'case_folder', 'volume_name'
    """
    nifti_files = []
    root_path = Path(root_dir)
    
    # Find all .nii and .nii.gz files
    for ext in ['*.nii', '*.nii.gz']:
        for filepath in root_path.rglob(ext):
            # Get relative path components
            rel_path = filepath.relative_to(root_path)
            parts = rel_path.parts
            
            # Extract hierarchy
            # Expected: category/case_folder/filename.nii.gz
            # Or: category/case_folder/subfolder/filename.nii.gz
            
            if len(parts) >= 2:
                category = parts[0]  # e.g., "ClinicalVariations"
                case_folder = parts[1]  # e.g., "T2_CUBE_FemaleBrachy"
                
                # Handle deeper nesting
                if len(parts) > 3:
                    # Include intermediate folders in case_folder
                    case_folder = "_".join(parts[1:-1])
            else:
                category = "Unknown"
                case_folder = "Unknown"
            
            # Extract volume name from filename (without extension)
            volume_name = filepath.stem
            if volume_name.endswith('.nii'):
                volume_name = volume_name[:-4]  # Remove .nii from .nii.gz files
            
            # Clean up volume name
            volume_name = sanitize_filename(volume_name)
            
            nifti_files.append({
                'filepath': str(filepath),
                'category': sanitize_filename(category),
                'case_folder': sanitize_filename(case_folder),
                'volume_name': volume_name,
            })
    
    return nifti_files


def process_external_dataset(
    input_dir: str,
    output_dir: str,
    resize_size: Tuple[int, int] = (320, 320),
    crop_size: Tuple[int, int] = (256, 256),
    slice_axis: int = 2,
    slice_range: Optional[Tuple[int, int]] = (38, 50),
    min_nonzero_ratio: float = 0.1,
    normalize: str = "zscore",
    create_manifest: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Process entire external dataset directory.
    
    Args:
        input_dir: Root input directory with NIfTI files
        output_dir: Output directory for .npy slices
        resize_size: Intermediate resize dimensions
        crop_size: Final crop dimensions
        slice_axis: Axis to slice along
        slice_range: Optional slice range to extract
        min_nonzero_ratio: Minimum non-zero ratio to keep slice
        normalize: Normalization method
        create_manifest: Whether to save a manifest file
        verbose: Print progress
        
    Returns:
        Summary dict with processing statistics
    """
    print(f"\n{'='*60}")
    print("EXTERNAL DATASET PREPROCESSING")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size:       {crop_size}")
    print(f"Normalization:    {normalize}")
    print(f"{'='*60}\n")
    
    # Find all NIfTI files
    nifti_files = find_nifti_files(input_dir)
    
    if not nifti_files:
        print("ERROR: No NIfTI files found!")
        return {"error": "No files found"}
    
    print(f"Found {len(nifti_files)} NIfTI volumes\n")
    
    # Group by category for summary
    categories = {}
    for f in nifti_files:
        cat = f['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f)
    
    print("Files by category:")
    for cat, files in categories.items():
        case_folders = set(f['case_folder'] for f in files)
        print(f"  {cat}: {len(files)} volumes in {len(case_folders)} case folders")
        for cf in sorted(case_folders):
            count = sum(1 for f in files if f['case_folder'] == cf)
            print(f"    - {cf}: {count} volumes")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each volume
    manifest = []
    total_slices = 0
    failed_volumes = []
    
    for file_info in tqdm(nifti_files, desc="Processing volumes"):
        filepath = file_info['filepath']
        category = file_info['category']
        case_folder = file_info['case_folder']
        volume_name = file_info['volume_name']
        
        # Create output prefix: category_case_volumename
        output_prefix = f"{category}_{case_folder}_{volume_name}"
        
        if verbose:
            print(f"\nProcessing: {filepath}")
            print(f"  Output prefix: {output_prefix}")
        
        try:
            saved_paths = process_nifti_volume(
                filepath=filepath,
                output_dir=output_dir,
                output_prefix=output_prefix,
                resize_size=resize_size,
                crop_size=crop_size,
                slice_axis=slice_axis,
                slice_range=slice_range,
                min_nonzero_ratio=min_nonzero_ratio,
                normalize=normalize,
                verbose=verbose,
            )
            
            total_slices += len(saved_paths)
            
            # Add to manifest
            for path in saved_paths:
                manifest.append({
                    'slice_path': path,
                    'source_volume': filepath,
                    'category': category,
                    'case_folder': case_folder,
                    'volume_name': volume_name,
                })
                
        except Exception as e:
            print(f"  ERROR processing {filepath}: {e}")
            failed_volumes.append({'filepath': filepath, 'error': str(e)})
    
    # Save manifest
    if create_manifest:
        import json
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump({
                'input_dir': input_dir,
                'output_dir': output_dir,
                'preprocessing': {
                    'resize_size': resize_size,
                    'crop_size': crop_size,
                    'slice_axis': slice_axis,
                    'normalize': normalize,
                },
                'slices': manifest,
                'failed_volumes': failed_volumes,
            }, f, indent=2)
        print(f"\nManifest saved to: {manifest_path}")
    
    # Summary
    summary = {
        'total_volumes': len(nifti_files),
        'total_slices': total_slices,
        'failed_volumes': len(failed_volumes),
        'categories': {cat: len(files) for cat, files in categories.items()},
        'output_dir': output_dir,
    }
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total volumes processed: {len(nifti_files) - len(failed_volumes)}/{len(nifti_files)}")
    print(f"Total slices saved:      {total_slices}")
    print(f"Output directory:        {output_dir}")
    
    if failed_volumes:
        print(f"\nFailed volumes ({len(failed_volumes)}):")
        for fv in failed_volumes:
            print(f"  - {fv['filepath']}: {fv['error']}")
    
    print(f"{'='*60}\n")
    
    return summary


# =============================================================================
# Dataset Class for Inference (compatible with existing DataLoader)
# =============================================================================

class ExternalNpyDataset:
    """
    Dataset class for loading preprocessed external .npy slices.
    Compatible with existing inference pipeline.
    """
    
    def __init__(
        self,
        data_dir: str,
        manifest_path: Optional[str] = None,
        category_filter: Optional[List[str]] = None,
        case_filter: Optional[List[str]] = None,
        patient_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dir: Directory containing .npy slices
            manifest_path: Optional path to manifest.json for metadata
            category_filter: Only load slices from these categories
            case_filter: Only load slices from these case folders
            patient_filter: Only load slices matching these filename patterns (exact or substring)
        """
        self.data_dir = data_dir
        self.manifest = None
        self.patient_filter = patient_filter
        
        # Load manifest if available
        if manifest_path is None:
            manifest_path = os.path.join(data_dir, "manifest.json")
        
        if os.path.exists(manifest_path):
            import json
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        
        # Find all .npy files
        self.files = []
        self.metadata = []
        
        if self.manifest is not None:
            # Use manifest for metadata
            for entry in self.manifest['slices']:
                path = entry['slice_path']
                if not os.path.exists(path):
                    path = os.path.join(data_dir, os.path.basename(path))
                
                # Apply filters
                if category_filter and entry['category'] not in category_filter:
                    continue
                if case_filter and entry['case_folder'] not in case_filter:
                    continue
                
                # Apply patient filter (filename pattern matching)
                if patient_filter:
                    filename = os.path.basename(path)
                    if not any(pattern in filename for pattern in patient_filter):
                        continue
                
                if os.path.exists(path):
                    self.files.append(path)
                    self.metadata.append(entry)
        else:
            # Fallback: just load all .npy files
            import glob
            all_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
            
            # Apply patient filter if provided
            if patient_filter:
                for f in all_files:
                    filename = os.path.basename(f)
                    if any(pattern in filename for pattern in patient_filter):
                        self.files.append(f)
                        self.metadata.append({'slice_path': f})
            else:
                self.files = all_files
                self.metadata = [{'slice_path': f} for f in self.files]
        
        print(f"ExternalNpyDataset: Loaded {len(self.files)} slices from {data_dir}")
        if patient_filter:
            print(f"  Patient filter applied: {patient_filter}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path).astype(np.float32)
        
        # Ensure correct shape for model
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # Add channel dimension
        
        import torch
        image = torch.from_numpy(arr)
        
        return {
            'image': image,
            'path': path,
            'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
        }
    
    def get_by_category(self, category: str) -> List[int]:
        """Get indices of slices from a specific category."""
        indices = []
        for i, meta in enumerate(self.metadata):
            if meta.get('category') == category:
                indices.append(i)
        return indices
    
    def get_by_case(self, case_folder: str) -> List[int]:
        """Get indices of slices from a specific case folder."""
        indices = []
        for i, meta in enumerate(self.metadata):
            if meta.get('case_folder') == case_folder:
                indices.append(i)
        return indices
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        return list(set(m.get('category', 'Unknown') for m in self.metadata))
    
    def get_case_folders(self) -> List[str]:
        """Get list of unique case folders."""
        return list(set(m.get('case_folder', 'Unknown') for m in self.metadata))


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    category_filter: Optional[List[str]] = None,
    case_filter: Optional[List[str]] = None,
    patient_filter: Optional[List[str]] = None,
) -> "torch.utils.data.DataLoader":
    """
    Create a DataLoader for external dataset inference.
    
    Args:
        data_dir: Directory with preprocessed .npy slices
        batch_size: Batch size
        num_workers: Number of data loading workers
        category_filter: Only load from these categories
        case_filter: Only load from these case folders
        patient_filter: Only load slices matching these filename patterns
        
    Returns:
        PyTorch DataLoader
    """
    import torch
    from torch.utils.data import DataLoader
    
    dataset = ExternalNpyDataset(
        data_dir=data_dir,
        category_filter=category_filter,
        case_filter=case_filter,
        patient_filter=patient_filter,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess external NIfTI dataset for anomaly detection inference"
    )
    parser.add_argument("--input-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/test_LUND_PROBE_extended", help="Root directory containing NIfTI files")
    parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/test_LUND_PROBE_extended_npy", help="Output directory for .npy slices")
    parser.add_argument("--resize-size", type=int, nargs=2, default=[320, 320], help="Intermediate resize dimensions")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[256, 256], help="Final crop dimensions")
    parser.add_argument("--slice-axis", type=int, default=2, help="Axis to extract slices (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--slice-range", type=int, nargs=2, default=[38, 50], help="Slice range (start, end)")
    parser.add_argument("--min-nonzero", type=float, default=0.1, help="Minimum non-zero pixel ratio to keep slice")
    parser.add_argument("--normalize", type=str, choices=["zscore", "minmax"], default="zscore", help="Normalization method")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Run preprocessing
    summary = process_external_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resize_size=tuple(args.resize_size),
        crop_size=tuple(args.crop_size),
        slice_axis=args.slice_axis,
        slice_range=tuple(args.slice_range) if args.slice_range else None,
        min_nonzero_ratio=args.min_nonzero,
        normalize=args.normalize,
        verbose=args.verbose,
    )
    
    print("\nYou can now run inference with:")
    print(f"  python run_inference_example.py \\")
    print(f"      --stage1-ckpt /path/to/stage1.ckpt \\")
    print(f"      --stage2-ckpt /path/to/stage2.ckpt \\")
    print(f"      --data-dir {args.output_dir} \\")
    print(f"      --output-dir inference_results/")


if __name__ == "__main__":
    main()
