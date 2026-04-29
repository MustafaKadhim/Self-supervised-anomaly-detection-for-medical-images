#!/usr/bin/env python3
"""
Pre-slicing Script for VQ-VAE Training

This script loads each 3D NIfTI volume once, extracts all 2D slices,
applies normalization, and saves them as lightweight .npy files.

This eliminates the slow NIfTI loading during training and makes
data loading nearly instant.

Usage:
    python preslice_volumes_224x224.py
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from glob import glob
import json
import time
from config import dictConfig #please provide your own config file with the appropriate dataPath for SOURCE_GLOB
# ============ Configuration ============
SOURCE_GLOB = dictConfig["dataPath"]
OUTPUT_DIR = "/home/mluser1/Musti_Anomaly_Detection/Data/PreSliced"
# Note: Slices are saved at original resolution (e.g., 512x512 or similar).
# ========================================


def preprocess_and_save():
    """
    Main function to pre-slice all volumes and save as .npy files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob(SOURCE_GLOB, recursive=True))
    
    print(f"Found {len(files)} volumes. Starting slicing...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Track statistics
    total_slices = 0
    skipped_slices = 0
    patient_slice_counts = {}
    
    for file_path in tqdm(files, desc="Processing volumes"):
        # 1. Load Volume ONCE
        try:
            nii = nib.load(file_path)
            vol = nii.get_fdata().astype(np.float32)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue
            
        # Get patient ID for naming
        patient_id = Path(file_path).parents[1].name
        
        # 2. Normalize using same approach as ExtractSlice2D
        # Percentile clipping
        flat = vol.reshape(-1)
        
        # Subsample for percentile estimation if needed
        if flat.size > 100_000:
            idx_sample = np.random.choice(flat.size, 100_000, replace=False)
            sample = flat[idx_sample]
        else:
            sample = flat
            
        #p05 = np.percentile(sample, 0.5)
        #p995 = np.percentile(sample, 99.5)
        #vol = np.clip(vol, p05, p995)
        
        # Z-score normalization
        mean = vol.mean()
        std = vol.std()
        if std < 1e-8:
            std = 1.0
        vol = (vol - mean) / std
        
        # 3. Extract and Save Slices (assuming axis=2 is the slice axis)
        patient_slices = 0
        for slice_idx in range(vol.shape[2]):
            slice_img = vol[:, :, slice_idx]
            
            # Filter empty slices (optional but recommended)
            # Using a threshold based on normalized values
            #if slice_img.max() - slice_img.min() < 0.1:  # Skip nearly uniform slices
            #    skipped_slices += 1
            #    continue
                
            # Save as .npy (fastest I/O)
            save_name = f"{patient_id}_slice_{slice_idx:03d}.npy"
            np.save(os.path.join(OUTPUT_DIR, save_name), slice_img.astype(np.float32))
            
            total_slices += 1
            patient_slices += 1
        
        patient_slice_counts[patient_id] = patient_slices
    
    # Save metadata about the slicing
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_glob": SOURCE_GLOB,
        "output_dir": OUTPUT_DIR,
        "total_volumes": len(files),
        "total_slices": total_slices,
        "skipped_slices": skipped_slices,
        "patient_slice_counts": patient_slice_counts,
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "preslice_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Pre-slicing complete!")
    print(f"Total volumes processed: {len(files)}")
    print(f"Total slices saved: {total_slices}")
    print(f"Slices skipped (empty): {skipped_slices}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    preprocess_and_save()
