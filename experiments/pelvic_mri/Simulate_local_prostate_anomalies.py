# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Functions to support inference scripts
# *********************************************************************************

import os
import cv2
import csv
import sys
import numpy as np
import os.path
import pydicom
import nibabel as nib
import SimpleITK as sitk
import scipy
import shutil
from datetime import datetime, timedelta
import time 
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation, gaussian_filter, distance_transform_edt
from skimage.restoration import inpaint_biharmonic
import numpy as np




class outlierTransformMethodsClass:
    """
    Class describing functions needed for outlier transform in the form of a bright square in the center of the image
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img_tio):
        data = img_tio.data
        c, h, w, d = data.shape
        cx, cy, cz = h // 2, w // 2, d // 2
        meanValue = (data[:, cx-self.size:cx+self.size, cy-self.size:cy+self.size, cz-1:cz+2]).mean()
        data[:, cx-self.size:cx+self.size, cy-self.size:cy+self.size, cz-1:cz+2] = meanValue * 5.0
        img_tio.set_data(data)
        return img_tio
    
 
class convertDataMethodsClass:
    """
    Class describing functions needed for converting DICOM data to Nifti 
    """

    def __init__ (self):
        """
        Init function
        """
        pass


    def editMaskContent(self, image, mask, sigma, editMethod='gaussianblur'):
        """
        Edit the contents of the image within the mask using a 3D Gaussian filter.

        Args:
            image (numpy.ndarray): The 3D image array.
            mask (numpy.ndarray): The 3D binary mask array (same size as image, values 0 or 1).
            editMethod (str): The method to apply ('blur' or other future methods).
            sigma (tuple): Standard deviation for Gaussian blur in 3D (anisotropic).

        Returns:
            numpy.ndarray: The modified image.
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array"
        assert isinstance(mask, np.ndarray), "Mask must be a numpy array"
        assert image.shape == mask.shape, "Image and mask must have the same shape"
        assert np.all(np.logical_or(mask == 0, mask == 1)), "Mask must be binary (0 or 1)"
        assert isinstance(sigma, (tuple, list)) and len(sigma) == 3, "Sigma must be a tuple of three values (x, y, z)"
        # Make a copy of the image
        modified_image = image.copy()

        if editMethod == 'gaussianblur':
            # Apply a 3D Gaussian blur with anisotropic sigma
            blurred_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
            # Replace only the masked regions
            np.putmask(modified_image, mask, blurred_image)

        elif editMethod == 'average':
            # Compute the mean value within the masked region
            mean_value = image[mask > 0].mean()
            # Replace masked regions with the mean value
            np.putmask(modified_image, mask, mean_value)

        # Return the modified image with blurred region
        return modified_image
    

    def applyGaussianFilterWholeImage(self, image, sigma):
        """
        Apply a 3D Gaussian filter to the entire image.

        Args:
            image (numpy.ndarray): The 3D image array.
            sigma (tuple): Standard deviation for Gaussian blur in 3D (anisotropic).
                          Can be a single float or tuple of three values (sigma_x, sigma_y, sigma_z).

        Returns:
            numpy.ndarray: The blurred image.
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array"
        assert isinstance(sigma, (tuple, list, float, int)), "Sigma must be a number or tuple of values"
        
        if isinstance(sigma, (tuple, list)):
            assert len(sigma) == 3, "Sigma must be a tuple of three values (x, y, z)"
        
        # Apply a 3D Gaussian blur with anisotropic sigma
        blurred_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
        
        return blurred_image
    
   

    def insertMultipleCTVRandomLocations(self, image, ctv_mask, body_mask, num_copies=1, blend_sigma=(1, 1, 0.5)):
        """
        Insert multiple copies of CTV content at random locations within the body mask.
        This simulates placing multiple prostates at random locations for outlier detection testing.
        
        Args:
            image (numpy.ndarray): The 3D image array.
            ctv_mask (numpy.ndarray): The 3D binary mask of the CTV (prostate).
            body_mask (numpy.ndarray): The 3D binary mask of the body region for valid placement.
            num_copies (int): Number of CTV copies to insert at random locations.
            blend_sigma (tuple): Sigma for Gaussian smoothing at boundaries to blend insertion.
        
        Returns:
            tuple: (modified_image, combined_mask)
                - modified_image: Image with all CTV copies inserted at random locations
                - combined_mask: Binary mask showing where all CTVs were inserted
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array"
        assert isinstance(ctv_mask, np.ndarray), "CTV mask must be a numpy array"
        assert isinstance(body_mask, np.ndarray), "Body mask must be a numpy array"
        assert image.shape == ctv_mask.shape, "Image and mask must have the same shape"
        assert image.shape == body_mask.shape, "Image and body mask must have the same shape"
        assert num_copies > 0, "Number of copies must be positive"
        
        print(f"  Inserting {num_copies} CTV cop{'y' if num_copies == 1 else 'ies'} at random location(s) within body mask...")
        
        # Start with original image
        modified_image = image.copy()
        combined_mask = np.zeros_like(ctv_mask, dtype=np.uint8)
        
        # Get image dimensions
        D, H, W = image.shape
        
        # Find bounding box of original CTV
        ctv_bool = ctv_mask.astype(bool)
        if not ctv_bool.any():
            print("  WARNING: CTV mask is empty, returning original image")
            return modified_image, combined_mask
        
        # Get CTV bounding box
        z_indices, y_indices, x_indices = np.where(ctv_bool)
        z_min, z_max = z_indices.min(), z_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        ctv_size_z = z_max - z_min + 1
        ctv_size_y = y_max - y_min + 1
        ctv_size_x = x_max - x_min + 1
        
        print(f"  CTV bounding box size: {ctv_size_z}×{ctv_size_y}×{ctv_size_x}")
        
        # Use body mask to define valid placement region
        body_bool = body_mask.astype(bool)
        if not body_bool.any():
            print("  WARNING: Body mask is empty, cannot place CTV copies")
            return modified_image, combined_mask
        
        # Get body bounding box
        body_z, body_y, body_x = np.where(body_bool)
        body_z_min, body_z_max = body_z.min(), body_z.max()
        body_y_min, body_y_max = body_y.min(), body_y.max()
        body_x_min, body_x_max = body_x.min(), body_x.max()
        
        print(f"  Body mask region: Z[{body_z_min}:{body_z_max}], Y[{body_y_min}:{body_y_max}], X[{body_x_min}:{body_x_max}]")
        
        # Valid range for center of CTV placement (must fit within body and image bounds)
        valid_z_min = max(body_z_min + ctv_size_z // 2, ctv_size_z // 2)
        valid_z_max = min(body_z_max - ctv_size_z // 2, D - ctv_size_z // 2 - 1)
        valid_y_min = max(body_y_min + ctv_size_y // 2, ctv_size_y // 2)
        valid_y_max = min(body_y_max - ctv_size_y // 2, H - ctv_size_y // 2 - 1)
        valid_x_min = max(body_x_min + ctv_size_x // 2, ctv_size_x // 2)
        valid_x_max = min(body_x_max - ctv_size_x // 2, W - ctv_size_x // 2 - 1)
        
        # Check if we have valid placement space
        if valid_z_min >= valid_z_max or valid_y_min >= valid_y_max or valid_x_min >= valid_x_max:
            print("  WARNING: CTV is too large to fit within body mask")
            return modified_image, combined_mask
        
        print(f"  Valid placement range: Z[{valid_z_min}:{valid_z_max}], Y[{valid_y_min}:{valid_y_max}], X[{valid_x_min}:{valid_x_max}]")
        
        # Get CTV intensities once
        ctv_intensities = image[ctv_bool]
        orig_center_z = (z_min + z_max) // 2
        orig_center_y = (y_min + y_max) // 2
        orig_center_x = (x_min + x_max) // 2
        
        # Insert multiple copies
        for copy_idx in range(num_copies):
            print(f"  Copy {copy_idx + 1}/{num_copies}:")
            
            # Try multiple attempts to find a valid position within body mask
            max_attempts = 50
            valid_position_found = False
            
            for attempt in range(max_attempts):
                # Random center position within valid ranges
                new_center_z = np.random.randint(valid_z_min, valid_z_max + 1)
                new_center_y = np.random.randint(valid_y_min, valid_y_max + 1)
                new_center_x = np.random.randint(valid_x_min, valid_x_max + 1)
                
                # Calculate offset
                offset_z = new_center_z - orig_center_z
                offset_y = new_center_y - orig_center_y
                offset_x = new_center_x - orig_center_x
                
                # Create temporary mask to check if CTV fits within body
                temp_mask = np.zeros_like(ctv_mask, dtype=bool)
                ctv_fits_in_body = True
                
                for z in range(z_min, z_max + 1):
                    for y in range(y_min, y_max + 1):
                        for x in range(x_min, x_max + 1):
                            if ctv_bool[z, y, x]:
                                new_z = z + offset_z
                                new_y = y + offset_y
                                new_x = x + offset_x
                                
                                # Check if within bounds and within body mask
                                if 0 <= new_z < D and 0 <= new_y < H and 0 <= new_x < W:
                                    if body_bool[new_z, new_y, new_x]:
                                        temp_mask[new_z, new_y, new_x] = True
                                    else:
                                        ctv_fits_in_body = False
                                        break
                                else:
                                    ctv_fits_in_body = False
                                    break
                        if not ctv_fits_in_body:
                            break
                    if not ctv_fits_in_body:
                        break
                
                # Check if ENTIRE CTV is within body (100% requirement)
                if temp_mask.sum() == ctv_bool.sum():  # All CTV voxels must be within body
                    valid_position_found = True
                    copy_mask = temp_mask
                    print(f"    New center: ({new_center_z}, {new_center_y}, {new_center_x}) [attempt {attempt + 1}]")
                    break
            
            if not valid_position_found:
                print(f"    WARNING: Could not find valid position within body mask after {max_attempts} attempts, skipping this copy")
                continue
            
            # Calculate offset for logging
            offset_z = new_center_z - orig_center_z
            offset_y = new_center_y - orig_center_y
            offset_x = new_center_x - orig_center_x
            
            # Insert CTV content at new location
            copy_mask_indices = np.where(copy_mask)
            if len(copy_mask_indices[0]) > 0:
                # Assign intensities to new location
                num_new_voxels = len(copy_mask_indices[0])
                num_orig_voxels = len(ctv_intensities)
                
                if num_new_voxels <= num_orig_voxels:
                    # Use first N intensities
                    modified_image[copy_mask] = ctv_intensities[:num_new_voxels]
                else:
                    # Repeat and sample
                    sampled_intensities = np.random.choice(ctv_intensities, size=num_new_voxels, replace=True)
                    modified_image[copy_mask] = sampled_intensities
                
                # Add to combined mask
                combined_mask[copy_mask] = copy_idx + 1  # Label each copy with unique ID
                
                # Optional: Blend boundaries with Gaussian smoothing
                if blend_sigma is not None:
                    from scipy.ndimage import binary_dilation
                    blend_mask = binary_dilation(copy_mask, iterations=3) & ~copy_mask
                    
                    if blend_mask.any():
                        # Smooth the transition
                        smoothed = scipy.ndimage.gaussian_filter(modified_image, sigma=blend_sigma)
                        modified_image = np.where(blend_mask, smoothed, modified_image)
                
                print(f"    ✓ Inserted (offset: {offset_z}, {offset_y}, {offset_x})")
        
        print(f"  ✓ All {num_copies} CTV cop{'y' if num_copies == 1 else 'ies'} inserted successfully!")
        
        return modified_image, combined_mask
        
  
    
    def getSeriesDescriptionDICOM(self, inputDicomDir):
        """
        Get series descriptions from DICOM directory. 
        Validates that all DICOM files have the same SeriesDescription.
        Args:
            inputDicomDir: Directory with DICOM files
        Returns:
            seriesDescription: Series description string
        Raises:
            ValueError: If DICOM files have different SeriesDescription values
        """
        dicom_files = [f for f in os.listdir(inputDicomDir) if f.endswith('.dcm')]
        if not dicom_files:
            raise FileNotFoundError("No DICOM files found in the specified directory.")
        
        # Read all DICOM files and check for consistency
        series_descriptions = set()
        for dicom_file in dicom_files:
            dicom_path = os.path.join(inputDicomDir, dicom_file)
            ds = pydicom.dcmread(dicom_path)
            if hasattr(ds, 'SeriesDescription'):
                series_descriptions.add(ds.SeriesDescription)
            else:
                raise AttributeError(f"SeriesDescription not found in {dicom_file}")
        
        # Verify all files have the same SeriesDescription
        if len(series_descriptions) > 1:
            raise ValueError(f"Inconsistent SeriesDescription values found: {series_descriptions}")
        
        return series_descriptions.pop()
    

    def getPatientNameDICOM(self, inputDicomDir):
        """
        Get patient name from DICOM directory.
        Validates that all DICOM files have the same PatientName.
        Args:
            inputDicomDir: Directory with DICOM files
        Returns:
            patientName: Patient name string (format: FamilyName_GivenName)
        Raises:
            ValueError: If DICOM files have different PatientName values
        """
        dicom_files = [f for f in os.listdir(inputDicomDir) if f.endswith('.dcm')]
        if not dicom_files:
            raise FileNotFoundError("No DICOM files found in the specified directory.")
        
        # Read all DICOM files and check for consistency
        patient_names = set()
        for dicom_file in dicom_files:
            dicom_path = os.path.join(inputDicomDir, dicom_file)
            ds = pydicom.dcmread(dicom_path)
            if hasattr(ds, 'PatientName'):
                # Create consistent name format
                patient_name = ds.PatientName.family_name + "_" + ds.PatientName.given_name
                patient_names.add(patient_name)
            else:
                raise AttributeError(f"PatientName not found in {dicom_file}")
        
        # Verify all files have the same PatientName
        if len(patient_names) > 1:
            raise ValueError(f"Inconsistent PatientName values found: {patient_names}")
        
        return patient_names.pop()
    

    def convertDICOM2Nifti(self, inputDicomDir, outputNiftiFile):
        """
        Convert DICOM series to Nifti file
        Args:
            inputDicomDir: Directory with DICOM files
            outputNiftiFile: Output Nifti file path
        Returns:
            None
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(inputDicomDir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, outputNiftiFile)
        return

