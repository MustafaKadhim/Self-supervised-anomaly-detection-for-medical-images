import os
from glob import glob
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torchio as tio
import numpy as np
import pytorch_lightning as pl
from monai.data import Dataset, DataLoader
import wandb
from joblib import Parallel, delayed
#from config import dictConfig
   
from inference_v3_support_CJG import convertDataMethodsClass
from inference_v3_support_CJG import outlierTransformMethodsClass
# Init needed class instances
convertData = convertDataMethodsClass() # Functions for converting DICOM to Nifti data
# Outlier transform class
outlierTransform = outlierTransformMethodsClass

# Load config 
#from config import dictConfig


# ---------------------------
# 1. Setup WandB and config
# ---------------------------
wandb.init(
    project="prostate-mri-vq-vae-2",
    name="inference",
    dir="/mnt/md1/ProjectData/Christian/wandb_logs",
    #config=dictConfig
)
config = wandb.config
checkpoint_path = config["checkpoint_path"]


# Config section for inference 
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(script_dir, 'tempInference')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
inputDICOMDirBase = os.path.join(script_dir, 'inferenceDICOM')

# Set scenarios to happen with flags
useScenarioMinus1 = False
useScenario0 = False
useScenario1 = False
useScenario2 = True


##### Scenario -1 - T2 propeller patients without prostate #####
if useScenarioMinus1==True:
    print("SCENARIO -1: T2 propeller patients without prostate")
    
    # Loop over all patients 
    for pat in ['Pat1', 'Pat2']:
        patFolder = f'T2_tra_PROP_noProstate/{pat}/'
        
        print(f"\nProcessing patient: {pat}")
        print("Converting DICOM to Nifti in no prostate folder...")
        
        # Get full path to patient folder (DICOM files are directly in this folder)
        patientFolderPath = os.path.join(inputDICOMDirBase, patFolder)
        
        # Check if patient folder exists
        if not os.path.exists(patientFolderPath):
            print(f"WARNING: Patient folder not found: {patientFolderPath}")
            continue
        
        # Check if there are DICOM files in the folder
        try:
            dicom_files = [f for f in os.listdir(patientFolderPath) 
                          if f.endswith('.dcm') or f.endswith('.DCM')]
            if not dicom_files:
                print(f"WARNING: No DICOM files found in {patientFolderPath}")
                continue
            print(f"Found {len(dicom_files)} DICOM files in {pat}")
        except Exception as e:
            print(f"ERROR reading patient folder {patientFolderPath}: {e}")
            continue
        
        # Get patient name from DICOM folder
        try:
            patientName = convertData.getPatientNameDICOM(patientFolderPath)
            print(f"  Patient name: {patientName}")
            
            # Get series description from DICOM folder
            seriesDescription = convertData.getSeriesDescriptionDICOM(patientFolderPath)
            print(f"  Series description: {seriesDescription}")
            
            # Create output subfolder for FemaleBrachyCube data
            outputSubfolder = 'T2_tra_PROP_noProstate'
            
            print(f"  Output subfolder: '{outputSubfolder}'")
            
            # Create output subfolder path
            outputSubfolderPath = os.path.join(temp_dir, outputSubfolder)
            os.makedirs(outputSubfolderPath, exist_ok=True)
            
            # Output Nifti file path (in the subfolder)
            outputNiftiFile = os.path.join(outputSubfolderPath, f"{patientName}_{seriesDescription}.nii.gz")
            
            # Convert DICOM to Nifti
            convertData.convertDICOM2Nifti(patientFolderPath, outputNiftiFile)
            print(f"  ✓ Converted to: {outputNiftiFile}")
            
        except Exception as e:
            print(f"  ERROR processing {patientFolderPath}: {e}")
            continue
    

    print("SCENARIO minus 1: Completed")
 
 
##### Scenario 0 - Female brachytherapy patients with T2 CUBE MRI #####
if useScenario0:
    print("\n" + "="*80)
    print("SCENARIO 0: Processing female brachytherapy patients with T2 CUBE MRI")
    print("="*80)
    
    # Loop over all patients 
    for pat in ['1anon', '2anon', '3anon', '4anon', '5anon']:
        patFolder = f'FemaleBrachyCube/{pat}/'
        
        print(f"\nProcessing patient: {pat}")
        print("Converting DICOM to Nifti in temp folder...")
        
        # Get full path to patient folder (DICOM files are directly in this folder)
        patientFolderPath = os.path.join(inputDICOMDirBase, patFolder)
        
        # Check if patient folder exists
        if not os.path.exists(patientFolderPath):
            print(f"WARNING: Patient folder not found: {patientFolderPath}")
            continue
        
        # Check if there are DICOM files in the folder
        try:
            dicom_files = [f for f in os.listdir(patientFolderPath) 
                          if f.endswith('.dcm') or f.endswith('.DCM')]
            if not dicom_files:
                print(f"WARNING: No DICOM files found in {patientFolderPath}")
                continue
            print(f"Found {len(dicom_files)} DICOM files in {pat}")
        except Exception as e:
            print(f"ERROR reading patient folder {patientFolderPath}: {e}")
            continue
        
        # Get patient name from DICOM folder
        try:
            patientName = convertData.getPatientNameDICOM(patientFolderPath)
            print(f"  Patient name: {patientName}")
            
            # Get series description from DICOM folder
            seriesDescription = convertData.getSeriesDescriptionDICOM(patientFolderPath)
            print(f"  Series description: {seriesDescription}")
            
            # Create output subfolder for FemaleBrachyCube data
            outputSubfolder = 'T2_CUBE_FemaleBrachy'
            
            print(f"  Output subfolder: '{outputSubfolder}'")
            
            # Create output subfolder path
            outputSubfolderPath = os.path.join(temp_dir, outputSubfolder)
            os.makedirs(outputSubfolderPath, exist_ok=True)
            
            # Output Nifti file path (in the subfolder)
            outputNiftiFile = os.path.join(outputSubfolderPath, f"{patientName}_{seriesDescription}.nii.gz")
            
            # Convert DICOM to Nifti
            convertData.convertDICOM2Nifti(patientFolderPath, outputNiftiFile)
            print(f"  ✓ Converted to: {outputNiftiFile}")
            
        except Exception as e:
            print(f"  ERROR processing {patientFolderPath}: {e}")
            continue
    
    print("\n" + "="*80)
    print("SCENARIO 0: Completed")
    print("="*80 + "\n")



##### Scenario 1 - Protes patients #####
# Convert DICOM to Nifti and save temporarily
# Set patient folder to load data from with respect to inputDICOMDir

# Loop over all patients 
for pat in ['Pat1', 'Pat2', 'Pat3', 'Pat4']:
    patFolder = f'hipProtes/{pat}/'
    seriesToConvert = ['sCT', 'T2 tra PROP', 'MAVRIC'] 

    if useScenario1:
        print(f"\nProcessing patient: {pat}")
        print("Converting DICOM to Nifti in temp folder...")
        
        # Get full path to patient folder
        patientFolderPath = os.path.join(inputDICOMDirBase, patFolder)
        
        # Check if patient folder exists
        if not os.path.exists(patientFolderPath):
            print(f"WARNING: Patient folder not found: {patientFolderPath}")
            continue
        
        # Get all subfolders in patient directory
        try:
            allSubfolders = [f for f in os.listdir(patientFolderPath) 
                           if os.path.isdir(os.path.join(patientFolderPath, f))]
            print(f"Found {len(allSubfolders)} subfolders in {pat}: {allSubfolders}")
        except Exception as e:
            print(f"ERROR reading patient folder {patientFolderPath}: {e}")
            continue
        
        # Loop over series to convert
        for seriesName in seriesToConvert:
            print(f"\nLooking for series containing: '{seriesName}'")
            
            # Find matching subfolders (containing the series name)
            matchingFolders = [folder for folder in allSubfolders 
                             if seriesName in folder]
            
            if not matchingFolders:
                print(f"  WARNING: No folder found containing '{seriesName}'")
                continue
            
            if len(matchingFolders) > 1:
                print(f"  WARNING: Multiple folders match '{seriesName}': {matchingFolders}")
                print(f"  Using first match: {matchingFolders[0]}")
            
            # Use the first matching folder
            matchedFolder = matchingFolders[0]
            print(f"  ✓ Matched folder: '{matchedFolder}'")
            
            # Determine output subfolder name
            # If sCT found, name it T2forCT, otherwise use seriesName
            if 'sCT' in seriesName or 'sct' in seriesName.lower():
                outputSubfolder = 'T2_for_sCT_protes'
            else:
                # Clean series name for folder (remove spaces, special chars)
                cleanSeriesName = seriesName.replace(' ', '_').replace('/', '_')
                outputSubfolder = f"{cleanSeriesName}_protes"
            
            print(f"  Output subfolder: '{outputSubfolder}'")
            
            # Construct input DICOM directory path
            inputDICOMDir = os.path.join(patientFolderPath, matchedFolder)
            
            # Get patient name from DICOM folder
            try:
                patientName = convertData.getPatientNameDICOM(inputDICOMDir)
                print(f"  Patient name: {patientName}")
                
                # Get series description from DICOM folder
                seriesDescription = convertData.getSeriesDescriptionDICOM(inputDICOMDir)
                print(f"  Series description: {seriesDescription}")
                
                # Create output subfolder path
                outputSubfolderPath = os.path.join(temp_dir, outputSubfolder)
                os.makedirs(outputSubfolderPath, exist_ok=True)
                
                # Output Nifti file path (in the subfolder)
                outputNiftiFile = os.path.join(outputSubfolderPath, f"{patientName}_{seriesDescription}.nii.gz")
                
                # Convert DICOM to Nifti
                convertData.convertDICOM2Nifti(inputDICOMDir, outputNiftiFile)
                print(f"  ✓ Converted to: {outputNiftiFile}")
                
            except Exception as e:
                print(f"  ERROR processing {inputDICOMDir}: {e}")
                continue


##### Scenario 2 - use TorchIO to augment normal scans #####
if useScenario2:
    print("Using TorchIO to augment normal scans and save in temp folder...")
    # Define number of patients to process (for testing)
    #nrPatToProcess = 10
    # Extended part of LUND-PROBE dataset
    orig_test_files = sorted(glob(
        # Extended part of LUND-PROBE dataset
        '/mnt/md1/ProjectData/Christian/AnomalyDetection/AnomalyDetection/data/LUND-PROBE/extendedPart/**/MR_StorT2/image.nii.gz'
        # Base part of LUND-PROBE dataset
        # '/mnt/md1/ProjectData/Christian/AnomalyDetection/AnomalyDetection/data/LUND-PROBE/basePart/**/MR_StorT2/image.nii.gz'
        # To use sCT
        #'/mnt/md1/ProjectData/Christian/AnomalyDetection/AnomalyDetection/data/LUND-PROBE/basePart/**/sCT/image.nii.gz'
        ))
    

    # Print the patients selected for augmentation
    print(f"Selected {len(orig_test_files)} patients for augmentation:")
    for f in orig_test_files:
        print(f"  {f}")

    # Define 3D TorchIO transforms (p is probability)
    # Outlier - make a square outlier in the center of the image
    torchio_transform_outlier = outlierTransform(size=10)
    # Motion artefact
    torchio_transform_randomMotion = tio.RandomMotion(degrees=30, translation=50, num_transforms=10, p=1.0, image_interpolation='linear')
    # Spike artefact
    torchio_transform_randomSpike = tio.RandomSpike(num_spikes=7, intensity=(0.95), p=1.0)
    # Ghosting artefact
    torchio_transform_randomGhosting = tio.RandomGhosting(num_ghosts=(2, 4), axes=(0, 1), intensity=(2), restore=0.02, p=1.0)
    # Noise artefact
    torchio_transform_randomNoise = tio.RandomNoise(mean=2000, std=600, p=1.0)
    # No transform (for control)
    torchio_transform_none = tio.Compose([])


    print("Applying 3D TorchIO transforms in parallell and saving in temp folder...")
    def torchioAugPatLoop(patNr, fpath, temp_dir, torchio_transform):
        # Get transform name from class
        transform_name = torchio_transform.__class__.__name__
        # If name is compose rename to orig beacause no transform is applied
        if transform_name == 'Compose':
            transform_name = 'orig'
        # If outlier transform rename to square
        if transform_name == 'outlierTransformMethodsClass':
            transform_name = 'square'             
        # Get subject name
        patient_id = Path(fpath).parents[1].name
        # Add tranform name to patient id
        patient_id = f"{patient_id}_{transform_name}"
        # Load NIfTI
        img = nib.load(fpath)
        # Get data and add channel dim
        data = img.get_fdata()[None, ...]  # add channel dim (C,H,W,D)
        # Convert to torch tensor
        img_tio = tio.ScalarImage(tensor=data)
        # # Get mean and std of original image 
        # orig_mean = img_tio.data.mean()
        # orig_std = img_tio.data.std()
        # print(f"Patient {patNr}: {patient_id}, original mean: {orig_mean:.2f}, std: {orig_std:.2f}")

        # Apply 3D TorchIO transforms
        img_aug = torchio_transform(img_tio).data
        # Save augmented NIfTI (remove channel dim)
        save_path = os.path.join(temp_dir, transform_name, f"{patient_id}.nii.gz")
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(nib.Nifti1Image(img_aug[0], img.affine, img.header), save_path)
        print(f"Augmented volumes saved in: {save_path}")
        return {"MRI_image": str(save_path), "patient_id": patient_id}
    

    def blurAugPatLoop(patNr, fpath, temp_dir):
        # Get subject name
        patient_id = Path(fpath).parents[1].name
        # Add tranform name to patient id
        patient_id = f"{patient_id}_CTVBlurred"
        # Load NIfTI
        img = nib.load(fpath)
        # Get data and add channel dim
        data = img.get_fdata()
        # Get mask for CTV. It is in the parent folder of the fpath
        mask_path = os.path.join(Path(fpath).parents[0], 'mask_CTVT_427.nii.gz')
        if not os.path.exists(mask_path):
            # throw error
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        # Blur it
        sigmaSetting = (2, 2, 1)  # Anisotropic sigma for Gaussian blur (x, y, z)
        img_aug = convertData.editMaskContent(data, mask_data, sigma=sigmaSetting, editMethod='gaussianblur')
        # Save augmented NIfTI (remove channel dim)
        save_path = os.path.join(temp_dir, 'CTVBlur', f"{patient_id}.nii.gz")
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(nib.Nifti1Image(img_aug, img.affine, img.header), save_path)
        print(f"Augmented volumes saved in: {save_path}")
        return {"MRI_image": str(save_path), "patient_id": patient_id}
    

    def averageAugPatLoop(patNr, fpath, temp_dir):
        # Get subject name
        patient_id = Path(fpath).parents[1].name
        # Add tranform name to patient id
        patient_id = f"{patient_id}_CTVAverage"
        # Load NIfTI
        img = nib.load(fpath)
        # Get data and add channel dim
        data = img.get_fdata()
        # Get mask for CTV. It is in the parent folder of the fpath
        mask_path = os.path.join(Path(fpath).parents[0], 'mask_CTVT_427.nii.gz')
        if not os.path.exists(mask_path):
            # throw error
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        # Average it (sigma has no affect)
        img_aug = convertData.editMaskContent(data, mask_data, sigma=(2, 2, 1), editMethod='average')
        # Save augmented NIfTI (remove channel dim)
        save_path = os.path.join(temp_dir, 'CTVAverage', f"{patient_id}.nii.gz")
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(nib.Nifti1Image(img_aug, img.affine, img.header), save_path)
        print(f"Augmented volumes saved in: {save_path}")
        return {"MRI_image": str(save_path), "patient_id": patient_id}
    

    def gaussianWholeImageAugPatLoop(patNr, fpath, temp_dir):
        # Get subject name
        patient_id = Path(fpath).parents[1].name
        # Add tranform name to patient id
        patient_id = f"{patient_id}_WholeImageGaussian"
        # Load NIfTI
        img = nib.load(fpath)
        # Get data
        data = img.get_fdata()
        # Apply Gaussian filter to whole image
        sigmaSetting = (6, 6, 2)  # Anisotropic sigma for Gaussian blur (x, y, z)
        img_aug = convertData.applyGaussianFilterWholeImage(data, sigma=sigmaSetting)
        # Save augmented NIfTI
        save_path = os.path.join(temp_dir, 'WholeImageGaussian', f"{patient_id}.nii.gz")
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(nib.Nifti1Image(img_aug, img.affine, img.header), save_path)
        print(f"Augmented volumes saved in: {save_path}")
        return {"MRI_image": str(save_path), "patient_id": patient_id}
    

    def randomCTVInsertionAugPatLoop(patNr, fpath, temp_dir, num_copies=3):
        """Insert multiple CTV copies at random locations within body mask for outlier detection testing"""
        # Get subject name
        patient_id = Path(fpath).parents[1].name
        # Add tranform name to patient id
        patient_id = f"{patient_id}_RandomCTVx{num_copies}"
        
        # Load NIfTI
        img = nib.load(fpath)
        # Get data
        data = img.get_fdata()
        
        # Get mask for CTV. It is in the parent folder of the fpath
        mask_path = os.path.join(Path(fpath).parents[0], 'mask_CTVT_427.nii.gz')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        # Get body mask (mask_BODY.nii.gz in the same folder)
        body_mask_path = os.path.join(Path(fpath).parents[0], 'mask_BODY.nii.gz')
        if not os.path.exists(body_mask_path):
            raise FileNotFoundError(f"Body mask file not found: {body_mask_path}")
        
        body_mask_img = nib.load(body_mask_path)
        body_mask_data = body_mask_img.get_fdata()
        
        print(f"Patient {patNr}: {patient_id} - Inserting {num_copies} CTV copies at random locations within body mask...")
        
        # Insert multiple CTV copies at random locations within body mask
        img_aug, new_mask = convertData.insertMultipleCTVRandomLocations(
            data, 
            mask_data,
            body_mask_data,  # Use body mask to constrain placement
            num_copies=num_copies,  # Number of random insertions
            blend_sigma=(1, 1, 0.5)  # Light blending at boundaries
        )
        
        # Save augmented NIfTI
        save_path = os.path.join(temp_dir, 'RandomCTVInsertion', f"{patient_id}.nii.gz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nib.save(nib.Nifti1Image(img_aug, img.affine, img.header), save_path)
        print(f"  Augmented volume saved: {save_path}")
        
        # Also save the new mask location (for reference/validation)
        mask_save_path = os.path.join(temp_dir, 'RandomCTVInsertion', f"{patient_id}_mask.nii.gz")
        nib.save(nib.Nifti1Image(new_mask, mask_img.affine, mask_img.header), mask_save_path)
        print(f"  New mask locations saved: {mask_save_path}")
        
        return {"MRI_image": str(save_path), "patient_id": patient_id}
    


    #Init parallell job for augmentation of original files in different ways
    # #Outlier
    augmented_files_square = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(torchioAugPatLoop)(patNr, patient, temp_dir, torchio_transform_outlier) for patNr, patient in enumerate(orig_test_files))
    # #Motion
    augmented_files_motion = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(torchioAugPatLoop)(patNr, patient, temp_dir, torchio_transform_randomMotion) for patNr, patient in enumerate(orig_test_files))
    # #Spike
    augmented_files_spike = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(torchioAugPatLoop)(patNr, patient, temp_dir, torchio_transform_randomSpike) for patNr, patient in enumerate(orig_test_files))
    # #Ghosting
    augmented_files_ghosting = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(torchioAugPatLoop)(patNr, patient, temp_dir, torchio_transform_randomGhosting) for patNr, patient in enumerate(orig_test_files))
    # #Noise
    augmented_files_noise = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(torchioAugPatLoop)(patNr, patient, temp_dir, torchio_transform_randomNoise) for patNr, patient in enumerate(orig_test_files))
    # #No transform (for control)
    augmented_files_none = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(torchioAugPatLoop)(patNr, patient, temp_dir, torchio_transform_none) for patNr, patient in enumerate(orig_test_files))
    # #Mask blurring
    augmented_files_blurring = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(blurAugPatLoop)(patNr, patient, temp_dir) for patNr, patient in enumerate(orig_test_files))
    # #Mask averaging
    augmented_files_averaging = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(averageAugPatLoop)(patNr, patient, temp_dir) for patNr, patient in enumerate(orig_test_files))
    # #Whole image Gaussian blur
    augmented_files_whole_gaussian = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(gaussianWholeImageAugPatLoop)(patNr, patient, temp_dir) for patNr, patient in enumerate(orig_test_files))
    # #Random CTV insertion (for outlier detection) - you can change num_copies parameter
    # Examples:
    augmented_files_random_ctv = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(randomCTVInsertionAugPatLoop)(patNr, patient, temp_dir, num_copies=4) for patNr, patient in enumerate(orig_test_files)) 
    
    #Prostatectomy simulation
    #augmented_files_prostatectomy = Parallel(n_jobs=config.nrCPU, verbose=10)(delayed(prostatectomyAugPatLoop)(patNr, patient, temp_dir) for patNr, patient in enumerate(orig_test_files))
