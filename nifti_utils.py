# ---------------------------------------------------------
# Python Implementation for NIFTI data loading
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2

def load_nifti(file_path):
    """Load a NIfTI file and return its data as a numpy array."""
    img = nib.load(file_path)
    return img.get_fdata()

def normalize_volume(volume, min_val=0, max_val=255):
    """Normalize volume to range [min_val, max_val]."""
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-7)
    volume = volume * (max_val - min_val) + min_val
    return volume

def extract_slices(volume, axis=2):
    """Extract 2D slices from 3D volume along specified axis."""
    if axis == 0:
        slices = [volume[i, :, :] for i in range(volume.shape[0])]
    elif axis == 1:
        slices = [volume[:, i, :] for i in range(volume.shape[1])]
    else:  # axis == 2, default
        slices = [volume[:, :, i] for i in range(volume.shape[2])]
    
    return slices

def preprocess_slice(slice_data, target_size=(256, 256)):
    """Preprocess a single slice: resize, normalize."""
    # Normalize to 0-255
    slice_data = normalize_volume(slice_data)
    
    # Resize to target size if needed
    if slice_data.shape[0] != target_size[0] or slice_data.shape[1] != target_size[1]:
        slice_data = cv2.resize(slice_data, target_size, interpolation=cv2.INTER_LINEAR)
    
    return slice_data.astype(np.uint8)

def get_corresponding_slices(mr_path, ct_path, mask_path=None, axis=2):
    """
    Load MR and CT volumes and extract corresponding slices.
    
    Args:
        mr_path: Path to MR .nii.gz file
        ct_path: Path to CT .nii.gz file
        mask_path: Optional path to mask .nii.gz file
        axis: Axis along which to extract slices (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        Tuple of (mr_slices, ct_slices, mask_slices)
    """
    # Load volumes
    mr_vol = load_nifti(mr_path)
    ct_vol = load_nifti(ct_path)
    
    # Make sure they have the same shape
    if mr_vol.shape != ct_vol.shape:
        raise ValueError(f"MR and CT volumes have different shapes: {mr_vol.shape} vs {ct_vol.shape}")
    
    # Extract slices
    mr_slices = extract_slices(mr_vol, axis)
    ct_slices = extract_slices(ct_vol, axis)
    
    # Process each slice
    processed_mr = [preprocess_slice(s) for s in mr_slices]
    processed_ct = [preprocess_slice(s) for s in ct_slices]
    
    # Handle mask if provided
    if mask_path:
        mask_vol = load_nifti(mask_path)
        mask_slices = extract_slices(mask_vol, axis)
        processed_masks = [preprocess_slice(s) for s in mask_slices]
        return processed_mr, processed_ct, processed_masks
    
    # Generate binary masks from CT images if no mask provided
    processed_masks = []
    for ct_slice in processed_ct:
        from utils import get_mask
        mask = get_mask(ct_slice, task='m2c')
        processed_masks.append(mask)
    
    return processed_mr, processed_ct, processed_masks

def prepare_dataset_from_nifti(mr_paths, ct_paths, mask_paths=None):
    """
    Prepare dataset from lists of NIFTI files.
    
    Args:
        mr_paths: List of paths to MR .nii.gz files
        ct_paths: List of paths to CT .nii.gz files
        mask_paths: Optional list of paths to mask .nii.gz files
        
    Returns:
        List of triplets (mr_slice, ct_slice, mask_slice)
    """
    all_triplets = []
    
    for i in range(len(mr_paths)):
        mr_path = mr_paths[i]
        ct_path = ct_paths[i]
        mask_path = mask_paths[i] if mask_paths else None
        
        mr_slices, ct_slices, mask_slices = get_corresponding_slices(mr_path, ct_path, mask_path)
        
        # Create triplets
        for j in range(len(mr_slices)):
            if np.sum(mask_slices[j]) > 100:  # Skip slices with too small masks
                all_triplets.append((mr_slices[j], ct_slices[j], mask_slices[j]))
    
    return all_triplets 