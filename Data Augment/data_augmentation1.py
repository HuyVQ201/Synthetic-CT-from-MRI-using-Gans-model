import os
import numpy as np
from scipy.ndimage import rotate, zoom
import cv2
from tqdm import tqdm
import random

def normalize_data(data):
    """Normalize data to [0,1] range"""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        return data
    return (data - data_min) / (data_max - data_min)

def rotate_image(image, angle):
    """Rotate image by given angle."""
    # Ensure input is 2D
    if image.ndim > 2:
        image = image.squeeze()
    
    # Perform rotation
    rotated = rotate(image, angle, reshape=False, mode='nearest')
    
    # Normalize back to [0,1]
    return normalize_data(rotated)

def zoom_image(image, factor):
    """Zoom image by given factor."""
    # Ensure input is 2D
    if image.ndim > 2:
        image = image.squeeze()
    
    h, w = image.shape
    
    # Apply zoom
    zoomed = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    
    # Handle size difference
    if factor > 1:  # Zoom in
        # Crop center
        zh, zw = zoomed.shape
        start_h = (zh - h) // 2
        start_w = (zw - w) // 2
        zoomed = zoomed[start_h:start_h+h, start_w:start_w+w]
    else:  # Zoom out
        # Pad with zeros
        new_img = np.zeros_like(image)
        zh, zw = zoomed.shape
        start_h = (h - zh) // 2
        start_w = (w - zw) // 2
        new_img[start_h:start_h+zh, start_w:start_w+zw] = zoomed
        zoomed = new_img
    
    # Normalize back to [0,1]
    return normalize_data(zoomed)

def flip_image(image, direction='horizontal'):
    """Flip image horizontally or vertically."""
    if direction == 'horizontal':
        return np.fliplr(image)
    return np.flipud(image)

def adjust_contrast(image, factor):
    """Adjust image contrast."""
    mean = np.mean(image)
    adjusted = (image - mean) * factor + mean
    return np.clip(adjusted, 0, 1)

def random_augment_slice(mr_slice, ct_slice):
    """Apply random augmentation to a pair of slices."""
    # Ensure inputs are 2D
    mr_slice = mr_slice.squeeze()
    ct_slice = ct_slice.squeeze()
    
    # Define augmentation options
    augmentation_options = [
        ('rotate', [90, 180, 270]),
        ('zoom', [0.9, 1.1]),
        ('flip', ['horizontal']),
        ('contrast', [0.8, 1.2])
    ]
    
    # Randomly select one augmentation
    aug_type, params = random.choice(augmentation_options)
    param = random.choice(params)
    
    try:
        if aug_type == 'rotate':
            aug_mr = rotate_image(mr_slice, param)
            aug_ct = rotate_image(ct_slice, param)
        elif aug_type == 'zoom':
            aug_mr = zoom_image(mr_slice, param)
            aug_ct = zoom_image(ct_slice, param)
        elif aug_type == 'flip':
            aug_mr = flip_image(mr_slice, param)
            aug_ct = flip_image(ct_slice, param)
        else:  # contrast
            aug_mr = adjust_contrast(mr_slice, param)
            aug_ct = ct_slice.copy()
        
        # Add channel dimension back
        aug_mr = aug_mr[..., np.newaxis]
        aug_ct = aug_ct[..., np.newaxis]
        
        return aug_mr, aug_ct
        
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")
        # Return original data if augmentation fails
        return mr_slice[..., np.newaxis], ct_slice[..., np.newaxis]

def augment_training_data(train_mr, train_ct):
    """Augment entire training dataset."""
    augmented_mr = []
    augmented_ct = []
    
    print("Applying data augmentation to training set...")
    for i in tqdm(range(len(train_mr))):
        # Get original data
        mr = train_mr[i]
        ct = train_ct[i]
        
        # Apply random augmentation
        aug_mr, aug_ct = random_augment_slice(mr, ct)
        
        # Store augmented data
        augmented_mr.append(aug_mr)
        augmented_ct.append(aug_ct)
    
    return np.array(augmented_mr), np.array(augmented_ct)

if __name__ == "__main__":
    # Define directories
    input_dir = "Data/Preprocessed"  # Directory containing preprocessed data
    output_dir = "Data/Augmented"    # Directory for augmented data
    
    # Create output directories
    mr_output = os.path.join(output_dir, "MR")
    ct_output = os.path.join(output_dir, "CT")
    
    os.makedirs(mr_output, exist_ok=True)
    os.makedirs(ct_output, exist_ok=True)
    
    # Get list of files
    mr_files = sorted([f for f in os.listdir(os.path.join(input_dir, "MR")) if f.endswith('.npy')])
    
    print(f"Found {len(mr_files)} original files. Starting augmentation...")
    
    file_idx = 0
    for mr_file in tqdm(mr_files):
        # Load original data
        mr_path = os.path.join(input_dir, "MR_and_CT_only", "MR", mr_file)
        ct_path = os.path.join(input_dir, "MR_and_CT_only", "CT", mr_file)
        
        mr_slice = np.load(mr_path)
        ct_slice = np.load(ct_path)
        
        # Apply random augmentation
        aug_mr, aug_ct = random_augment_slice(mr_slice, ct_slice)
        
        # Save augmented pair
        output_base = f"slice_{file_idx:04d}"
        np.save(os.path.join(mr_output, f"{output_base}.npy"), aug_mr)
        np.save(os.path.join(ct_output, f"{output_base}.npy"), aug_ct)
        file_idx += 1
    
    print(f"Augmentation complete. Generated {file_idx} total slices.") 