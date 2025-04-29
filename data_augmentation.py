import os
import numpy as np
import cv2
from tqdm import tqdm
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def normalize_data(data):
    """Normalize data to [0,1] range"""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        return data
    return (data - data_min) / (data_max - data_min)

def create_augmentation_sequence():
    """Create an imgaug sequence for data augmentation"""
    seq = iaa.Sequential([
        # Geometric Augmentations
        iaa.OneOf([
            iaa.Affine(
                rotate=(-10, 10),
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.PiecewiseAffine(scale=(0.01, 0.02)),
            iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.25)
        ]),
        
        # Flip operations
        iaa.OneOf([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ]),
        
        # Intensity Augmentations (only for MR images)
        iaa.Sometimes(0.3, [
            iaa.OneOf([
                iaa.GammaContrast((0.8, 1.2)),
                iaa.LinearContrast((0.8, 1.2)),
            ])
        ])
    ], random_order=False)
    
    return seq

def augment_slice_pair(mr_slice, ct_slice, seq):
    """Apply augmentation to a pair of slices using imgaug"""
    # First, ensure we're starting with different data
    # Calculate initial similarity
    initial_similarity = np.mean(np.abs(mr_slice - ct_slice))
    if initial_similarity < 0.05:
        print(f"WARNING: MR and CT slices are very similar (similarity: {initial_similarity:.4f})")
        print(f"MR shape: {mr_slice.shape}, range: [{np.min(mr_slice)}, {np.max(mr_slice)}]")
        print(f"CT shape: {ct_slice.shape}, range: [{np.min(ct_slice)}, {np.max(ct_slice)}]")
    
    # Make a copy to ensure we don't modify the originals by reference
    mr_slice_copy = mr_slice.copy()
    ct_slice_copy = ct_slice.copy()
    
    # Ensure inputs are 2D for processing
    mr_2d = mr_slice_copy.squeeze()
    ct_2d = ct_slice_copy.squeeze()
    
    # Convert to uint8 for imgaug (temporary)
    mr_uint8 = (mr_2d * 255).astype(np.uint8)
    ct_uint8 = (ct_2d * 255).astype(np.uint8)
    
    # Create segmentation map for CT (to ensure same geometric transformations)
    segmap = SegmentationMapsOnImage(ct_uint8, shape=mr_uint8.shape)
    
    # Apply augmentation
    mr_aug, ct_segmap_aug = seq(image=mr_uint8, segmentation_maps=segmap)
    
    # Get the augmented CT image from the segmentation map
    ct_aug = ct_segmap_aug.get_arr()
    
    # Verify the outputs are different
    if np.array_equal(mr_aug, ct_aug):
        print("ERROR: Augmented MR and CT are identical after augmentation!")
        # In this case, we'll use the original data with minimal augmentation
        simple_seq = iaa.Sequential([iaa.Fliplr(0.5)])
        mr_aug, ct_segmap_simple = simple_seq(image=mr_uint8, segmentation_maps=segmap)
        ct_aug = ct_segmap_simple.get_arr()
        
        # If they're still identical, just return the original
        if np.array_equal(mr_aug, ct_aug):
            print("ERROR: Cannot create distinct augmentation. Using original data.")
            mr_aug = mr_uint8
            ct_aug = ct_uint8
    
    # Convert back to float and normalize
    mr_aug_float = normalize_data(mr_aug.astype(np.float32))
    ct_aug_float = normalize_data(ct_aug.astype(np.float32))
    
    # Calculate final similarity to check
    final_similarity = np.mean(np.abs(mr_aug_float - ct_aug_float))
    if final_similarity < 0.05:
        print(f"WARNING: Augmented MR and CT are very similar (similarity: {final_similarity:.4f})")
    
    # Add channel dimension back if needed
    if mr_slice.ndim > 2:
        mr_aug_float = mr_aug_float[..., np.newaxis]
    if ct_slice.ndim > 2:
        ct_aug_float = ct_aug_float[..., np.newaxis]
    
    return mr_aug_float, ct_aug_float

def random_augment_slice(mr_slice, ct_slice):
    """Random augmentation for a single MR-CT slice pair
    
    This function creates a random augmentation sequence and applies
    it to both MR and CT slices, ensuring they undergo the same
    geometric transformations.
    
    Args:
        mr_slice: MRI slice as numpy array
        ct_slice: CT slice as numpy array
        
    Returns:
        Tuple of (augmented_mr, augmented_ct)
    """
    # Create a random augmentation sequence
    seq = create_augmentation_sequence()
    
    # Apply the same augmentation to both slices
    return augment_slice_pair(mr_slice, ct_slice, seq)

def augment_training_data(train_mr, train_ct):
    """Augment entire training dataset using imgaug"""
    augmented_mr = []
    augmented_ct = []
    
    # Create augmentation sequence
    seq = create_augmentation_sequence()
    
    print("Applying data augmentation to training set...")
    for i in tqdm(range(len(train_mr))):
        # Get original data
        mr = train_mr[i]
        ct = train_ct[i]
        
        # Apply augmentation
        aug_mr, aug_ct = augment_slice_pair(mr, ct, seq)
        
        # Store augmented data
        augmented_mr.append(aug_mr)
        augmented_ct.append(aug_ct)
    
    return np.array(augmented_mr), np.array(augmented_ct)

if __name__ == "__main__":
    # Define directories
    input_dir = "Data/Preprocessed"
    output_dir = "Data/Augmented"
    
    # Create output directories
    mr_output = os.path.join(output_dir, "MR")
    ct_output = os.path.join(output_dir, "CT")
    
    os.makedirs(mr_output, exist_ok=True)
    os.makedirs(ct_output, exist_ok=True)
    
    # Get list of files
    mr_files = sorted([f for f in os.listdir(os.path.join(input_dir, "MR")) if f.endswith('.npy')])
    
    print(f"Found {len(mr_files)} original files. Starting augmentation...")
    
    # Create augmentation sequence
    seq = create_augmentation_sequence()
    
    file_idx = 0
    for mr_file in tqdm(mr_files):
        # Load original data
        mr_path = os.path.join(input_dir, "MR", mr_file)
        ct_path = os.path.join(input_dir, "CT", mr_file)
        
        mr_slice = np.load(mr_path)
        ct_slice = np.load(ct_path)
        
        # Normalize if needed
        mr_slice = normalize_data(mr_slice)
        ct_slice = normalize_data(ct_slice)
        
        # Apply augmentation
        aug_mr, aug_ct = augment_slice_pair(mr_slice, ct_slice, seq)
        
        # Save augmented pair
        output_base = f"slice_{file_idx:04d}"
        np.save(os.path.join(mr_output, f"{output_base}.npy"), aug_mr)
        np.save(os.path.join(ct_output, f"{output_base}.npy"), aug_ct)
        file_idx += 1
    
    print(f"Augmentation complete. Generated {file_idx} total slices.")