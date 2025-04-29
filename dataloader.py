import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm
import shutil

class DataLoader:
    """DataLoader for handling preprocessed MRI and CT data"""
    
    def __init__(self, data_dir, split_output_dir="Data/splits"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing preprocessed MR and CT data
            split_output_dir: Directory to save train/val/test splits
        """
        self.data_dir = data_dir
        self.split_output_dir = split_output_dir
        
        # Create split directories
        self.train_dir = os.path.join(split_output_dir, "train")
        self.val_dir = os.path.join(split_output_dir, "val")
        self.test_dir = os.path.join(split_output_dir, "test")
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(os.path.join(split_dir, "MR"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "CT"), exist_ok=True)
        
        # Get all file paths
        mr_dir = os.path.join(data_dir, "MR")
        ct_dir = os.path.join(data_dir, "CT")
        
        # Get list of files
        self.mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.npy')])
        self.ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.npy')])
        
        # Verify that MR and CT files match one-to-one
        if len(self.mr_files) != len(self.ct_files):
            raise ValueError(f"Number of MR files ({len(self.mr_files)}) does not match number of CT files ({len(self.ct_files)})")
        
        # Verify that filenames match to ensure proper pairing
        for mr_file, ct_file in zip(self.mr_files, self.ct_files):
            if mr_file != ct_file:
                raise ValueError(f"MR file {mr_file} does not match CT file {ct_file}. Files must have matching names.")
        
        # Split data (70% train, 20% val, 10% test)
        file_indices = np.arange(len(self.mr_files))
        train_idx, test_idx = train_test_split(file_indices, test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.22, random_state=42)  # 0.22 of 90% ≈ 20% of total
        
        # Save splits to respective directories
        print("Saving train/val/test splits...")
        self._save_split(train_idx, mr_dir, ct_dir, self.train_dir, "train")
        self._save_split(val_idx, mr_dir, ct_dir, self.val_dir, "validation")
        self._save_split(test_idx, mr_dir, ct_dir, self.test_dir, "test")
    
    def _save_split(self, indices, mr_src_dir, ct_src_dir, output_dir, split_name):
        """Save a data split to disk."""
        print(f"\nSaving {split_name} split...")
        for i, idx in enumerate(tqdm(indices)):
            # Get source file names - ensure we get the same filename for both MR and CT
            src_file = self.mr_files[idx]
            
            # Copy MR file
            src_mr = os.path.join(mr_src_dir, src_file)
            dst_mr = os.path.join(output_dir, "MR", f"slice_{i:04d}.npy")
            shutil.copy2(src_mr, dst_mr)
            
            # Copy matching CT file
            src_ct = os.path.join(ct_src_dir, src_file)  # Using same filename ensures proper pairing
            dst_ct = os.path.join(output_dir, "CT", f"slice_{i:04d}.npy")
            shutil.copy2(src_ct, dst_ct)

def apply_augmentation_to_train():
    """Apply augmentation to training data."""
    # Create backup directories
    train_dir = "Data/splits/train"
    mr_dir = os.path.join(train_dir, "MR")
    ct_dir = os.path.join(train_dir, "CT")
    
    backup_dir = "Data/splits/train_backup"
    backup_mr_dir = os.path.join(backup_dir, "MR")
    backup_ct_dir = os.path.join(backup_dir, "CT")
    
    # Create backup directories if they don't exist
    os.makedirs(backup_mr_dir, exist_ok=True)
    os.makedirs(backup_ct_dir, exist_ok=True)
    
    # Create lists of files
    mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.npy')])
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.npy')])
    
    # Verify same number of files
    if len(mr_files) != len(ct_files):
        raise ValueError(f"Number of MR files ({len(mr_files)}) does not match number of CT files ({len(ct_files)})")
    
    # Create backup of original data
    print("\nBacking up original training data...")
    for i, (mr_file, ct_file) in enumerate(zip(mr_files, ct_files)):
        shutil.copy2(os.path.join(mr_dir, mr_file), os.path.join(backup_mr_dir, mr_file))
        shutil.copy2(os.path.join(ct_dir, ct_file), os.path.join(backup_ct_dir, ct_file))
    
    # Import augmentation function
    try:
        from data_augmentation import random_augment_slice
    except ImportError:
        print("Error: Could not import random_augment_slice from data_augmentation.py")
        print("Please ensure the file exists and contains the required function.")
        return
    
    # Apply augmentation
    print("\nApplying augmentation to training data...")
    for i, (mr_file, ct_file) in enumerate(tqdm(zip(mr_files, ct_files))):
        try:
            # Load original data
            mr_data = np.load(os.path.join(mr_dir, mr_file))
            ct_data = np.load(os.path.join(ct_dir, ct_file))
            
            # Verify data is distinct before augmentation
            pre_similarity = np.mean(np.abs(mr_data - ct_data))
            if pre_similarity < 0.01:
                print(f"WARNING: MR and CT data for {mr_file} appear to be identical before augmentation!")
                continue
            
            # Apply augmentation
            aug_mr, aug_ct = random_augment_slice(mr_data, ct_data)
            
            # Verify that the augmented data is still distinct
            post_similarity = np.mean(np.abs(aug_mr - aug_ct))
            if post_similarity < 0.01:
                print(f"WARNING: Augmented MR and CT data for {mr_file} appear to be identical after augmentation!")
                print(f"Similarity before: {pre_similarity}, after: {post_similarity}")
                continue
            
            # Save augmented data to temporary files first
            tmp_mr_file = os.path.join(mr_dir, f"tmp_{mr_file}")
            tmp_ct_file = os.path.join(ct_dir, f"tmp_{ct_file}")
            
            np.save(tmp_mr_file, aug_mr)
            np.save(tmp_ct_file, aug_ct)
            
            # Verify the saved files are correct
            saved_mr = np.load(tmp_mr_file)
            saved_ct = np.load(tmp_ct_file)
            
            saved_similarity = np.mean(np.abs(saved_mr - saved_ct))
            if saved_similarity < 0.01:
                print(f"WARNING: Saved MR and CT files for {mr_file} appear to be identical after saving!")
                print(f"Not replacing original files.")
                os.remove(tmp_mr_file)
                os.remove(tmp_ct_file)
                continue
            
            # If everything looks good, replace original files
            os.replace(tmp_mr_file, os.path.join(mr_dir, mr_file))
            os.replace(tmp_ct_file, os.path.join(ct_dir, ct_file))
            
        except Exception as e:
            print(f"Error applying augmentation to {mr_file} and {ct_file}: {str(e)}")
            print("Restoring original files from backup...")
            shutil.copy2(os.path.join(backup_mr_dir, mr_file), os.path.join(mr_dir, mr_file))
            shutil.copy2(os.path.join(backup_ct_dir, ct_file), os.path.join(ct_dir, ct_file))

if __name__ == "__main__":
    # Split the data
    data_dir = "Data/preprocessed/MR_and_CT_only"
    loader = DataLoader(data_dir)
    
    # Apply augmentation to training data
    apply_augmentation_to_train()
    
    print("\nData preparation complete!")
    print("Train data location: Data/splits/train")
    print("Validation data location: Data/splits/val")
    print("Test data location: Data/splits/test")
    
    # Add message to check data
    print("\nTo verify data integrity, run: python src/debug_data.py")