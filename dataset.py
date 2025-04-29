# ---------------------------------------------------------
# Tensorflow DCNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Modified to support NIFTI data
# ---------------------------------------------------------
import os
import glob
import numpy as np
import random
import cv2
import nibabel as nib

from utils import all_files_under, load_data, transform
from nifti_utils import prepare_dataset_from_nifti, get_corresponding_slices

class Dataset(object):
    def __init__(self, dataset_name, num_cross_vals, idx_cross_val, use_nifti=True):
        self.use_nifti = use_nifti
        
        if use_nifti:
            # Load NIFTI data
            data_dir = os.path.join('../../Data', dataset_name)
            mr_files = sorted(glob.glob(os.path.join(data_dir, 'MR', '*.nii.gz')))
            ct_files = sorted(glob.glob(os.path.join(data_dir, 'CT', '*.nii.gz')))
            mask_files = sorted(glob.glob(os.path.join(data_dir, 'MASK', '*.nii.gz')))
            
            # If no masks, set to None
            if not mask_files:
                mask_files = None
                
            # Make sure we have matching MR and CT files
            if len(mr_files) != len(ct_files):
                raise ValueError(f"Number of MR files ({len(mr_files)}) doesn't match CT files ({len(ct_files)})")
                
            # Process all NIFTI triplets
            if mask_files:
                self.all_triplets = prepare_dataset_from_nifti(mr_files, ct_files, mask_files)
            else:
                self.all_triplets = prepare_dataset_from_nifti(mr_files, ct_files)
                
            # Shuffle data
            random.seed(42)  # For reproducibility
            random.shuffle(self.all_triplets)
            
            # Split into train/val/test
            total_samples = len(self.all_triplets)
            samples_per_split = total_samples // num_cross_vals
            
            test_start = idx_cross_val * samples_per_split
            test_end = (idx_cross_val + 1) * samples_per_split
            val_start = ((idx_cross_val + 1) % num_cross_vals) * samples_per_split
            val_end = ((idx_cross_val + 2) % num_cross_vals) * samples_per_split
            
            # Extract splits
            self.test_data = self.all_triplets[test_start:test_end]
            self.val_data = self.all_triplets[val_start:val_end]
            
            # Everything else is training data
            self.train_data = []
            for i in range(total_samples):
                if not (test_start <= i < test_end or val_start <= i < val_end):
                    self.train_data.append(self.all_triplets[i])
            
        else:
            # Original PNG-based dataset loading
            filenames = all_files_under(os.path.join('../../Data', dataset_name, 'post'), extension='png')

            blocks = []
            num_each_split = int(np.floor(len(filenames) / num_cross_vals))
            for idx in range(num_cross_vals):
                blocks.append(filenames[idx*num_each_split:(idx+1)*num_each_split])

            self.test_data = blocks[idx_cross_val]
            self.val_data = blocks[np.mod(idx_cross_val + 1, num_cross_vals)]
            del blocks[idx_cross_val], blocks[0 if idx_cross_val == len(blocks) else idx_cross_val]
            self.train_data = [item for sub_block in blocks for item in sub_block] + \
                          filenames[-np.mod(len(filenames), num_cross_vals):]

        self.num_train = len(self.train_data)
        self.num_val = len(self.val_data)
        self.num_test = len(self.test_data)

    def train_batch(self, batch_size):
        if self.use_nifti:
            # Select random batch from training triplets
            batch_triplets = random.sample(self.train_data, batch_size)
            
            # Extract MR, CT, and mask from triplets
            mrImgs = np.expand_dims(np.array([transform(triplet[0]) for triplet in batch_triplets]), axis=3)
            ctImgs = np.expand_dims(np.array([transform(triplet[1]) for triplet in batch_triplets]), axis=3)
            maskImgs = np.expand_dims(np.array([triplet[2].astype(np.uint8) for triplet in batch_triplets]), axis=3)
            
            return mrImgs, ctImgs, maskImgs
        else:
            # Original PNG loading
            batch_files = np.random.choice(self.train_data, batch_size, replace=False)
            batch_x, batch_y, batch_mask = load_data(batch_files, is_test=False)
            return batch_x, batch_y, batch_mask

    def val_batch(self):
        if self.use_nifti:
            # Extract all validation data
            mrImgs = np.expand_dims(np.array([transform(triplet[0]) for triplet in self.val_data]), axis=3)
            ctImgs = np.expand_dims(np.array([transform(triplet[1]) for triplet in self.val_data]), axis=3)
            maskImgs = np.expand_dims(np.array([triplet[2].astype(np.uint8) for triplet in self.val_data]), axis=3)
            
            return mrImgs, ctImgs, maskImgs
        else:
            # Original PNG loading
            x, y, mask = load_data(self.val_data, is_test=True)
            return x, y, mask

    def test_batch(self):
        if self.use_nifti:
            # Extract all test data
            mrImgs = np.expand_dims(np.array([transform(triplet[0]) for triplet in self.test_data]), axis=3)
            ctImgs = np.expand_dims(np.array([transform(triplet[1]) for triplet in self.test_data]), axis=3)
            maskImgs = np.expand_dims(np.array([triplet[2].astype(np.uint8) for triplet in self.test_data]), axis=3)
            
            return mrImgs, ctImgs, maskImgs
        else:
            # Original PNG loading
            x, y, mask = load_data(self.test_data, is_test=True)
            return x, y, mask

    def load_nifti_data(self, mr_path, ct_path, mask_path=None):
        """Load NIFTI data and preprocess"""
        # Load MR and CT volumes
        mr_img = nib.load(mr_path)
        ct_img = nib.load(ct_path)
        
        # Get data arrays
        mr_data = mr_img.get_fdata()
        ct_data = ct_img.get_fdata()
        
        # Resize volumes to match target size (256x256)
        target_size = (256, 256)
        mr_data = np.array([cv2.resize(slice, target_size) for slice in mr_data])
        ct_data = np.array([cv2.resize(slice, target_size) for slice in ct_data])
        
        # Normalize data
        mr_data = (mr_data - mr_data.min()) / (mr_data.max() - mr_data.min())
        ct_data = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min())
        
        # Add channel dimension
        mr_data = np.expand_dims(mr_data, axis=-1)
        ct_data = np.expand_dims(ct_data, axis=-1)
        
        # Load mask if provided
        mask_data = None
        if mask_path and os.path.exists(mask_path):
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            mask_data = np.array([cv2.resize(slice, target_size) for slice in mask_data])
            mask_data = np.expand_dims(mask_data, axis=-1)
        
        return mr_data, ct_data, mask_data
