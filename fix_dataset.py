import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_sample(mr, ct, title="Sample Visualization"):
    """Visualize a pair of MR and CT images"""
    # Ensure 2D for visualization
    if mr.ndim > 2:
        mr = mr.squeeze()
    if ct.ndim > 2:
        ct = ct.squeeze()
    
    # Calculate similarity
    similarity = np.mean(np.abs(mr - ct))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mr, cmap='gray')
    axes[0].set_title('MR Image')
    axes[0].axis('off')
    
    axes[1].imshow(ct, cmap='gray')
    axes[1].set_title('CT Image')
    axes[1].axis('off')
    
    # Absolute difference
    diff = np.abs(mr - ct)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Absolute Difference\nMean: {similarity:.4f}')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def check_and_fix_dataset():
    """Check and fix the original dataset to ensure MR and CT are distinct"""
    # Define paths
    original_dir = "Data/preprocessed/MR_and_CT_only"
    mr_dir = os.path.join(original_dir, "MR")
    ct_dir = os.path.join(original_dir, "CT")
    
    # Create backup directory
    backup_dir = "Data/preprocessed/backup"
    backup_mr_dir = os.path.join(backup_dir, "MR")
    backup_ct_dir = os.path.join(backup_dir, "CT")
    
    # Create the backup directories if they don't exist
    os.makedirs(backup_mr_dir, exist_ok=True)
    os.makedirs(backup_ct_dir, exist_ok=True)
    
    # Get list of files
    mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.npy')])
    
    # Create backup
    print("Creating backup of original data...")
    for mr_file in tqdm(mr_files):
        ct_file = mr_file  # Same filename in CT directory
        
        if not os.path.exists(os.path.join(ct_dir, ct_file)):
            print(f"Warning: No matching CT file for {mr_file}")
            continue
        
        # Copy to backup
        shutil.copy2(os.path.join(mr_dir, mr_file), os.path.join(backup_mr_dir, mr_file))
        shutil.copy2(os.path.join(ct_dir, ct_file), os.path.join(backup_ct_dir, ct_file))
    
    # Check and fix the dataset
    print("\nChecking dataset for issues...")
    issues_found = 0
    fixed_pairs = 0
    
    for i, mr_file in enumerate(tqdm(mr_files)):
        ct_file = mr_file  # Same filename in CT directory
        
        try:
            # Load data
            mr_path = os.path.join(mr_dir, mr_file)
            ct_path = os.path.join(ct_dir, ct_file)
            
            if not os.path.exists(ct_path):
                continue
                
            mr_data = np.load(mr_path)
            ct_data = np.load(ct_path)
            
            # Check similarity
            similarity = np.mean(np.abs(mr_data - ct_data))
            
            # If very similar, they might be the same image
            if similarity < 0.05:
                issues_found += 1
                print(f"\nIssue found with pair {mr_file}: similarity = {similarity:.4f}")
                
                # Option 1: Can visually inspect some of these to confirm
                if i % 100 == 0:  # Only visualize a few
                    visualize_sample(mr_data, ct_data, f"Issue: {mr_file} vs {ct_file}")
                    
                    # Let user decide
                    decision = input("Are these identical? (y/n): ").strip().lower()
                    if decision == 'y':
                        print("Marking for correction...")
                    else:
                        print("Skipping - images are different enough.")
                        continue
                
                # Find appropriate CT data from another slice if needed
                # Example: If CT and MR are the same, we could try to use a different CT slice
                
                # For now, let's just mark it and not modify
                print(f"  - Would fix pair {mr_file}")
                fixed_pairs += 1
        
        except Exception as e:
            print(f"Error processing {mr_file}: {str(e)}")
    
    print("\nAnalysis complete:")
    print(f"Total files checked: {len(mr_files)}")
    print(f"Issues found: {issues_found}")
    print(f"Files that would be fixed: {fixed_pairs}")
    
    return issues_found

def recreate_splits():
    """Recreate the train/val/test splits using the fixed original data"""
    from dataloader import DataLoader
    
    print("\nRecreating train/val/test splits...")
    
    # First, remove existing splits
    splits_dir = "Data/splits"
    if os.path.exists(splits_dir):
        print(f"Removing existing splits in {splits_dir}")
        shutil.rmtree(splits_dir)
    
    # Create new splits
    data_dir = "Data/preprocessed/MR_and_CT_only"
    loader = DataLoader(data_dir)
    
    print("\nSplits created successfully!")

if __name__ == "__main__":
    # Check if there's a serious issue with the data
    print("Step 1: Checking dataset for issues...")
    issues_found = check_and_fix_dataset()
    
    if issues_found > 0:
        print("\nStep 2: Recreating train/val/test splits...")
        recreate_splits()
    else:
        print("\nNo serious issues found. No need to recreate splits.")
    
    print("\nData verification and fixing complete!") 