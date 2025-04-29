import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_pair(mr_file, ct_file, output_path=None):
    """Visualize an MR-CT pair"""
    # Load data
    mr = np.load(mr_file)
    ct = np.load(ct_file)
    
    # Print basic info
    print(f"MR shape: {mr.shape}, dtype: {mr.dtype}")
    print(f"CT shape: {ct.shape}, dtype: {ct.dtype}")
    print(f"MR range: [{np.min(mr)}, {np.max(mr)}]")
    print(f"CT range: [{np.min(ct)}, {np.max(ct)}]")
    
    # Calculate similarity
    similarity = np.mean(np.abs(mr - ct))
    print(f"Similarity (mean abs diff): {similarity}")
    
    # Ensure 2D for visualization
    if mr.ndim > 2:
        mr = mr.squeeze()
    if ct.ndim > 2:
        ct = ct.squeeze()
    
    # Create plot
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
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return similarity

def check_original_data(data_dir="Data/preprocessed/MR_and_CT_only", num_samples=10):
    """Check original data before processing"""
    mr_dir = os.path.join(data_dir, "MR")
    ct_dir = os.path.join(data_dir, "CT")
    
    # Create output directory
    output_dir = "original_data_check"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Checking original data...")
    
    # Get list of files
    mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.npy')])
    
    # Get samples evenly spaced through the dataset
    indices = np.linspace(0, len(mr_files)-1, num_samples).astype(int)
    
    similarities = []
    for i, idx in enumerate(indices):
        mr_file = os.path.join(mr_dir, mr_files[idx])
        ct_file = os.path.join(ct_dir, mr_files[idx])
        
        if not os.path.exists(ct_file):
            print(f"WARNING: No matching CT file for {mr_files[idx]}")
            continue
        
        print(f"\nChecking file pair {i+1}/{len(indices)}:")
        print(f"MR: {mr_file}")
        print(f"CT: {ct_file}")
        
        output_path = os.path.join(output_dir, f"original_pair_{i}_{mr_files[idx]}.png")
        
        try:
            similarity = visualize_pair(mr_file, ct_file, output_path)
            similarities.append(similarity)
        except Exception as e:
            print(f"Error processing files: {str(e)}")
    
    avg_similarity = np.mean(similarities) if similarities else 0
    print(f"\nAverage similarity: {avg_similarity}")
    print(f"All similarities: {similarities}")
    
    return similarities

def compare_processed_with_original():
    """Compare original data with processed data in splits"""
    # Directories
    orig_mr_dir = "Data/preprocessed/MR_and_CT_only/MR"
    orig_ct_dir = "Data/preprocessed/MR_and_CT_only/CT"
    train_mr_dir = "Data/splits/train/MR"
    train_ct_dir = "Data/splits/train/CT"
    
    # Create output directory
    output_dir = "comparison_check"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get some sample files from train split
    train_mr_files = sorted([f for f in os.listdir(train_mr_dir) if f.endswith('.npy')])[:5]
    
    results = []
    for i, train_file in enumerate(train_mr_files):
        # Load training data
        train_mr = np.load(os.path.join(train_mr_dir, train_file))
        train_ct = np.load(os.path.join(train_ct_dir, train_file))
        
        print(f"\nChecking training file {i+1}/{len(train_mr_files)}: {train_file}")
        
        # Try to find a matching original file (this is approximate)
        orig_similarity = float('inf')
        best_match = None
        
        # Sample some original files
        orig_files = sorted([f for f in os.listdir(orig_mr_dir) if f.endswith('.npy')])
        sample_indices = np.linspace(0, len(orig_files)-1, 20).astype(int)
        
        for idx in sample_indices:
            orig_file = orig_files[idx]
            orig_mr = np.load(os.path.join(orig_mr_dir, orig_file))
            
            # Compare
            if orig_mr.shape == train_mr.shape:
                diff = np.mean(np.abs(orig_mr - train_mr))
                if diff < orig_similarity:
                    orig_similarity = diff
                    best_match = orig_file
        
        if best_match:
            print(f"Best matching original file: {best_match} (similarity: {orig_similarity:.4f})")
            
            # Load the matching original files
            orig_mr = np.load(os.path.join(orig_mr_dir, best_match))
            orig_ct = np.load(os.path.join(orig_ct_dir, best_match))
            
            # Compare train MR with orig MR
            mr_mr_similarity = np.mean(np.abs(train_mr - orig_mr))
            print(f"Train MR vs Original MR similarity: {mr_mr_similarity:.4f}")
            
            # Compare train CT with orig CT
            ct_ct_similarity = np.mean(np.abs(train_ct - orig_ct))
            print(f"Train CT vs Original CT similarity: {ct_ct_similarity:.4f}")
            
            # Compare train CT with orig MR (should be very different)
            ct_mr_similarity = np.mean(np.abs(train_ct - orig_mr))
            print(f"Train CT vs Original MR similarity: {ct_mr_similarity:.4f}")
            
            # Compare train MR with train CT
            train_similarity = np.mean(np.abs(train_mr - train_ct))
            print(f"Train MR vs Train CT similarity: {train_similarity:.4f}")
            
            # Compare orig MR with orig CT
            orig_similarity = np.mean(np.abs(orig_mr - orig_ct))
            print(f"Original MR vs Original CT similarity: {orig_similarity:.4f}")
            
            # Visualize both pairs
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Ensure 2D for visualization
            if train_mr.ndim > 2:
                train_mr = train_mr.squeeze()
            if train_ct.ndim > 2:
                train_ct = train_ct.squeeze()
            if orig_mr.ndim > 2:
                orig_mr = orig_mr.squeeze()
            if orig_ct.ndim > 2:
                orig_ct = orig_ct.squeeze()
            
            # Plot train data
            axes[0, 0].imshow(train_mr, cmap='gray')
            axes[0, 0].set_title('Train MR')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(train_ct, cmap='gray')
            axes[0, 1].set_title('Train CT')
            axes[0, 1].axis('off')
            
            diff_train = np.abs(train_mr - train_ct)
            axes[0, 2].imshow(diff_train, cmap='hot')
            axes[0, 2].set_title(f'Train Diff: {train_similarity:.4f}')
            axes[0, 2].axis('off')
            
            # Plot original data
            axes[1, 0].imshow(orig_mr, cmap='gray')
            axes[1, 0].set_title(f'Original MR: {best_match}')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(orig_ct, cmap='gray')
            axes[1, 1].set_title(f'Original CT: {best_match}')
            axes[1, 1].axis('off')
            
            diff_orig = np.abs(orig_mr - orig_ct)
            axes[1, 2].imshow(diff_orig, cmap='hot')
            axes[1, 2].set_title(f'Original Diff: {orig_similarity:.4f}')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{i}_{train_file}_vs_{best_match}.png"))
            plt.close()
            
            print(f"Comparison saved to {output_dir}/comparison_{i}_{train_file}_vs_{best_match}.png")
            
            results.append({
                'train_file': train_file,
                'orig_file': best_match,
                'train_similarity': train_similarity,
                'orig_similarity': orig_similarity,
                'mr_mr_similarity': mr_mr_similarity,
                'ct_ct_similarity': ct_ct_similarity,
                'ct_mr_similarity': ct_mr_similarity
            })
    
    return results

if __name__ == "__main__":
    # Check original data
    print("Step 1: Checking original data...")
    similarities = check_original_data(num_samples=10)
    
    print("\n\nStep 2: Comparing original data with processed data...")
    comparison_results = compare_processed_with_original()
    
    print("\n\nAnalysis complete. Check the output directories for visualizations.") 