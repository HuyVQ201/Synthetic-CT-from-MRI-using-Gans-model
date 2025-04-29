import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_comparison(mr_file, ct_file, save_path=None):
    """Visualize and compare a pair of MR and CT images"""
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
    
    # Ensure 2D
    if mr.ndim > 2:
        mr = mr.squeeze()
    if ct.ndim > 2:
        ct = ct.squeeze()
    
    # Visualization
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
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return similarity

def check_data_integrity(data_dir="Data/splits", num_samples=5):
    """Check data integrity in the train, val, and test sets"""
    results = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        mr_dir = os.path.join(split_dir, 'MR')
        ct_dir = os.path.join(split_dir, 'CT')
        
        if not os.path.exists(mr_dir) or not os.path.exists(ct_dir):
            print(f"Directory {split_dir} is missing MR or CT subdirectories")
            continue
        
        mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.npy')])
        ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.npy')])
        
        print(f"\n===== {split.upper()} SET =====")
        print(f"Found {len(mr_files)} MR files and {len(ct_files)} CT files")
        
        if len(mr_files) != len(ct_files):
            print(f"WARNING: Number of MR files ({len(mr_files)}) does not match number of CT files ({len(ct_files)})")
        
        # Check matching filenames
        matching_names = sum(1 for mr, ct in zip(mr_files, ct_files) if mr == ct)
        print(f"Matching filenames: {matching_names}/{min(len(mr_files), len(ct_files))}")
        
        # Sample a few files to visualize
        num_to_check = min(num_samples, len(mr_files))
        indices = np.linspace(0, len(mr_files)-1, num_to_check).astype(int)
        
        similarities = []
        for i in indices:
            if i < len(mr_files) and i < len(ct_files):
                mr_file = os.path.join(mr_dir, mr_files[i])
                ct_file = os.path.join(ct_dir, ct_files[i])
                
                print(f"\nChecking file pair {i+1}/{num_to_check}:")
                print(f"MR: {mr_files[i]}, CT: {ct_files[i]}")
                
                os.makedirs(f"debug_images/{split}", exist_ok=True)
                save_path = f"debug_images/{split}/pair_{i}_{mr_files[i]}.png"
                
                try:
                    similarity = visualize_comparison(mr_file, ct_file, save_path)
                    similarities.append(similarity)
                except Exception as e:
                    print(f"Error visualizing files: {str(e)}")
        
        results[split] = {
            'mean_similarity': np.mean(similarities) if similarities else None,
            'similarities': similarities
        }
        
        print(f"\nSummary for {split}:")
        print(f"Mean similarity: {results[split]['mean_similarity']}")
        print(f"All similarities: {results[split]['similarities']}")
    
    return results

if __name__ == "__main__":
    print("Checking data integrity...")
    os.makedirs("debug_images", exist_ok=True)
    results = check_data_integrity(num_samples=10)
    
    print("\n===== OVERALL SUMMARY =====")
    for split, result in results.items():
        print(f"{split}: Mean similarity = {result['mean_similarity']}") 