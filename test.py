# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
# pip install SimpleITK
import SimpleITK as sitk
from scipy.stats import pearsonr
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
# from models import Generator
from tqdm import tqdm
from utils import (
    calculate_me, calculate_mae, calculate_mse, calculate_rmse,
    calculate_psnr, calculate_ssim, calculate_ncc, calculate_dsc,
    evaluate_all_metrics, visualize_comparison, save_metrics_to_csv,
    plot_metrics_histogram
)

def cal_mae(gts, preds):
    num_data, h, w, _ = gts.shape
    # mae = np.sum(np.abs(preds - gts)) / (num_data * h * w)
    mae = np.mean(np.abs(preds - gts))

    return mae

def cal_me(gts, preds):
    num_data, h, w, _ = gts.shape
    # me = np.sum(preds - gts) / (num_data * h * w)
    me = np.mean(preds - gts)

    return me

def cal_mse(gts, preds):
    num_data, h, w, _ = gts.shape
    # mse = np.sum(np.abs(preds - gts)**2) / (num_data * h * w)
    mse = np.mean((np.abs(preds - gts))**2)
    return mse

def cal_pcc(gts, preds):
    pcc, _ = pearsonr(gts.ravel(), preds.ravel())
    return pcc

def make_folders(is_train=True, load_model=None, dataset=None):
    """Create necessary folders for training/testing"""
    model_dir, log_dir, sample_dir, test_dir = None, None, None, None

    if is_train:
        if load_model is None:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            model_dir = "model/{}/{}".format(dataset, cur_time)

            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        else:
            cur_time = load_model
            model_dir = "model/{}/{}".format(dataset, cur_time)

        sample_dir = "sample/{}/{}".format(dataset, cur_time)
        log_dir = "logs/{}/{}".format(dataset, cur_time)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

    else:
        model_dir = "model/{}/{}".format(dataset, load_model)
        log_dir = "logs/{}/{}".format(dataset, load_model)
        test_dir = "test/{}/{}".format(dataset, load_model)

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    return model_dir, sample_dir, log_dir, test_dir

def init_logger(log_dir):
    """Initialize logger"""
    # Tạo thư mục log nếu chưa tồn tại
    os.makedirs(log_dir, exist_ok=True)

    # Cấu hình logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Tạo file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    file_handler.setLevel(logging.INFO)

    # Tạo console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Thêm handlers vào logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def visualize_results(input_mr, real_ct, generated_ct, save_path=None, show=False):
    """
    Visualize MR inputs, real CT, and generated CT images

    Args:
        input_mr: Input MR images (batch, h, w, c)
        real_ct: Real CT images (batch, h, w, c)
        generated_ct: Generated CT images (batch, h, w, c)
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    batch_size = input_mr.shape[0]

    plt.figure(figsize=(12, 4 * batch_size))

    for i in range(batch_size):
        # Display input MR
        plt.subplot(batch_size, 3, i*3 + 1)
        if input_mr.shape[-1] == 1:
            plt.imshow(input_mr[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(input_mr[i])
        plt.title('Input MR')
        plt.axis('off')

        # Display real CT
        plt.subplot(batch_size, 3, i*3 + 2)
        if real_ct.shape[-1] == 1:
            plt.imshow(real_ct[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(real_ct[i])
        plt.title('Real CT')
        plt.axis('off')

        # Display generated CT
        plt.subplot(batch_size, 3, i*3 + 3)
        if generated_ct.shape[-1] == 1:
            plt.imshow(generated_ct[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(generated_ct[i])
        plt.title('Generated CT')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def save_model(model, filepath):
    """
    Save Keras model to disk

    Args:
        model: Keras model to save
        filepath: Path to save model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load Keras model from disk

    Args:
        filepath: Path to load model from

    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(filepath)

def calculate_metrics(real, generated):
    """
    Calculate common image quality metrics

    Args:
        real: Real images (batch, h, w, c)
        generated: Generated images (batch, h, w, c)

    Returns:
        Dictionary of metrics
    """
    # Flatten tensors
    real_flat = tf.reshape(real, [-1])
    generated_flat = tf.reshape(generated, [-1])

    # Calculate metrics
    mae = tf.reduce_mean(tf.abs(real_flat - generated_flat))
    mse = tf.reduce_mean(tf.square(real_flat - generated_flat))

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    max_val = tf.reduce_max(real) - tf.reduce_min(real)
    psnr = 10.0 * tf.math.log(max_val ** 2 / mse) / tf.math.log(10.0)

    # Calculate SSIM (Structural Similarity Index)
    ssim = tf.image.ssim(real, generated, max_val=max_val)

    return {
        'mae': mae.numpy(),
        'mse': mse.numpy(),
        'psnr': psnr.numpy(),
        'ssim': tf.reduce_mean(ssim).numpy()
    }

def preprocess_for_model(images, normalize=True):
    """
    Preprocess images for use with model

    Args:
        images: Input images
        normalize: Whether to normalize to [-1, 1]

    Returns:
        Preprocessed images
    """
    # Convert to float
    images = tf.cast(images, tf.float32)

    # Normalize to [-1, 1]
    if normalize:
        images = (images / 127.5) - 1.0

    return images

def postprocess_from_model(images, denormalize=True):
    """
    Postprocess images from model

    Args:
        images: Model output images
        denormalize: Whether to denormalize from [-1, 1]

    Returns:
        Postprocessed images
    """
    # Denormalize from [-1, 1] to [0, 255]
    if denormalize:
        images = (images + 1.0) * 127.5

    # Clip values to valid range
    images = tf.clip_by_value(images, 0, 255)

    # Convert to uint8
    images = tf.cast(images, tf.uint8)

    return images

def load_and_preprocess_data(test_dir, target_size=(256, 256)):
    """Load and preprocess test data"""
    mr_dir = os.path.join(test_dir, "MR")
    ct_dir = os.path.join(test_dir, "CT")  # Original CT for comparison

    # Verify directories exist
    if not os.path.exists(mr_dir):
        raise ValueError(f"MR directory not found: {mr_dir}")

    # Get all test files
    mr_files = sorted([f for f in os.listdir(mr_dir) if f.endswith('.npy')])
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.npy')])

    mr_data = []
    ct_data = []  # Original CT data if available

    print("Loading test data...")
    for f in tqdm(mr_files):
        # Load MR data
        mr = np.load(os.path.join(mr_dir, f))

        # Resize if needed
        if mr.shape[0] != target_size[0] or mr.shape[1] != target_size[1]:
            mr = cv2.resize(mr, target_size, interpolation=cv2.INTER_LINEAR)

        # Add channel dimension if needed
        if mr.ndim == 2:
            mr = mr[..., np.newaxis]

        # Convert to float32 and normalize
        mr = mr.astype(np.float32)
        mr = (mr - np.min(mr)) / (np.max(mr) - np.min(mr) + 1e-7)

        mr_data.append(mr)

        # Load original CT if available
        if os.path.exists(ct_dir) and os.path.exists(os.path.join(ct_dir, f)):
            ct = np.load(os.path.join(ct_dir, f))

            # Resize if needed
            if ct.shape[0] != target_size[0] or ct.shape[1] != target_size[1]:
                ct = cv2.resize(ct, target_size, interpolation=cv2.INTER_LINEAR)

            # Add channel dimension if needed
            if ct.ndim == 2:
                ct = ct[..., np.newaxis]

            # Convert to float32 and normalize
            ct = ct.astype(np.float32)
            ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-7)

            ct_data.append(ct)

    # Stack into numpy arrays
    mr_data = np.array(mr_data)
    if ct_data:
        ct_data = np.array(ct_data)

    print(f"Loaded {len(mr_data)} test images, shape: {mr_data.shape}")
    return mr_data, ct_data, mr_files

def predict_ct_images(generator, mr_data, batch_size=4):
    """Generate synthetic CT images from MR images"""
    predictions = []

    # Process in batches to avoid memory issues
    for i in range(0, len(mr_data), batch_size):
        batch = mr_data[i:i+batch_size]
        batch_preds = generator.predict(batch)
        predictions.extend(batch_preds)

    return np.array(predictions)

def visualize_results(mr_images, syn_ct_images, real_ct_images=None, num_samples=5, save_dir=None):
    """Visualize original MR, synthetic CT, and real CT if available"""
    # Choose random samples
    if num_samples > len(mr_images):
        num_samples = len(mr_images)

    indices = np.random.choice(len(mr_images), num_samples, replace=False)

    # Create a figure with 2 or 3 columns based on whether real CT is available
    cols = 3 if real_ct_images is not None else 2
    fig, axes = plt.subplots(num_samples, cols, figsize=(4*cols, 4*num_samples))

    if num_samples == 1:
        axes = [axes]

    titles = ["MR Input", "Generated CT"]
    if real_ct_images is not None:
        titles.append("Original CT")

    # Add a row of titles at the top
    for j, title in enumerate(titles):
        if num_samples > 1:
            axes[0][j].set_title(title, fontsize=15)
        else:
            axes[j].set_title(title, fontsize=15)

    # Display the images
    for i, idx in enumerate(indices):
        mr = mr_images[idx].squeeze()
        syn_ct = syn_ct_images[idx].squeeze()

        if num_samples > 1:
            axes[i][0].imshow(mr, cmap='gray')
            axes[i][0].axis('off')

            axes[i][1].imshow(syn_ct, cmap='gray')
            axes[i][1].axis('off')

            if real_ct_images is not None:
                real_ct = real_ct_images[idx].squeeze()
                axes[i][2].imshow(real_ct, cmap='gray')
                axes[i][2].axis('off')
        else:
            axes[0].imshow(mr, cmap='gray')
            axes[0].axis('off')

            axes[1].imshow(syn_ct, cmap='gray')
            axes[1].axis('off')

            if real_ct_images is not None:
                real_ct = real_ct_images[idx].squeeze()
                axes[2].imshow(real_ct, cmap='gray')
                axes[2].axis('off')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"comparison_samples.png"), dpi=300, bbox_inches='tight')

    plt.show()

def save_results(syn_ct_images, filenames, output_dir):
    """Save synthetic CT images to disk"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(syn_ct_images)} synthetic CT images...")
    for i, (image, filename) in enumerate(zip(syn_ct_images, filenames)):
        # Convert to uint8 for visualization
        image_norm = image.squeeze()
        image_uint8 = (image_norm * 255).astype(np.uint8)

        # Save as PNG
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_synCT.png"), image_uint8)

        # Save as NPY for further processing
        np.save(os.path.join(output_dir, f"{base_name}_synCT.npy"), image)

    print(f"Results saved to {output_dir}")

def evaluate_performance(syn_ct_images, real_ct_images):
    """Calculate performance metrics between synthetic and real CT images"""
    if len(syn_ct_images) != len(real_ct_images):
        print("Warning: Number of synthetic and real CT images don't match!")
        min_len = min(len(syn_ct_images), len(real_ct_images))
        syn_ct_images = syn_ct_images[:min_len]
        real_ct_images = real_ct_images[:min_len]
    
    # Khởi tạo biến lưu kết quả
    metrics = {
        'ME': 0.0,       # Mean Error
        'MAE': 0.0,      # Mean Absolute Error
        'MSE': 0.0,      # Mean Squared Error
        'RMSE': 0.0,     # Root Mean Squared Error
        'PSNR': 0.0,     # Peak Signal-to-Noise Ratio
        'SSIM': 0.0,     # Structural Similarity Index
        'NCC': 0.0,      # Normalized Cross-Correlation
        'DSC': 0.0       # Dice Similarity Coefficient
    }
    
    # Tính toán từng chỉ số riêng lẻ
    metrics_list = []
    for i in range(len(syn_ct_images)):
        # Trích xuất ảnh
        syn_img = syn_ct_images[i]
        real_img = real_ct_images[i]
        
        # Đảm bảo kích thước đúng
        if len(syn_img.shape) == 3 and syn_img.shape[-1] == 1:
            syn_img = syn_img.squeeze()
        if len(real_img.shape) == 3 and real_img.shape[-1] == 1:
            real_img = real_img.squeeze()
            
        # Tính toán các chỉ số
        me = calculate_me(syn_img, real_img)
        mae = calculate_mae(syn_img, real_img)
        mse = calculate_mse(syn_img, real_img)
        rmse = calculate_rmse(syn_img, real_img)
        psnr = calculate_psnr(syn_img, real_img)
        ssim = calculate_ssim(syn_img, real_img)
        ncc = calculate_ncc(syn_img, real_img)
        dsc = calculate_dsc(syn_img, real_img)
        
        # Lưu kết quả
        img_metrics = {
            'ME': me,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'PSNR': psnr,
            'SSIM': ssim,
            'NCC': ncc,
            'DSC': dsc
        }
        metrics_list.append(img_metrics)
        
    # Tính giá trị trung bình
    for key in metrics.keys():
        metrics[key] = np.mean([m[key] for m in metrics_list])
    
    # Hiển thị kết quả
    print("\nCác chỉ số đánh giá chất lượng tái tạo CT từ MR:")
    print(f"Mean Error (ME): {metrics['ME']:.4f} (Lý tưởng: 0)")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} (Lý tưởng: 0)")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f} (Lý tưởng: 0)")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f} (Lý tưởng: 0)")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {metrics['PSNR']:.2f} dB (Cao hơn tốt hơn)")
    print(f"Structural Similarity Index (SSIM): {metrics['SSIM']:.4f} (Lý tưởng: 1)")
    print(f"Normalized Cross-Correlation (NCC): {metrics['NCC']:.4f} (Lý tưởng: 1)")
    print(f"Dice Similarity Coefficient (DSC): {metrics['DSC']:.4f} (Lý tưởng: 1)")
    
    return metrics

# if __name__ == "__main__":
#     # Configuration
#     model_path = "/content/my_folder/Brain - Copy/checkpoints/training_checkpoints/epoch=9-step=2529.ckpt-9.index"  # Thay đổi theo checkpoint bạn muốn sử dụng
#     test_data_path = "/content/my_folder/Brain - Copy/Data/splits/test/MR"               # Thư mục chứa ảnh MR test
#     output_dir = "/content/my_folder/Brain - Copy/results"                            # Thư mục lưu kết quả
#     gt_dir = None
#     batch_size = 8
#     Visualize = True
#     save_results = True                                     # Thư mục chứa CT ground truth (nếu có)

#     # Run test
#     test_model(model_path, test_data_path, output_dir)

#     # Calculate metrics if ground truth is available
#     if gt_dir:
#         calculate_metrics(gt_dir, output_dir)

def main():
    # Configuration
    config = {
        'test_dir': "/content/my_folder/Brain - Copy/Data/splits/test",
        'output_dir': "/content/my_folder/Brain - Copy/results",
        'model_path': "/content/drive/MyDrive/Bachelor Thesis/sCT from MRI/model_saved/generator_epoch_10.keras",  # Path to the saved generator model
        'batch_size': 8,
        'visualize': True,
        'save_results': True
    }

    # Load the generator model
    print(f"Loading generator model from {config['model_path']}...")
    try:
        generator = tf.keras.models.load_model(config['model_path'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Falling back to creating a new model (weights will not be loaded)...")
        generator = Generator()

    # Print model summary
    generator.summary()

    # Load and preprocess test data
    mr_data, ct_data, filenames = load_and_preprocess_data(config['test_dir'])

    # Generate synthetic CT images
    print("Generating synthetic CT images...")
    syn_ct_images = predict_ct_images(generator, mr_data, batch_size=config['batch_size'])

    # Visualize results
    if config['visualize']:
        visualize_results(
            mr_data, syn_ct_images, ct_data if len(ct_data) > 0 else None,
            num_samples=5, save_dir=config['output_dir']
        )

    # Save results
    if config['save_results']:
        save_results(syn_ct_images, filenames, config['output_dir'])

    # Evaluate performance if real CT images are available
    if len(ct_data) > 0:
        metrics = evaluate_performance(syn_ct_images, ct_data)

        # Save metrics to file
        with open(os.path.join(config['output_dir'], "performance_metrics.txt"), "w") as f:
            f.write("Các chỉ số đánh giá chất lượng tái tạo CT từ MR:\n")
            f.write(f"Mean Error (ME): {metrics['ME']:.4f} (Lý tưởng: 0)\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} (Lý tưởng: 0)\n")
            f.write(f"Mean Squared Error (MSE): {metrics['MSE']:.4f} (Lý tưởng: 0)\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f} (Lý tưởng: 0)\n")
            f.write(f"Peak Signal-to-Noise Ratio (PSNR): {metrics['PSNR']:.2f} dB (Cao hơn tốt hơn)\n")
            f.write(f"Structural Similarity Index (SSIM): {metrics['SSIM']:.4f} (Lý tưởng: 1)\n")
            f.write(f"Normalized Cross-Correlation (NCC): {metrics['NCC']:.4f} (Lý tưởng: 1)\n")
            f.write(f"Dice Similarity Coefficient (DSC): {metrics['DSC']:.4f} (Lý tưởng: 1)\n")
            
        # Tạo biểu đồ so sánh
        plt.figure(figsize=(10, 6))
        metrics_names = ['ME', 'MAE', 'MSE', 'RMSE']
        values = [metrics[metric] for metric in metrics_names]
        
        plt.subplot(1, 2, 1)
        plt.bar(metrics_names, values)
        plt.title('Các chỉ số sai số (thấp hơn tốt hơn)')
        plt.grid(True, alpha=0.3)
        
        metrics_names = ['PSNR', 'SSIM', 'NCC', 'DSC']
        values = [metrics[metric] for metric in metrics_names]
        
        plt.subplot(1, 2, 2)
        plt.bar(metrics_names, values)
        plt.title('Các chỉ số tương đồng (cao hơn tốt hơn)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], "metrics_chart.png"), dpi=200)
        plt.close()
        
        print(f"Các chỉ số đánh giá đã được lưu vào {os.path.join(config['output_dir'], 'performance_metrics.txt')}")
        print(f"Biểu đồ đánh giá đã được lưu vào {os.path.join(config['output_dir'], 'metrics_chart.png')}")

if __name__ == "__main__":
    main()