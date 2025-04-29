# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
import SimpleITK as sitk
from scipy.stats import pearsonr
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf


# def load_data(img_names, is_test=False, size=256):
#     mrImgs, ctImgs, maskImgs = [], [], []
#     for _, img_name in enumerate(img_names):
#         img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
#         mrImg, ctImg, maskImg = img[:, :size], img[:, size:2*size], img[:, -size:]

#         if not is_test:
#             mrImg, ctImg, maskImg = data_augment(mrImg, ctImg, maskImg)

#         # maskImg converte to binary image
#         maskImg[maskImg < 127.5] = 0.
#         maskImg[maskImg >= 127.5] = 1.

#         mrImgs.append(transform(mrImg))
#         ctImgs.append(transform(ctImg))
#         maskImgs.append(maskImg.astype(np.uint8))

#     return np.expand_dims(np.asarray(mrImgs), axis=3), np.expand_dims(np.asarray(ctImgs), axis=3), \
#            np.expand_dims(np.asarray(maskImgs), axis=3)

# def data_augment(mrImg, ctImg, maskImg, size=256, scope=20):
#     # Random translation
#     jitter = np.random.randint(low=0, high=scope)

#     mrImg = cv2.resize(mrImg, dsize=(size+scope, size+scope), interpolation=cv2.INTER_LINEAR)
#     ctImg = cv2.resize(ctImg, dsize=(size+scope, size+scope), interpolation=cv2.INTER_LINEAR)
#     maskImg = cv2.resize(maskImg, dsize=(size+scope, size+scope), interpolation=cv2.INTER_LINEAR)

#     mrImg = mrImg[jitter:jitter+size, jitter:jitter+size]
#     ctImg = ctImg[jitter:jitter+size, jitter:jitter+size]
#     maskImg = maskImg[jitter:jitter+size, jitter:jitter+size]

#     # Random flip
#     if np.random.uniform() > 0.5:
#         mrImg, ctImg, maskImg = mrImg[:, ::-1], ctImg[:, ::-1], maskImg[:, ::-1]

#     return mrImg, ctImg, maskImg

# def transform(img):
#     return (img - 127.5).astype(np.float32)

# def inv_transform(img, max_value=255., min_value=0., is_squeeze=True, dtype=np.uint8):
#     if is_squeeze:
#         img = np.squeeze(img)           # (N, H, W, 1) to (N, H, W)

#     img = np.round(img + 127.5)     # (-127.5~127.5) to (0~255)
#     img[img>max_value] = max_value
#     img[img<min_value] = min_value

#     return img.astype(dtype)

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

