import os
import shutil
import tempfile
import tensorflow as tf
from dataloader import DataLoader
from models import Generator, Discriminator
import datetime
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_data(train_dir, val_dir, batch_size=8):
    """Prepare training and validation datasets

    Args:
        train_dir: Directory containing training data with MR and CT subdirectories
        val_dir: Directory containing validation data with MR and CT subdirectories
        batch_size: Batch size for training

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print(f"Loading training data from {train_dir}")
    train_mr_dir = os.path.join(train_dir, "MR")
    train_ct_dir = os.path.join(train_dir, "CT")

    print(f"Loading validation data from {val_dir}")
    val_mr_dir = os.path.join(val_dir, "MR")
    val_ct_dir = os.path.join(val_dir, "CT")

    # Verify directories exist
    for dir_path in [train_mr_dir, train_ct_dir, val_mr_dir, val_ct_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")

    # Load and preprocess training data
    train_mr_data = []
    train_ct_data = []

    # Get all training file names
    train_mr_files = sorted([f for f in os.listdir(train_mr_dir) if f.endswith('.npy')])

    target_size = (256, 256)

    # Process all training files
    print("Loading training data...")
    for f in tqdm(train_mr_files):
        # Load data
        mr = np.load(os.path.join(train_mr_dir, f))
        ct = np.load(os.path.join(train_ct_dir, f))

        # Resize to 256x256 if needed
        if mr.shape[0] != 256 or mr.shape[1] != 256:
            mr = cv2.resize(mr, target_size, interpolation=cv2.INTER_LINEAR)
        if ct.shape[0] != 256 or ct.shape[1] != 256:
            ct = cv2.resize(ct, target_size, interpolation=cv2.INTER_LINEAR)

        # Add channel dimension if needed
        if mr.ndim == 2:
            mr = mr[..., np.newaxis]
        if ct.ndim == 2:
            ct = ct[..., np.newaxis]

        # Convert to float32
        mr = mr.astype(np.float32)
        ct = ct.astype(np.float32)

        # Normalize data to [0,1] range
        mr = (mr - np.min(mr)) / (np.max(mr) - np.min(mr) + 1e-7)
        ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-7)

        train_mr_data.append(mr)
        train_ct_data.append(ct)

    # Load and preprocess validation data
    val_mr_data = []
    val_ct_data = []

    # Get all validation file names
    val_mr_files = sorted([f for f in os.listdir(val_mr_dir) if f.endswith('.npy')])

    # Process all validation files
    print("Loading validation data...")
    for f in tqdm(val_mr_files):
        # Load data
        mr = np.load(os.path.join(val_mr_dir, f))
        ct = np.load(os.path.join(val_ct_dir, f))

        # Resize to 256x256 if needed
        if mr.shape[0] != 256 or mr.shape[1] != 256:
            mr = cv2.resize(mr, target_size, interpolation=cv2.INTER_LINEAR)
        if ct.shape[0] != 256 or ct.shape[1] != 256:
            ct = cv2.resize(ct, target_size, interpolation=cv2.INTER_LINEAR)

        # Add channel dimension if needed
        if mr.ndim == 2:
            mr = mr[..., np.newaxis]
        if ct.ndim == 2:
            ct = ct[..., np.newaxis]

        # Convert to float32
        mr = mr.astype(np.float32)
        ct = ct.astype(np.float32)

        # Normalize data to [0,1] range
        mr = (mr - np.min(mr)) / (np.max(mr) - np.min(mr) + 1e-7)
        ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-7)

        val_mr_data.append(mr)
        val_ct_data.append(ct)

    # Convert to numpy arrays
    train_mr_data = np.array(train_mr_data)
    train_ct_data = np.array(train_ct_data)
    val_mr_data = np.array(val_mr_data)
    val_ct_data = np.array(val_ct_data)

    print(f"Loaded {len(train_mr_data)} training pairs and {len(val_mr_data)} validation pairs")
    print(f"Training data shape: MR={train_mr_data.shape}, CT={train_ct_data.shape}")
    print(f"Validation data shape: MR={val_mr_data.shape}, CT={val_ct_data.shape}")
    print(f"Data type: MR={train_mr_data.dtype}, CT={train_ct_data.dtype}")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_mr_data, train_ct_data))
    train_dataset = train_dataset.shuffle(len(train_mr_data)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_mr_data, val_ct_data))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset

def ensure_clean_dir(directory):
    """Ensure directory exists and is empty"""
    directory = os.path.abspath(directory)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def evaluate_model(generator, test_dataset, save_dir=None):
    """Đánh giá mô hình dựa trên các chỉ số từ bài báo

    Args:
        generator: Mô hình generator đã được huấn luyện
        test_dataset: Tập dữ liệu kiểm tra (validation hoặc test)
        save_dir: Thư mục lưu kết quả hình ảnh (nếu có)

    Returns:
        dict: Từ điển chứa các chỉ số đánh giá
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Khởi tạo biến lưu kết quả
    metrics = {
        'ME': [],           # Mean Error
        'MAE': [],          # Mean Absolute Error
        'PSNR': [],         # Peak Signal-to-Noise Ratio
        'SSIM': [],         # Structural Similarity Index
        'NCC': [],          # Normalized Cross-Correlation
        'DSC': []           # Dice Similarity Coefficient
    }
    
    # Đánh giá từng batch
    num_samples = 0
    for i, (mr_batch, ct_batch) in enumerate(test_dataset):
        # Dự đoán CT từ MR
        predicted_ct = generator(mr_batch, training=False)
        
        # Chuyển về numpy để tính toán
        predicted_ct_np = predicted_ct.numpy()
        real_ct_np = ct_batch.numpy()
        
        batch_size = mr_batch.shape[0]
        num_samples += batch_size
        
        # Tính các chỉ số cho từng ảnh trong batch
        for j in range(batch_size):
            pred_img = predicted_ct_np[j, :, :, 0]  # Lấy channel đầu tiên
            real_img = real_ct_np[j, :, :, 0]
            
            # Lưu ảnh nếu cần
            if save_dir is not None and i < 10:  # Chỉ lưu 10 batch đầu tiên
                mr_img = mr_batch[j].numpy()[:, :, 0]
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(mr_img, cmap='gray')
                axes[0].set_title('Input MR')
                axes[0].axis('off')
                
                axes[1].imshow(pred_img, cmap='gray')
                axes[1].set_title('Predicted CT')
                axes[1].axis('off')
                
                axes[2].imshow(real_img, cmap='gray')
                axes[2].set_title('Real CT')
                axes[2].axis('off')
                
                plt.savefig(os.path.join(save_dir, f'sample_{i}_{j}.png'))
                plt.close(fig)
            
            # Tính Mean Error (ME)
            me = np.mean(pred_img - real_img)
            metrics['ME'].append(me)
            
            # Tính Mean Absolute Error (MAE)
            mae = np.mean(np.abs(pred_img - real_img))
            metrics['MAE'].append(mae)
            
            # Tính Peak Signal-to-Noise Ratio (PSNR)
            mse = np.mean((pred_img - real_img) ** 2)
            if mse == 0:  # Tránh chia cho 0
                psnr = 100
            else:
                max_pixel = 1.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            metrics['PSNR'].append(psnr)
            
            # Tính Structural Similarity Index (SSIM)
            ssim = tf.image.ssim(
                tf.convert_to_tensor(pred_img[np.newaxis, :, :, np.newaxis], dtype=tf.float32),
                tf.convert_to_tensor(real_img[np.newaxis, :, :, np.newaxis], dtype=tf.float32),
                max_val=1.0
            ).numpy()[0]
            metrics['SSIM'].append(ssim)
            
            # Tính Normalized Cross-Correlation (NCC)
            # NCC = sum((I1-μ1)*(I2-μ2))/(sqrt(sum((I1-μ1)^2)*sum((I2-μ2)^2)))
            pred_flat = pred_img.flatten()
            real_flat = real_img.flatten()
            
            pred_mean = np.mean(pred_flat)
            real_mean = np.mean(real_flat)
            
            numerator = np.sum((pred_flat - pred_mean) * (real_flat - real_mean))
            denominator = np.sqrt(np.sum((pred_flat - pred_mean)**2) * np.sum((real_flat - real_mean)**2))
            
            if denominator == 0:  # Tránh chia cho 0
                ncc = 0
            else:
                ncc = numerator / denominator
            metrics['NCC'].append(ncc)
            
            # Tính Dice Similarity Coefficient (DSC)
            # Chuyển ảnh về binary
            threshold = 0.5  # Ngưỡng để phân loại
            pred_binary = (pred_img > threshold).astype(np.float32)
            real_binary = (real_img > threshold).astype(np.float32)
            
            intersection = np.sum(pred_binary * real_binary)
            dice = (2. * intersection) / (np.sum(pred_binary) + np.sum(real_binary) + 1e-7)
            metrics['DSC'].append(dice)
    
    # Tính trung bình của các chỉ số
    result = {}
    for metric_name, values in metrics.items():
        result[metric_name] = np.mean(values)
        
    return result

def train(generator, discriminator, train_dataset, val_dataset, config, checkpoint_path=None):
    """Train the GAN model"""
    # Create optimizers
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        beta_1=config['beta1']
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        beta_1=config['beta1']
    )
    
    # Create learning rate schedulers for learning rate decay
    # Exponential decay when approaching overfitting
    lr_decay_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],
        decay_steps=1000,  # Apply decay every 1000 steps
        decay_rate=0.9,    # Reduce by 10% each time
        staircase=True
    )
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience = 10  # Wait for 10 epochs without improvement before stopping
    patience_counter = 0
    best_epoch = 0
    
    # Create checkpoint manager
    checkpoint_dir = os.path.join(config['checkpoint_dir'], 'training_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=3
    )
    
    # Khôi phục từ checkpoint nếu có
    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()
        print(f"Restored from checkpoint: {checkpoint_path}")

    # Create summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(config['log_dir'], 'train', current_time)
    val_log_dir = os.path.join(config['log_dir'], 'validation', current_time)
    
    # Create log directories
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    
    # Create summary writers
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # Set up the matplotlib figure for visualizing loss
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Store loss values for plotting
    epochs_list = []
    gan_losses_avg = []
    l1_losses_avg = []
    perceptual_losses_avg = []
    hist_losses_avg = []
    
    # Store loss values for current epoch
    current_gan_losses = []
    current_l1_losses = []
    current_perceptual_losses = []
    current_hist_losses = []
    
    # Function to update the plot
    def update_plot():
        """Update loss plot"""
        plt.figure(fig.number)  # Đảm bảo đang vẽ trên figure đúng
        ax.clear()
        
        ax.plot(epochs_list, gan_losses_avg, 'r-', label='GAN Loss')
        ax.plot(epochs_list, l1_losses_avg, 'g-', label='L1 Loss')
        ax.plot(epochs_list, perceptual_losses_avg, 'b-', label='Perceptual Loss')
        ax.plot(epochs_list, hist_losses_avg, 'y-', label='Hist Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True)
        
        # Add early stopping marker
        if best_epoch > 0:
            ax.axvline(x=best_epoch, color='m', linestyle='--', alpha=0.5)
            ax.text(best_epoch, ax.get_ylim()[1]*0.9, f'Best: {best_epoch}', color='m')
        
        plt.draw()
        plt.pause(0.001)
        fig.canvas.flush_events()  # Cập nhật canvas
        
        # Lưu biểu đồ
        plt.savefig(os.path.join(config['log_dir'], 'loss_plot.png'))
    
    @tf.function
    def train_step(mr_image, ct_image, generator, discriminator, generator_optimizer, discriminator_optimizer, config):
        """Single training step, returns detailed losses"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Ensure both tensors are float32
            mr_image = tf.cast(mr_image, tf.float32)
            ct_image = tf.cast(ct_image, tf.float32)

            # Generate synthetic CT
            generated_ct = generator(mr_image, training=True)

            # Concatenate MR with real/fake CT for discriminator
            real_input = tf.concat([mr_image, ct_image], axis=-1)
            fake_input = tf.concat([mr_image, generated_ct], axis=-1)

            # Get discriminator outputs
            real_output = discriminator(real_input, training=True)
            fake_output = discriminator(fake_input, training=True)

            # Calculate losses
            total_gen_loss, gan_loss, l1_loss, perceptual_loss, hist_loss = generator_loss(fake_output, generated_ct, ct_image, config)
            disc_loss = discriminator_loss(real_output, fake_output)

        # Calculate gradients
        gen_gradients = gen_tape.gradient(
            total_gen_loss, generator.trainable_variables
        )
        disc_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        # Apply gradients
        generator_optimizer.apply_gradients(
            zip(gen_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(disc_gradients, discriminator.trainable_variables)
        )

        return total_gen_loss, disc_loss, gan_loss, l1_loss, perceptual_loss, hist_loss

    @tf.function
    def val_step(mr_image, ct_image, generator, discriminator, config):
        """Single validation step, returns detailed losses"""
        # Ensure both tensors are float32
        mr_image = tf.cast(mr_image, tf.float32)
        ct_image = tf.cast(ct_image, tf.float32)

        # Generate synthetic CT
        generated_ct = generator(mr_image, training=False)

        # Concatenate MR with real/fake CT for discriminator
        real_input = tf.concat([mr_image, ct_image], axis=-1)
        fake_input = tf.concat([mr_image, generated_ct], axis=-1)

        # Get discriminator outputs
        real_output = discriminator(real_input, training=False)
        fake_output = discriminator(fake_input, training=False)

        # Calculate losses
        total_gen_loss, gan_loss, l1_loss, perceptual_loss, hist_loss = generator_loss(fake_output, generated_ct, ct_image, config)
        disc_loss = discriminator_loss(real_output, fake_output)

        return total_gen_loss, disc_loss, gan_loss, l1_loss, perceptual_loss, hist_loss

    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Reset the loss lists for this epoch
        current_gan_losses = []
        current_l1_losses = []
        current_perceptual_losses = []
        current_hist_losses = []
        
        # Apply learning rate decay if we're starting to overfit
        if patience_counter > 3:
            # Reduce learning rate if we've seen no improvement for a few epochs
            current_lr = config['learning_rate'] * (0.7 ** patience_counter)
            print(f"Reducing learning rate to {current_lr:.2e} due to lack of improvement")
            generator_optimizer.lr.assign(current_lr)
            discriminator_optimizer.lr.assign(current_lr)
        
        # Training
        for step, (mr_batch, ct_batch) in enumerate(train_dataset):
            total_gen_loss, disc_loss, gan_loss, l1_loss, perceptual_loss, hist_loss = train_step(
                mr_batch, ct_batch, generator, discriminator, 
                generator_optimizer, discriminator_optimizer, config
            )
            
            # Convert losses to numpy for plotting
            gan_loss_np = gan_loss.numpy()
            l1_loss_np = l1_loss.numpy()
            perceptual_loss_np = perceptual_loss.numpy()
            hist_loss_np = hist_loss.numpy()
            
            # Store current epoch's losses
            current_gan_losses.append(gan_loss_np)
            current_l1_losses.append(l1_loss_np)
            current_perceptual_losses.append(perceptual_loss_np)
            current_hist_losses.append(hist_loss_np)
            
            # Calculate global step for TensorBoard
            global_step = epoch * len(list(train_dataset)) + step
            
            if step % 100 == 0:
                print(f"Step {step}: Gen Loss = {total_gen_loss:.4f}, Disc Loss = {disc_loss:.4f}")
                print(f"Components: GAN = {gan_loss_np:.4f}, L1 = {l1_loss_np:.4f}, Perceptual = {perceptual_loss_np:.4f}, Hist = {hist_loss_np:.4f}")
                
                with train_summary_writer.as_default():
                    tf.summary.scalar('generator_loss', total_gen_loss, step=global_step)
                    tf.summary.scalar('discriminator_loss', disc_loss, step=global_step)
                    tf.summary.scalar('gan_loss', gan_loss, step=global_step)
                    tf.summary.scalar('l1_loss', l1_loss, step=global_step)
                    tf.summary.scalar('perceptual_loss', perceptual_loss, step=global_step)
                    tf.summary.scalar('hist_loss', hist_loss, step=global_step)
        
        # Validation
        val_gen_losses = []
        val_disc_losses = []
        val_gan_losses = []
        val_l1_losses = []
        val_perceptual_losses = []
        val_hist_losses = []
        
        for mr_batch, ct_batch in val_dataset:
            val_total_gen_loss, val_disc_loss, val_gan_loss, val_l1_loss, val_perceptual_loss, val_hist_loss = val_step(
                mr_batch, ct_batch, generator, discriminator, config
            )
            val_gen_losses.append(val_total_gen_loss)
            val_disc_losses.append(val_disc_loss)
            val_gan_losses.append(val_gan_loss)
            val_l1_losses.append(val_l1_loss)
            val_perceptual_losses.append(val_perceptual_loss)
            val_hist_losses.append(val_hist_loss)
        
        # Calculate mean validation losses
        val_total_gen_loss = tf.reduce_mean(val_gen_losses)
        val_disc_loss = tf.reduce_mean(val_disc_losses)
        val_gan_loss = tf.reduce_mean(val_gan_losses)
        val_l1_loss = tf.reduce_mean(val_l1_losses)
        val_perceptual_loss = tf.reduce_mean(val_perceptual_losses)
        val_hist_loss = tf.reduce_mean(val_hist_losses)
        
        # Log validation results
        print(f"Validation: Gen Loss = {val_total_gen_loss:.4f}, Disc Loss = {val_disc_loss:.4f}")
        print(f"Components: GAN = {val_gan_loss:.4f}, L1 = {val_l1_loss:.4f}, Perceptual = {val_perceptual_loss:.4f}, Hist = {val_hist_loss:.4f}")
        
        with val_summary_writer.as_default():
            tf.summary.scalar('generator_loss', val_total_gen_loss, step=epoch)
            tf.summary.scalar('discriminator_loss', val_disc_loss, step=epoch)
            tf.summary.scalar('gan_loss', val_gan_loss, step=epoch)
            tf.summary.scalar('l1_loss', val_l1_loss, step=epoch)
            tf.summary.scalar('perceptual_loss', val_perceptual_loss, step=epoch)
            tf.summary.scalar('hist_loss', val_hist_loss, step=epoch)
        
        # Calculate averages for the epoch and update plot
        epochs_list.append(epoch + 1)  # Add 1 to make it 1-indexed
        gan_losses_avg.append(np.mean(current_gan_losses))
        l1_losses_avg.append(np.mean(current_l1_losses))
        perceptual_losses_avg.append(np.mean(current_perceptual_losses))
        hist_losses_avg.append(np.mean(current_hist_losses))
        
        # Update plot after each epoch
        update_plot()
        
        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for epoch {epoch+1} at {save_path}")
            
            # Also save a separate checkpoint for easy loading
            model_save_path = os.path.join(config['checkpoint_dir'], f'generator_epoch_{epoch+1}.keras')  # Added .keras extension
            generator.save(model_save_path)
            print(f"Saved generator model for epoch {epoch+1}")
        
        # Early stopping logic
        if val_total_gen_loss < best_val_loss:
            best_val_loss = val_total_gen_loss
            patience_counter = 0
            best_epoch = epoch + 1
            
            # Save best model
            best_model_path = os.path.join(config['checkpoint_dir'], 'generator_best.keras')
            generator.save(best_model_path)
            print(f"New best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load and return the best model if available
    best_model_path = os.path.join(config['checkpoint_dir'], 'generator_best.keras')
    if os.path.exists(best_model_path):
        print(f"Loading best model from epoch {best_epoch}")
        best_generator = tf.keras.models.load_model(best_model_path)
        generator = best_generator
    
    # Save final model
    generator.save(os.path.join(config['checkpoint_dir'], 'generator_final.keras'))  # Added .keras extension
    print("Training complete. Final model saved.")
    
    # Đánh giá mô hình sau khi huấn luyện
    print("\nEvaluating model on validation data...")
    eval_results = evaluate_model(
        generator, 
        val_dataset, 
        save_dir=os.path.join(config['log_dir'], 'validation_samples')
    )
    
    # In kết quả đánh giá
    print("\nEvaluation Metrics:")
    print(f"Mean Error (ME): {eval_results['ME']:.4f} (Ideal: 0)")
    print(f"Mean Absolute Error (MAE): {eval_results['MAE']:.4f} (Ideal: 0)")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {eval_results['PSNR']:.4f} dB (Higher is better)")
    print(f"Structural Similarity Index (SSIM): {eval_results['SSIM']:.4f} (Ideal: 1)")
    print(f"Normalized Cross-Correlation (NCC): {eval_results['NCC']:.4f} (Ideal: 1 for perfect match)")
    print(f"Dice Similarity Coefficient (DSC): {eval_results['DSC']:.4f} (Ideal: 1)")
    
    # Lưu kết quả đánh giá vào file
    with open(os.path.join(config['log_dir'], 'evaluation_results.txt'), 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Mean Error (ME): {eval_results['ME']:.4f} (Ideal: 0)\n")
        f.write(f"Mean Absolute Error (MAE): {eval_results['MAE']:.4f} (Ideal: 0)\n")
        f.write(f"Peak Signal-to-Noise Ratio (PSNR): {eval_results['PSNR']:.4f} dB (Higher is better)\n")
        f.write(f"Structural Similarity Index (SSIM): {eval_results['SSIM']:.4f} (Ideal: 1)\n")
        f.write(f"Normalized Cross-Correlation (NCC): {eval_results['NCC']:.4f} (Ideal: 1 for perfect match)\n")
        f.write(f"Dice Similarity Coefficient (DSC): {eval_results['DSC']:.4f} (Ideal: 1)\n")
    
    return eval_results

def generator_loss(fake_output, generated_ct, target_ct, config):
    """Calculate generator loss and its components"""
    # GAN loss (adversarial loss - binary cross-entropy)
    gan_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            tf.ones_like(fake_output), fake_output
        )
    )

    # L1 loss (MAE - Mean Absolute Error)
    # Better for preserving details and bone structures
    l1_loss = tf.reduce_mean(tf.abs(target_ct - generated_ct))

    # Calculate perceptual loss using VGG-based features
    # This helps prevent over-smoothing and preserves structural details
    # Simplified version using gradient magnitude to detect edges and structure
    # This approximates perceptual loss without needing a pre-trained VGG
    gen_gradients = tf.image.image_gradients(generated_ct)
    target_gradients = tf.image.image_gradients(target_ct)
    
    # Calculate gradient differences (captures structure preservation)
    grad_diff_x = tf.reduce_mean(tf.abs(gen_gradients[0] - target_gradients[0]))
    grad_diff_y = tf.reduce_mean(tf.abs(gen_gradients[1] - target_gradients[1]))
    perceptual_loss = grad_diff_x + grad_diff_y

    # Total loss with weights based on best practices from Pix2Pix and cGAN papers
    # Higher weight on L1 for bone and tissue preservation in brain images
    total_gen_loss = (
        0.5 * gan_loss +           # Reduce GAN loss weight to prevent mode collapse
        config['lambda_l1'] * l1_loss +  # High weight on L1 for detail preservation
        5.0 * perceptual_loss    # Moderate weight on perceptual loss
    )

    return total_gen_loss, gan_loss, l1_loss, perceptual_loss, 0.0  # Keep 5 return values for compatibility

def discriminator_loss(real_output, fake_output):
    """Calculate discriminator loss"""
    # Real loss
    real_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            tf.ones_like(real_output), real_output
        )
    )

    # Fake loss
    fake_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            tf.zeros_like(fake_output), fake_output
        )
    )

    # Balance the discriminator loss
    total_disc_loss = real_loss + fake_loss

    return total_disc_loss

def main():
    # Configuration
    config = {
        'batch_size': 8,
        'epochs': 50,  # Increased from 10 to 50 to see the full learning curve
        'learning_rate': 2e-5,  # Slightly reduced learning rate for stability
        'beta1': 0.5,
        'lambda_l1': 100,  # Increased from 80 to 100 to focus more on pixel-level accuracy
        'save_freq': 1,
        'checkpoint_dir': "/content/my_folder/Brain - Copy/checkpoints",
        'log_dir': "/content/my_folder/Brain - Copy/logs",
        'l2_reg': 1e-5  # Added L2 regularization factor
    }

    # Create directories for logs and checkpoints if they do not exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Create subdirectories for logs
    train_base = os.path.join(config['log_dir'], 'train')
    val_base = os.path.join(config['log_dir'], 'validation')
    os.makedirs(train_base, exist_ok=True)
    os.makedirs(val_base, exist_ok=True)

    # Prepare data paths
    current_dir = os.getcwd()
    train_dir = os.path.join(current_dir, "my_folder", "Brain - Copy", "Data", "splits", "train")
    val_dir = os.path.join(current_dir, "my_folder", "Brain - Copy", "Data", "splits", "val")

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError(f"Data directories not found: {train_dir} or {val_dir}")

    # Prepare datasets
    train_dataset, val_dataset = prepare_data(train_dir, val_dir, batch_size=config['batch_size'])

    # Load an existing model or create a new one
    if os.path.exists('/content/drive/MyDrive/sCT from MRI/model_saved/generator_epoch_10.keras'):
        print("Loading pre-trained generator model...")
        generator = tf.keras.models.load_model('/content/drive/MyDrive/sCT from MRI/model_saved/generator_epoch_10.keras')
    else:
        print("Creating new generator model...")
        # You need to import or define your Generator class here
        from models import Generator
        generator = Generator(l2_reg=config['l2_reg'])  # Pass L2 regularization

    # Create a new discriminator or load an existing one
    from models import Discriminator
    discriminator = Discriminator(l2_reg=config['l2_reg'])  # Pass L2 regularization

    # Optimize with new training parameters
    train(generator, discriminator, train_dataset, val_dataset, config, checkpoint_path=None)

if __name__ == "__main__":
    main()

