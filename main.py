# ---------------------------------------------------------
# Tensorflow DCNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime

from dataset import Dataset
from solver import Solver
from gan_model import GANModel
from utils import make_folders, init_logger
from Model import build_model
from src.models import Generator, Discriminator
from src.train import train, prepare_data, evaluate_model

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--gpu_index', dest='gpu_index', default='0', help='gpu index if you have multiple gpus, default: 0')
parser.add_argument('--is_train', dest='is_train', default=False, action='store_true', help='training or test mode, default: False (test mode)')
parser.add_argument('--batch_size', dest='batch_size', default=8, type=int, help='batch size for one iteration, default: 8')
parser.add_argument('--dataset', dest='dataset', default='brain01', help='dataset name, default: brain01')
parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3, type=float, help='learning rate, default: 2e-4')
parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float, help='weight decay, default: 1e-5')
parser.add_argument('--epoch', dest='epoch', default=600, type=int, help='number of epochs, default: 600')
parser.add_argument('--print_freq', dest='print_freq', default=100, type=int, help='print frequency for loss information, default: 100')
parser.add_argument('--load_model', dest='load_model', default=None, help='folder of saved model that you wish to continue training, (e.g., 20190411-2217), default: None')
parser.add_argument('--use_nifti', dest='use_nifti', default=True, action='store_true', help='use NIFTI data format, default: True')
parser.add_argument('--num_cross_vals', type=int, default=5, help='Number of cross-validation folds')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                   help='Mode: train or evaluate')
parser.add_argument('--train_dir', type=str, default='./Data/splits/train',
                   help='Directory containing training data')
parser.add_argument('--val_dir', type=str, default='./Data/splits/val',
                   help='Directory containing validation data')
parser.add_argument('--epochs', type=int, default=100,
                   help='Number of epochs for training')
parser.add_argument('--beta1', type=float, default=0.5,
                   help='Beta1 parameter for Adam optimizer')
parser.add_argument('--lambda_l1', type=float, default=100,
                   help='Weight for L1 loss')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                   help='Directory to save model checkpoints')
parser.add_argument('--log_dir', type=str, default='./logs',
                   help='Directory to save training logs')
parser.add_argument('--save_freq', type=int, default=1,
                   help='Save checkpoint frequency (epochs)')
parser.add_argument('--model_path', type=str, default=None,
                   help='Path to pretrained model for evaluation')
args = parser.parse_args()

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


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


def make_folders(is_train=True, load_model=None, dataset=None):
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


def main():
    # Parse arguments
    args = parse_args()
    
    # Create config dictionary
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'beta1': args.beta1,
        'lambda_l1': args.lambda_l1,
        'save_freq': args.save_freq,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir
    }
    
    # Create directories for logs and checkpoints if they do not exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Create subdirectories for logs
    train_base = os.path.join(config['log_dir'], 'train')
    val_base = os.path.join(config['log_dir'], 'validation')
    os.makedirs(train_base, exist_ok=True)
    os.makedirs(val_base, exist_ok=True)
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(args.train_dir, args.val_dir, batch_size=config['batch_size'])
    
    if args.mode == 'train':
        # Create models
        generator = Generator()
        discriminator = Discriminator()
        
        # Train model
        train(generator, discriminator, train_dataset, val_dataset, config)
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluation mode")
        
        print(f"Loading model from {args.model_path}")
        generator = tf.keras.models.load_model(args.model_path)
        
        # Đánh giá mô hình
        print("\nEvaluating model on validation data...")
        save_dir = os.path.join(config['log_dir'], 'evaluation_samples')
        eval_results = evaluate_model(generator, val_dataset, save_dir=save_dir)
        
        # In kết quả đánh giá
        print("\nEvaluation Metrics:")
        print(f"Mean Error (ME): {eval_results['ME']:.4f} (Ideal: 0)")
        print(f"Mean Absolute Error (MAE): {eval_results['MAE']:.4f} (Ideal: 0)")
        print(f"Peak Signal-to-Noise Ratio (PSNR): {eval_results['PSNR']:.4f} dB (Higher is better)")
        print(f"Structural Similarity Index (SSIM): {eval_results['SSIM']:.4f} (Ideal: 1)")
        print(f"Normalized Cross-Correlation (NCC): {eval_results['NCC']:.4f} (Ideal: 1 for perfect match)")
        print(f"Dice Similarity Coefficient (DSC): {eval_results['DSC']:.4f} (Ideal: 1)")
        
        # Lưu kết quả đánh giá vào file
        eval_file = os.path.join(config['log_dir'], 'evaluation_results.txt')
        with open(eval_file, 'w') as f:
            f.write("Evaluation Metrics:\n")
            f.write(f"Mean Error (ME): {eval_results['ME']:.4f} (Ideal: 0)\n")
            f.write(f"Mean Absolute Error (MAE): {eval_results['MAE']:.4f} (Ideal: 0)\n")
            f.write(f"Peak Signal-to-Noise Ratio (PSNR): {eval_results['PSNR']:.4f} dB (Higher is better)\n")
            f.write(f"Structural Similarity Index (SSIM): {eval_results['SSIM']:.4f} (Ideal: 1)\n")
            f.write(f"Normalized Cross-Correlation (NCC): {eval_results['NCC']:.4f} (Ideal: 1 for perfect match)\n")
            f.write(f"Dice Similarity Coefficient (DSC): {eval_results['DSC']:.4f} (Ideal: 1)\n")
        
        print(f"Evaluation results saved to {eval_file}")
        print(f"Sample images saved to {save_dir}")


def train(num_cross_vals, model_dir, sample_dir, log_dir, solver):
    for model_id in range(num_cross_vals):
        model_sub_dir = os.path.join(model_dir, 'model' + str(model_id))
        sample_sub_dir = os.path.join(sample_dir, 'model' + str(model_id))
        log_sub_dir = os.path.join(log_dir, 'model' + str(model_id))

        if not os.path.isdir(model_sub_dir):
            os.makedirs(model_sub_dir)

        if not os.path.isdir(sample_sub_dir):
            os.makedirs(sample_sub_dir)

        if not os.path.isdir(log_sub_dir):
            os.makedirs(log_sub_dir)

        saver = tf.train.Saver(max_to_keep=1)
        tb_writer = tf.summary.FileWriter(log_sub_dir, graph_def=solver.sess.graph_def)
        data = Dataset(args.dataset, num_cross_vals, model_id, use_nifti=args.use_nifti)
        solver.init()  # initialize model weights
        best_mae = sys.float_info.max

        epoch_time = 0
        num_iters = int(args.epoch * data.num_train / args.batch_size)
        for iter_time in range(num_iters):
            mrImgs, ctImgs, maskImgs = data.train_batch(batch_size=args.batch_size)
            _, total_loss, data_loss, reg_term, mrImgs_, preds, ctImgs_, summary = solver.train(mrImgs, ctImgs)
            tb_writer.add_summary(summary, iter_time)
            tb_writer.flush()

            if np.mod(iter_time, args.print_freq) == 0:
                print('Model id: {}, {} / {} Total Loss: {:.3f}, Data Loss: {:.3f}, Reg Term: {:.3f}'.format(
                    model_id, iter_time, num_iters, total_loss, data_loss, reg_term))

            if (np.mod(iter_time + 1, int(data.num_train / args.batch_size)) == 0) or (iter_time + 1 == num_iters):
                epoch_time += 1

                mrImgs, ctImgs, maskImgs = data.val_batch()
                preds = solver.test(mrImgs, batch_size=args.batch_size)
                mae, summary = solver.evaluate(ctImgs, preds, maskImgs, is_train=True)
                print('Epoch: {}, MAE: {:.3f}, Best MAE: {:.3f}'.format(epoch_time, mae, best_mae))

                # write to tensorbaord
                tb_writer.add_summary(summary, epoch_time)

                # Save validation results
                solver.save_imgs(mrImgs, ctImgs, preds, maskImgs, iter_time, save_folder=sample_sub_dir)

                if mae < best_mae:
                    best_mae = mae
                    save_model(saver, solver, model_sub_dir, model_id, iter_time)


def test(num_cross_vals, model_dir, test_dir, solver):
    mae = np.zeros(num_cross_vals, dtype=np.float32)    # Mean Absolute Error
    me = np.zeros(num_cross_vals, dtype=np.float32)     # Mean Error
    mse = np.zeros(num_cross_vals, dtype=np.float32)    # Mean Squared Error
    pcc = np.zeros(num_cross_vals, dtype=np.float32)    # Pearson Correlation Coefficient

    for model_id in range(num_cross_vals):
        model_sub_dir = os.path.join(model_dir, 'model' + str(model_id))
        test_sub_dir = os.path.join(test_dir, 'model' + str(model_id))
        if not os.path.isdir(test_sub_dir):
            os.makedirs(test_sub_dir)

        data = Dataset(args.dataset, num_cross_vals, model_id, use_nifti=args.use_nifti)

        saver = tf.train.Saver(max_to_keep=1)
        solver.init()
        if restore_model(saver, solver, model_sub_dir, model_id):  # Restore models
            logger.info(' [*] Load model ID: {} SUCCESS!'.format(model_id))
        else:
            logger.info(' [!] Load model ID: {} Failed...'.format(model_id))
            sys.exit(' [!] Cannot find checkpoint...')

        mrImgs, ctImgs, maskImgs = data.test_batch()
        preds = solver.test(mrImgs, batch_size=args.batch_size)
        mae[model_id], me[model_id], mse[model_id], pcc[model_id]  = solver.evaluate(ctImgs, preds, maskImgs, is_train=False)

        # save imgs
        solver.save_imgs(mrImgs, ctImgs, preds, maskImgs, save_folder=test_sub_dir)

    for model_id in range(num_cross_vals):
        print('Model ID: {} - MAE: {:.3f}, ME: {:.3f}, MSE: {:.3f}, PCC: {:.3f}'.format(
            model_id, mae[model_id], me[model_id], mse[model_id], pcc[model_id]))

    print('Avearge MAE: {:.3f}'.format(np.mean(mae)))
    print('Average ME: {:.3f}'.format(np.mean(me)))
    print('Average MSE: {:.3f}'.format(np.mean(mse)))
    print('Average PCC: {:.3f}'.format(np.mean(pcc)))

    bar_plot(num_cross_vals, mae, me, mse, pcc, names=['MAE', 'ME', 'MSE', 'PCC'], save_folder=test_dir)


def bar_plot(num_cross_vals, mae, me, mse, pcc, names, save_folder):
    y_pos = np.arange(num_cross_vals+1)

    measures = np.zeros((len(names), num_cross_vals+1), dtype=np.float32)
    measures[0, :-1], measures[0, -1] = mae, np.mean(mae)
    measures[1, :-1], measures[1, -1] = me, np.mean(me)
    measures[2, :-1], measures[2, -1] = mse, np.mean(mse)
    measures[3, :-1], measures[3, -1] = pcc, np.mean(pcc)

    for i in range(len(names)):
        performance = measures[i, :]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, ['Model_0', 'Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5', 'Average'])
        plt.ylabel(names[i])
        plt.savefig(os.path.join(save_folder, names[i] + '.png'), dpi=300)
        plt.close()


def save_model(saver, solver, model_dir, model_id, iter_time):
    saver.save(solver.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
    logger.info(' [*] Model saved! Model ID: {}, Iter: {}'.format(model_id, iter_time))


def restore_model(saver, solver, model_dir, model_id):
    logger.info(' [*] Reading model: {} checkpoint...'.format(model_id))

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(solver.sess, os.path.join(model_dir, ckpt_name))
        return True
    else:
        return False


def train_model(model, train_dataset, val_dataset, args):
    """Train the model with cross-validation"""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join('logs', args.dataset, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, 'model_epoch_{epoch:02d}.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,  # Set to 30 epochs
        callbacks=[checkpoint, tensorboard],
        verbose=1
    )
    
    return history


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MRI to CT Conversion with GAN')
    
    # Thêm tham số
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    parser.add_argument('--train_dir', type=str, default='./Data/splits/train',
                       help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default='./Data/splits/val',
                       help='Directory containing validation data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--lambda_l1', type=float, default=100,
                       help='Weight for L1 loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save training logs')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Save checkpoint frequency (epochs)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model for evaluation')
    
    return parser.parse_args()


if __name__ == '__main__':
    main()
