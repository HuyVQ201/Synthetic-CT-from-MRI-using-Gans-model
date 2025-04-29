# Hướng dẫn Huấn luyện MRI-to-CT

## 1. Chuẩn bị Môi trường

### 1.1. Cài đặt Python và các thư viện cần thiết
```bash
conda create -n mri2ct python=3.9
conda activate mri2ct
pip install tensorflow==2.12.0
pip install tensorflow-gpu==2.12.0
pip install nibabel
pip install scipy
pip install matplotlib
pip install opencv-python
pip install tqdm
```

### 1.2. Cấu trúc thư mục
```
MRI-to-CT-DCNN-TensorFlow/
├── Data/
│   ├── raw/          # Dữ liệu gốc sau khi tách lát cắt
│   │   ├── MR/      # Ảnh MRI
│   │   └── CT/      # Ảnh CT
│   └── Preprocessed/ # Dữ liệu sau tiền xử lý
├── Preprocessing 2/
│   ├── Split Slice to Preprocess  # Script tách lát cắt
│   └── Preprocess_OK              # Script tiền xử lý
├── Model/
│   ├── generator.py
│   └── discriminator.py
├── train.py
└── test.py
```

## 2. Quy trình Huấn luyện

### 2.1. Tách lát cắt dữ liệu
Chạy script `Split Slice to Preprocess` để tách các file NIfTI thành các lát cắt riêng lẻ:
```bash
python "Preprocessing 2/Split Slice to Preprocess"
```
Script này sẽ:
- Đọc các file NIfTI từ thư mục Image Train và Image Label
- Tách mỗi file thành các lát cắt riêng lẻ
- Lưu các lát cắt dưới dạng file .npy vào thư mục Data/raw

### 2.2. Tiền xử lý dữ liệu
Chạy script `Preprocess_OK` để tiền xử lý dữ liệu:
```bash
python "Preprocessing 2/Preprocess_OK"
```
Script này sẽ thực hiện các bước:
- N4 Bias Correction
- Histogram Matching
- Tạo mask vùng đầu
- Lưu kết quả vào thư mục Data/Preprocessed

### 2.3. Huấn luyện mô hình
Chạy script huấn luyện:
```bash
python train.py
```
Quá trình huấn luyện sẽ:
- Tải dữ liệu đã tiền xử lý
- Huấn luyện generator và discriminator
- Lưu các checkpoint định kỳ
- Hiển thị kết quả trung gian

### 2.4. Đánh giá mô hình
Chạy script đánh giá:
```bash
python test.py
```
Script này sẽ:
- Tải mô hình đã huấn luyện
- Tạo ảnh CT mô phỏng từ ảnh MRI
- Tính toán các chỉ số đánh giá (MAE, MSE, PSNR, SSIM)
- Hiển thị kết quả

## 3. Cấu hình Huấn luyện

### 3.1. Tham số Huấn luyện
- Batch size: 1
- Số epochs: 200
- Learning rate: 2e-4
- Loss weights:
  - L1 loss: 100
  - GAN loss: 1
- Optimizer: Adam

### 3.2. Kiến trúc Mô hình
- Generator: U-Net với skip connections
- Discriminator: PatchGAN

## 4. Lưu ý
- Đảm bảo có đủ dung lượng ổ đĩa cho dữ liệu và checkpoint
- Kiểm tra GPU memory trước khi huấn luyện
- Theo dõi quá trình huấn luyện qua tensorboard

## Kiểm tra điểm cải thiện

Sau đây là một số điểm cần kiểm tra nếu quá trình không hoạt động tốt:

1. **Lỗi trong tiền xử lý dữ liệu**:
   - Kiểm tra kích thước ảnh đầu vào và đầu ra
   - Xác nhận định dạng dữ liệu (npy)
   - Kiểm tra phạm vi giá trị pixel (normalization)

2. **Lỗi trong quá trình huấn luyện**:
   - Nếu G Loss không giảm: Thử giảm learning rate
   - Nếu D Loss quá nhỏ: Có thể cần điều chỉnh cân bằng giữa Generator và Discriminator
   - Nếu có lỗi out-of-memory: Giảm batch_size

3. **Lỗi khi chạy test**:
   - Kiểm tra lại định dạng dữ liệu đầu vào và normalization
   - Đảm bảo kích thước của ảnh test khớp với kích thước ảnh huấn luyện

## Lưu ý

1. **Dữ liệu**:
   - Tỉ lệ train-validation split mặc định là 80/20
   - Đảm bảo dữ liệu MRI và CT được căn chỉnh (aligned) chính xác

2. **Huấn luyện**:
   - Quá trình huấn luyện GAN có thể không ổn định, bạn nên lưu checkpoint thường xuyên
   - Tăng số lượng epochs có thể cải thiện kết quả, nhưng cũng có thể dẫn đến overfitting

3. **Đánh giá**:
   - Ngoài việc đánh giá trực quan, bạn nên sử dụng các metrics khác như PSNR, SSIM nếu có ground truth CT

4. **Môi trường**:
   - Đảm bảo bạn có đủ VRAM nếu sử dụng GPU
   - Tiền xử lý có thể tốn nhiều thời gian với dữ liệu lớn 

# Training Pipeline Documentation

## Data Processing Pipeline

1. **Data Preprocessing** (`Preprocessing 2/Preprocess_OK`)
   - Normalizes MR and CT images
   - Applies bias field correction to MR images
   - Creates head masks
   - Saves preprocessed data to `Data/Preprocessed/`

2. **Data Augmentation** (`Data Augment/data_augmentation.py`)
   - Takes preprocessed data as input
   - Applies multiple augmentation techniques:
     * Rotation (90°, 180°, 270°)
     * Zoom in/out (±10%)
     * Horizontal flipping
     * Contrast adjustment (±20%)
   - Saves augmented data to `Data/Augmented/`
   - Increases dataset size by approximately 7x

3. **Data Loading** (`dataloader.py`)
   - Loads augmented data from `Data/Augmented/`
   - Creates training and validation splits
   - Applies any final preprocessing needed for model input

## Running the Pipeline

1. First, preprocess the raw data:
```bash
python Preprocessing\ 2/Preprocess_OK
```

2. Then, apply data augmentation:
```bash
python Data\ Augment/data_augmentation.py
```

3. Finally, run the training pipeline:
```bash
python run_pipeline.py
```

## Data Augmentation Parameters

You can customize the augmentation parameters in `data_augmentation.py`:

```python
augmentation_params = {
    'angles': [90, 180, 270],          # Rotation angles
    'zoom_factors': [0.9, 1.1],        # Zoom in/out factors
    'do_flip': True,                   # Whether to include flipped versions
    'contrast_factors': [0.8, 1.2]     # Contrast adjustment factors
}
```

## Directory Structure

```
.
├── Data/
│   ├── raw/              # Raw MR and CT data
│   ├── Preprocessed/     # Preprocessed data
│   └── Augmented/        # Augmented data
├── Preprocessing 2/
│   └── Preprocess_OK     # Preprocessing script
├── Data Augment/
│   └── data_augmentation.py  # Data augmentation script
└── src/
    └── dataloader.py     # Data loading and batching
```

## Notes

- The augmentation process preserves the alignment between MR, CT, and mask images
- All augmentations are applied in-memory before saving
- The process automatically handles edge cases and padding
- Progress bars show augmentation status
- Original data is always included in the augmented dataset 