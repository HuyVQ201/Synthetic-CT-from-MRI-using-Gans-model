# Tổng quan quy trình MRI-to-CT

## Mục tiêu

Quy trình này nhằm chuyển đổi ảnh MRI sang ảnh CT mô phỏng (Synthetic CT) sử dụng mô hình Deep Convolutional Neural Network (DCNN) dựa trên kiến trúc GAN (Generative Adversarial Network).

## Dữ liệu

Dữ liệu đầu vào bao gồm:
- Ảnh MRI não
- Ảnh CT não tương ứng (ground truth)

## Quy trình

### 1. Tiền xử lý

Tiền xử lý dữ liệu là bước cần thiết để đảm bảo dữ liệu MRI và CT có chất lượng cao và phù hợp với nhau:

1. **N4 Bias Correction**: Loại bỏ nhiễu từ trường không đồng nhất trong ảnh MRI
2. **Histogram Matching**: Đồng bộ phân bố cường độ giữa các ảnh MRI
3. **Head Mask Generation**: Tạo và áp dụng mask vùng đầu để loại bỏ nhiễu nền

### 2. Tổ chức dữ liệu

Dữ liệu sau khi tiền xử lý được chia thành:
- Tập huấn luyện (70%)
- Tập kiểm định (20%)
- Tập kiểm tra (10%)

### 3. Mô hình

Mô hình sử dụng kiến trúc GAN với:

1. **Generator (U-Net)**: 
   - Chuyển đổi ảnh MRI sang ảnh CT
   - Kiến trúc U-Net với skip connections để giữ lại thông tin cấu trúc

2. **Discriminator (PatchGAN)**:
   - Phân biệt ảnh CT thật và ảnh CT mô phỏng
   - Kiến trúc PatchGAN để đánh giá chất lượng từng phần nhỏ của ảnh

### 4. Huấn luyện

Quá trình huấn luyện bao gồm:
- Adversarial training theo phương pháp GAN
- Generator tìm cách tạo ảnh CT giống thật
- Discriminator tìm cách phân biệt ảnh thật/giả
- Loss functions: Binary Cross Entropy + L1/L2 Loss

### 5. Đánh giá

Mô hình được đánh giá qua các chỉ số:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

### 6. Ứng dụng

Ảnh CT mô phỏng có thể được sử dụng để:
- Lập kế hoạch xạ trị
- Chẩn đoán ban đầu khi không có CT thật
- Giảm liều xạ cho bệnh nhân

## Cấu trúc mã nguồn

Mã nguồn được tổ chức như sau:

1. **Preprocessing**:
   - `Preprocessing 2/Preprocess_OK`: Script tiền xử lý dữ liệu

2. **Training**:
   - `src/dataloader.py`: Lớp DataLoader để tải và chia dữ liệu
   - `src/models.py`: Định nghĩa mô hình Generator và Discriminator
   - `src/gan_solver.py`: Lớp Solver để huấn luyện mô hình GAN
   - `src/utils.py`: Các hàm tiện ích
   - `src/train.py`: Script chính để điều phối quá trình huấn luyện

3. **Testing**:
   - `src/test.py`: Script để kiểm tra mô hình sau khi huấn luyện

## Yêu cầu hệ thống

- Python 3.7+
- TensorFlow 2.x
- CUDA (để sử dụng GPU)
- SimpleITK, OpenCV, NumPy, Matplotlib 