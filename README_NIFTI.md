# MRI-to-CT-DCNN-TensorFlow với dữ liệu NIFTI

Phiên bản này đã được cập nhật để hỗ trợ định dạng dữ liệu NIFTI (.nii.gz) thường được sử dụng trong dữ liệu y tế.

## Yêu cầu

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục dữ liệu

Đặt dữ liệu NIFTI vào cấu trúc thư mục sau:

```
Data/
└── brain01/
    ├── MR/
    │   ├── patient1.nii.gz
    │   ├── patient2.nii.gz
    │   └── ...
    ├── CT/
    │   ├── patient1.nii.gz
    │   ├── patient2.nii.gz
    │   └── ...
    └── MASK/ (tùy chọn)
        ├── patient1.nii.gz
        ├── patient2.nii.gz
        └── ...
```

Lưu ý: 
- Tên file MR và CT phải tương ứng với nhau.
- Thư mục MASK là tùy chọn. Nếu không có, mặt nạ sẽ được tạo tự động từ ảnh CT.

## Kiểm tra dữ liệu NIFTI

Để kiểm tra xem dữ liệu NIFTI có hoạt động không:

```bash
cd src
python test_nifti.py
```

Nếu bạn muốn kiểm tra với file NIFTI cụ thể:

```bash
python test_nifti.py --nifti_paths=/path/to/mr.nii.gz,/path/to/ct.nii.gz,/path/to/mask.nii.gz
```

## Huấn luyện mô hình với dữ liệu NIFTI

```bash
cd src
python main.py --is_train --dataset brain01 --use_nifti
```

## Kiểm tra mô hình với dữ liệu NIFTI

```bash
cd src
python main.py --dataset brain01 --use_nifti --load_model YYYYMMDD-HHMM
```

## Tùy chọn dòng lệnh

```
--gpu_index: Chỉ số GPU nếu bạn có nhiều GPU
--batch_size: Kích thước batch (mặc định: 8)
--learning_rate: Tốc độ học (mặc định: 1e-3)
--epoch: Số epoch huấn luyện (mặc định: 600)
--use_nifti: Sử dụng định dạng NIFTI (mặc định: True)
--dataset: Tên tập dữ liệu (mặc định: brain01)
```

## Xử lý dữ liệu NIFTI

Quá trình xử lý dữ liệu NIFTI bao gồm các bước sau:
1. Tải các tệp NIFTI (MR, CT, và Mask tùy chọn)
2. Trích xuất các lát cắt 2D từ khối dữ liệu 3D (theo trục được chỉ định)
3. Tiền xử lý mỗi lát cắt (đổi kích thước thành 256×256, chuẩn hóa)
4. Tạo mặt nạ cho mỗi lát cắt CT nếu không có mặt nạ được cung cấp

## Giải quyết vấn đề

Nếu gặp lỗi khi chạy:

1. Đảm bảo cấu trúc thư mục dữ liệu chính xác
2. Kiểm tra thư viện `nibabel` đã được cài đặt 
3. Kiểm tra phiên bản TensorFlow (yêu cầu phiên bản 1.x)
4. Kiểm tra định dạng file NIFTI có hợp lệ

## Lưu ý

- Mã nguồn này sử dụng TensorFlow 1.15, không tương thích với TensorFlow 2.x
- Mô hình cần file trọng số VGG được tải trước (pre-trained weights). Nếu không có, mô hình sẽ gặp lỗi khi khởi tạo. 