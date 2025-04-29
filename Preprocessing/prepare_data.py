import os
import shutil
import glob
import sys
import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import zoom
import argparse
from pathlib import Path

def create_dir_if_not_exists(directory):
    """Tạo thư mục nếu nó không tồn tại."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Đã tạo thư mục: {directory}")

def detect_file_type(file_path):
    """Phát hiện loại file dựa trên đuôi và nội dung."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() in ['.dcm', '.dicom', '']:
        try:
            pydicom.dcmread(file_path)
            return 'dicom'
        except:
            pass
    
    if ext.lower() in ['.nii', '.gz', '.nii.gz']:
        try:
            nib.load(file_path)
            return 'nifti'
        except:
            pass
            
    # Các định dạng hình ảnh khác
    if ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
        return 'image'
    
    return 'unknown'

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Chuyển đổi một thư mục DICOM sang file NIfTI."""
    import SimpleITK as sitk
    
    # Đọc series DICOM
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Lưu thành file NIfTI
    sitk.WriteImage(image, output_path)
    print(f"Đã chuyển đổi DICOM sang NIfTI: {output_path}")
    return output_path

def convert_images_to_nifti(image_dir, output_path):
    """Chuyển đổi một thư mục hình ảnh 2D sang file NIfTI 3D."""
    import cv2
    
    # Lấy tất cả file hình ảnh
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not all_images:
        print(f"Không tìm thấy hình ảnh trong thư mục: {image_dir}")
        return None
        
    # Sắp xếp theo tên file
    all_images.sort()
    
    # Đọc hình ảnh đầu tiên để lấy kích thước
    img = cv2.imread(all_images[0], cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    # Tạo khối 3D
    volume = np.zeros((len(all_images), height, width), dtype=np.int16)
    
    # Đọc và thêm từng hình ảnh vào khối 3D
    for i, img_path in enumerate(all_images):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.shape != (height, width):
            img = cv2.resize(img, (width, height))
        volume[i] = img
    
    # Tạo NIfTI từ khối 3D
    nifti_img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti_img, output_path)
    print(f"Đã chuyển đổi {len(all_images)} hình ảnh sang NIfTI: {output_path}")
    return output_path

def prepare_data(mr_source, ct_source, output_dir, dataset_name='brain01_test'):
    """Chuẩn bị dữ liệu từ nguồn và tổ chức vào cấu trúc thư mục đúng."""
    # Tạo thư mục đích
    mr_target_dir = os.path.join(output_dir, dataset_name, 'MR')
    ct_target_dir = os.path.join(output_dir, dataset_name, 'CT')
    
    create_dir_if_not_exists(mr_target_dir)
    create_dir_if_not_exists(ct_target_dir)
    
    # Xử lý dữ liệu MRI
    mr_files_processed = process_directory(mr_source, mr_target_dir, 'MR')
    
    # Xử lý dữ liệu CT
    ct_files_processed = process_directory(ct_source, ct_target_dir, 'CT')
    
    print(f"\nĐã xử lý {len(mr_files_processed)} file MR và {len(ct_files_processed)} file CT")
    print(f"Dữ liệu đã được lưu vào thư mục: {output_dir}")
    
    # Kiểm tra xem có cùng số lượng file không
    if len(mr_files_processed) != len(ct_files_processed):
        print(f"Cảnh báo: Số lượng file MR ({len(mr_files_processed)}) khác với số lượng file CT ({len(ct_files_processed)})")
    
    return mr_files_processed, ct_files_processed

def process_directory(source_dir, target_dir, prefix):
    """Xử lý một thư mục nguồn, tìm và chuyển đổi các file sang định dạng .nii.gz."""
    processed_files = []
    
    if not os.path.exists(source_dir):
        print(f"Thư mục nguồn không tồn tại: {source_dir}")
        return processed_files
    
    # Trường hợp 1: Kiểm tra xem nguồn có phải là một thư mục con DICOM
    if os.path.isdir(source_dir) and all_files_are_dicom(source_dir):
        output_path = os.path.join(target_dir, f"{prefix}_0001.nii.gz")
        convert_dicom_to_nifti(source_dir, output_path)
        processed_files.append(output_path)
        return processed_files
    
    # Trường hợp 2: Kiểm tra các thư mục con
    patient_count = 0
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        
        if os.path.isdir(item_path):
            # Kiểm tra xem thư mục con có phải DICOM không
            if all_files_are_dicom(item_path):
                patient_count += 1
                output_path = os.path.join(target_dir, f"{prefix}_{patient_count:04d}.nii.gz")
                convert_dicom_to_nifti(item_path, output_path)
                processed_files.append(output_path)
            # Kiểm tra xem có phải thư mục hình ảnh 2D không
            elif any_image_files(item_path):
                patient_count += 1
                output_path = os.path.join(target_dir, f"{prefix}_{patient_count:04d}.nii.gz")
                convert_images_to_nifti(item_path, output_path)
                processed_files.append(output_path)
            else:
                # Thử xử lý đệ quy
                sub_processed = process_directory(item_path, target_dir, f"{prefix}_{item}")
                processed_files.extend(sub_processed)
        else:
            # Xử lý một file riêng lẻ
            file_type = detect_file_type(item_path)
            
            if file_type == 'nifti':
                # Sao chép file NIfTI trực tiếp
                patient_count += 1
                output_path = os.path.join(target_dir, f"{prefix}_{patient_count:04d}.nii.gz")
                shutil.copy(item_path, output_path)
                processed_files.append(output_path)
                print(f"Đã sao chép file NIfTI: {output_path}")
            elif file_type == 'dicom':
                # Chuyển đổi file DICOM đơn lẻ
                patient_count += 1
                output_path = os.path.join(target_dir, f"{prefix}_{patient_count:04d}.nii.gz")
                try:
                    dcm = pydicom.dcmread(item_path)
                    pixel_array = dcm.pixel_array
                    nifti_img = nib.Nifti1Image(pixel_array, np.eye(4))
                    nib.save(nifti_img, output_path)
                    processed_files.append(output_path)
                    print(f"Đã chuyển đổi DICOM đơn lẻ sang NIfTI: {output_path}")
                except Exception as e:
                    print(f"Lỗi khi chuyển đổi DICOM đơn lẻ: {e}")
            elif file_type == 'image':
                # Nếu là hình ảnh thông thường, tạo thư mục tạm và xử lý
                temp_dir = os.path.join(target_dir, 'temp_images')
                os.makedirs(temp_dir, exist_ok=True)
                shutil.copy(item_path, os.path.join(temp_dir, item))
                
                patient_count += 1
                output_path = os.path.join(target_dir, f"{prefix}_{patient_count:04d}.nii.gz")
                convert_images_to_nifti(temp_dir, output_path)
                processed_files.append(output_path)
                
                # Xóa thư mục tạm
                shutil.rmtree(temp_dir)
    
    # Trường hợp 3: Nếu thư mục chứa các file hình ảnh
    if patient_count == 0 and any_image_files(source_dir):
        patient_count += 1
        output_path = os.path.join(target_dir, f"{prefix}_{patient_count:04d}.nii.gz")
        convert_images_to_nifti(source_dir, output_path)
        processed_files.append(output_path)
    
    return processed_files

def all_files_are_dicom(directory):
    """Kiểm tra xem tất cả các file trong thư mục có phải là DICOM không."""
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return False
    
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not all_files:
        return False
    
    # Kiểm tra một vài file đầu tiên
    sample_size = min(10, len(all_files))
    for i in range(sample_size):
        file_path = os.path.join(directory, all_files[i])
        try:
            pydicom.dcmread(file_path)
        except:
            return False
    
    return True

def any_image_files(directory):
    """Kiểm tra xem thư mục có chứa bất kỳ file hình ảnh nào không."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    for ext in extensions:
        if glob.glob(os.path.join(directory, ext)):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị dữ liệu MRI và CT')
    parser.add_argument('--mr_source', default='D:\\Documents\\Brain - Copy\\Image Train',
                        help='Thư mục chứa dữ liệu MRI nguồn')
    parser.add_argument('--ct_source', default='D:\\Documents\\Brain - Copy\\Image Label',
                        help='Thư mục chứa dữ liệu CT nguồn')
    parser.add_argument('--test_source', default='D:\\Documents\\Brain - Copy\\Image Test',
                        help='Thư mục chứa dữ liệu test')
    parser.add_argument('--output_dir', default='../../Data',
                        help='Thư mục đích để lưu dữ liệu đã xử lý')
    parser.add_argument('--dataset_name', default='brain01_test',
                        help='Tên tập dữ liệu (tên thư mục con trong thư mục đích)')
    
    args = parser.parse_args()
    
    # Chuẩn bị dữ liệu
    print(f"Bắt đầu chuẩn bị dữ liệu từ:")
    print(f"MRI: {args.mr_source}")
    print(f"CT: {args.ct_source}")
    print(f"Test: {args.test_source}")
    print(f"Lưu vào: {os.path.join(args.output_dir, args.dataset_name)}")
    
    try:
        # Xử lý dữ liệu huấn luyện
        mr_files, ct_files = prepare_data(args.mr_source, args.ct_source, args.output_dir, args.dataset_name)
        
        # Xử lý dữ liệu test
        test_mr_files, test_ct_files = prepare_data(args.test_source, args.test_source, args.output_dir, f"{args.dataset_name}_test")
        
        if mr_files and ct_files:
            print("\nDữ liệu huấn luyện đã được chuẩn bị thành công!")
            print(f"Đã xử lý {len(mr_files)} file MR và {len(ct_files)} file CT")
            
        if test_mr_files and test_ct_files:
            print("\nDữ liệu test đã được chuẩn bị thành công!")
            print(f"Đã xử lý {len(test_mr_files)} file test MR và {len(test_ct_files)} file test CT")
            
        print(f"\nDữ liệu đã được lưu vào thư mục: {args.output_dir}")
        print("\nĐể kiểm tra dữ liệu, hãy chạy lệnh:")
        print(f"python test_nifti.py")
    except Exception as e:
        print(f"Lỗi khi chuẩn bị dữ liệu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 