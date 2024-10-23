import os
import cv2
from mtcnn import MTCNN

def process_images_in_folders(input_folder, output_folder):
    # Tạo đối tượng MTCNN
    detector = MTCNN()
    
    # Duyệt qua tất cả các thư mục con và các file ảnh
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Kiểm tra định dạng ảnh
                # Đường dẫn đầy đủ của file ảnh
                file_path = os.path.join(root, file)
                
                # Đọc ảnh
                img = cv2.imread(file_path)
                
                # Kiểm tra xem ảnh có thể đọc được hay không
                if img is None:
                    print(f"Could not read image {file_path}")
                    continue

                # Sử dụng MTCNN để phát hiện khuôn mặt
                result = detector.detect_faces(img)
                
                if result:
                    # Lấy bounding box của khuôn mặt đầu tiên (nếu có)
                    x, y, width, height = result[0]['box']
                    # Cắt và xử lý ảnh khuôn mặt
                    face = img[y:y+height, x:x+width]
                    
                    # Tạo đường dẫn lưu ảnh đã xử lý trong thư mục output
                    relative_path = os.path.relpath(root, input_folder)  # Đường dẫn tương đối từ thư mục gốc
                    output_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
                    
                    # Đường dẫn lưu ảnh đã xử lý
                    output_file_path = os.path.join(output_dir, file)
                    
                    # Lưu ảnh đã xử lý
                    cv2.imwrite(output_file_path, face)
                    
                    print(f"Processed and saved: {output_file_path}")
                else:
                    print(f"No face detected in {file_path}")

# Sử dụng hàm:\

current_dir = os.path.dirname(__file__)  # Thư mục hiện tại chứa file Python này
input_folder = os.path.join(current_dir, '../NewData')
output_folder = os.path.join(current_dir, '../NewData')
process_images_in_folders(input_folder, output_folder)
